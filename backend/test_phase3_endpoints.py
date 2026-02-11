"""Test script for Phase 3 API endpoints"""

import requests
import json
import os
from pathlib import Path

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

def test_health():
    """Test health check endpoint"""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("[PASS] Health check passed\n")

def test_root():
    """Test root endpoint"""
    print("\n=== Testing Root Endpoint ===")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    print("[PASS] Root endpoint passed\n")

def test_list_documents_empty():
    """Test listing documents when empty"""
    print("\n=== Testing List Documents (Empty) ===")
    response = requests.get(f"{API_BASE}/documents")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    print("[PASS] List documents (empty) passed\n")

def test_upload_document_invalid():
    """Test document upload with invalid data"""
    print("\n=== Testing Document Upload (Invalid) ===")
    
    # Test with missing file
    response = requests.post(
        f"{API_BASE}/documents/upload",
        data={"project_name": "Test Project", "document_type": "statement of work"}
    )
    print(f"Status (missing file): {response.status_code}")
    assert response.status_code == 422  # Validation error
    print("[PASS] Invalid upload (missing file) handled correctly")
    
    # Test with invalid document type
    files = {"file": ("test.txt", b"test content", "text/plain")}
    data = {"project_name": "Test Project", "document_type": "invalid_type"}
    response = requests.post(f"{API_BASE}/documents/upload", files=files, data=data)
    print(f"Status (invalid type): {response.status_code}")
    assert response.status_code == 400
    print("[PASS] Invalid upload (invalid type) handled correctly\n")

def test_query_empty():
    """Test query endpoint with empty database"""
    print("\n=== Testing Query (Empty Database) ===")
    query_data = {
        "query": "What is the project about?",
        "project_name": "Test Project",
        "top_k": 5
    }
    response = requests.post(f"{API_BASE}/query", json=query_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    assert "answer" in result
    assert "sources" in result
    print("[PASS] Query (empty) passed\n")

def test_query_with_filters():
    """Test query endpoint with filters"""
    print("\n=== Testing Query with Filters ===")
    query_data = {
        "query": "What are the high severity risks?",
        "project_name": "Test Project",
        "top_k": 5,
        "filters": {
            "severity": "High",
            "status": "Open"
        }
    }
    response = requests.post(f"{API_BASE}/query", json=query_data)
    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")
    assert response.status_code == 200
    print("[PASS] Query with filters passed\n")

def test_query_streaming():
    """Test streaming query endpoint"""
    print("\n=== Testing Streaming Query ===")
    query_data = {
        "query": "What is the project about?",
        "project_name": "Test Project",
        "top_k": 5
    }
    
    response = requests.post(
        f"{API_BASE}/query/stream",
        json=query_data,
        stream=True,
        headers={"Accept": "text/event-stream"}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("Content-Type", "")
    
    # Parse SSE stream
    chunks_received = []
    sources_received = False
    done_received = False
    
    try:
        for line in response.iter_lines():
            if not line:
                continue
            
            # SSE format: "data: {...}\n"
            if line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8")
                try:
                    data = json.loads(data_str)
                    msg_type = data.get("type")
                    
                    if msg_type == "sources":
                        sources_received = True
                        print(f"  Received sources: {len(data.get('sources', []))} sources")
                    elif msg_type == "chunk":
                        chunks_received.append(data.get("text", ""))
                        print(f"  Received chunk: {data.get('text', '')[:50]}...")
                    elif msg_type == "done":
                        done_received = True
                        print(f"  Stream completed in {data.get('total_time', 0):.3f}s")
                    elif msg_type == "error":
                        print(f"  Error received: {data.get('message', 'Unknown error')}")
                        break
                except json.JSONDecodeError:
                    print(f"  Failed to parse JSON: {data_str[:50]}")
    
    except Exception as e:
        print(f"  Error reading stream: {e}")
    
    assert sources_received, "Sources message should be received"
    assert done_received, "Done message should be received"
    print(f"[PASS] Streaming query passed ({len(chunks_received)} chunks received)\n")

def test_api_docs():
    """Test API documentation endpoint"""
    print("\n=== Testing API Documentation ===")
    response = requests.get(f"{BASE_URL}/docs")
    print(f"Status: {response.status_code}")
    assert response.status_code == 200
    print("[PASS] API docs accessible\n")

def test_openapi_spec():
    """Test OpenAPI specification endpoint"""
    print("\n=== Testing OpenAPI Spec ===")
    response = requests.get(f"{BASE_URL}/openapi.json")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        spec = response.json()
        print(f"OpenAPI version: {spec.get('openapi', 'N/A')}")
        print(f"Endpoints found: {len(spec.get('paths', {}))}")
        print("Available endpoints:")
        for path in spec.get('paths', {}).keys():
            print(f"  - {path}")
    assert response.status_code == 200
    print("[PASS] OpenAPI spec accessible\n")

def main():
    """Run all tests"""
    print("=" * 60)
    print("Phase 3 API Endpoints Test Suite")
    print("=" * 60)
    
    try:
        test_health()
        test_root()
        test_list_documents_empty()
        test_upload_document_invalid()
        test_query_empty()
        test_query_with_filters()
        test_query_streaming()
        test_api_docs()
        test_openapi_spec()
        
        print("=" * 60)
        print("All tests passed! [SUCCESS]")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Upload a test document using POST /api/documents/upload")
        print("2. Query the uploaded document using POST /api/query")
        print("3. View API documentation at http://localhost:8000/docs")
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        return 1
    except requests.exceptions.ConnectionError:
        print("\n[FAIL] Cannot connect to backend. Is the server running?")
        print("  Start with: docker-compose up -d")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())

