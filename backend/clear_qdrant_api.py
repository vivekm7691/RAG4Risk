"""Alternative script to clear Qdrant collection via API

This script uses the API to delete all documents individually.
Use this if you cannot run clear_qdrant.py in Docker.

Usage:
    python backend/clear_qdrant_api.py
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"


def clear_via_api():
    """Clear vector store by deleting all documents via API"""
    print("=" * 60)
    print("Clear Qdrant Vector Store via API")
    print("=" * 60)
    
    try:
        # Step 1: List all documents
        print("\nStep 1: Listing all documents...")
        response = requests.get(f"{API_BASE}/documents", timeout=30)
        
        if response.status_code != 200:
            print(f"[ERROR] Failed to list documents: {response.status_code}")
            print(f"Response: {response.text}")
            return False
        
        documents = response.json()
        print(f"Found {len(documents)} documents")
        
        if len(documents) == 0:
            print("Vector store is already empty.")
            return True
        
        # Step 2: Delete each document
        print(f"\nStep 2: Deleting {len(documents)} documents...")
        deleted_count = 0
        failed_count = 0
        
        for doc in documents:
            doc_id = doc.get("document_id")
            doc_name = doc.get("file_name", "Unknown")
            
            if not doc_id:
                print(f"[WARN] Document missing ID, skipping: {doc_name}")
                continue
            
            try:
                delete_response = requests.delete(
                    f"{API_BASE}/documents/{doc_id}",
                    timeout=30
                )
                
                if delete_response.status_code == 204:
                    deleted_count += 1
                    print(f"  [OK] Deleted: {doc_name} ({doc_id[:8]}...)")
                else:
                    failed_count += 1
                    print(f"  [FAIL] Failed to delete: {doc_name} ({doc_id[:8]}...) - Status: {delete_response.status_code}")
            except Exception as e:
                failed_count += 1
                print(f"  [ERROR] Error deleting {doc_name}: {str(e)}")
        
        # Step 3: Verify
        print(f"\nStep 3: Verifying deletion...")
        verify_response = requests.get(f"{API_BASE}/documents", timeout=30)
        
        if verify_response.status_code == 200:
            remaining_docs = verify_response.json()
            remaining_count = len(remaining_docs) if isinstance(remaining_docs, list) else 0
            
            print(f"\n{'=' * 60}")
            print("SUMMARY")
            print("=" * 60)
            print(f"Documents deleted: {deleted_count}")
            print(f"Documents failed: {failed_count}")
            print(f"Documents remaining: {remaining_count}")
            
            if remaining_count == 0:
                print("\n[SUCCESS] Vector store cleared successfully!")
                return True
            else:
                print(f"\n[WARNING] {remaining_count} documents still remain")
                return False
        else:
            print(f"[WARN] Could not verify deletion: {verify_response.status_code}")
            return deleted_count > 0
            
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to backend. Is the server running?")
        print("  Start with: docker-compose up -d")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = clear_via_api()
    sys.exit(0 if success else 1)

