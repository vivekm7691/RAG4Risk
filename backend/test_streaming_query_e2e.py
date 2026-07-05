"""End-to-end test script for streaming query endpoint with full investigation details

This script tests the streaming query endpoint with all optional parameters and displays:
- Query parameters
- Full chunks retrieved from vector search
- Final prompt sent to LLM
- Source citations
- Complete LLM response (untrimmed)
- Timing information
"""

import requests
import json
import sys
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"


def format_context(chunks: List[Dict[str, Any]], project_name: Optional[str] = None) -> str:
    """
    Format retrieved chunks as context text (same logic as RAGService._format_context)
    
    Args:
        chunks: List of chunk dictionaries with 'text' and 'metadata'
        project_name: Optional project name for context labeling
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")
        
        # Build source citation
        doc_name = metadata.get("file_name", "Unknown document")
        doc_type = metadata.get("document_type", "")
        chunk_id = metadata.get("chunk_id", "")
        
        # For Excel documents, include row and sheet info
        source_info = f"[Source {idx}: {doc_name}"
        if doc_type in ["risk register", "issue log"]:
            row_num = metadata.get("row_number")
            sheet_name = metadata.get("sheet_name")
            if row_num:
                source_info += f", Row {row_num}"
            if sheet_name:
                source_info += f", Sheet: {sheet_name}"
        source_info += f"]"
        
        context_parts.append(f"{source_info}\n{chunk_text}")
    
    return "\n\n---\n\n".join(context_parts)


def build_rag_prompt(query: str, context: str, project_name: Optional[str] = None) -> str:
    """
    Build the RAG prompt with context injection (same logic as RAGService._build_prompt)
    
    Args:
        query: User query
        context: Formatted context from retrieved chunks
        project_name: Optional project name
        
    Returns:
        Complete prompt string
    """
    project_context = ""
    if project_name:
        project_context = f" for the project '{project_name}'"
    
    prompt = f"""You are a helpful assistant that answers questions based on provided context documents{project_context}.

Context from documents:
{context}

Question: {query}

Instructions:
- Answer the question based ONLY on the information provided in the context above.
- If the context doesn't contain enough information to answer the question, say so clearly.
- Be concise and accurate.
- Cite specific sources when referencing information (use the [Source X: ...] markers from the context).
- If the question is about risks or issues, provide specific details from the context including severity, status, and other relevant metadata.

Answer:"""
    
    return prompt


def get_vector_search_results(
    query: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Get full chunks from vector search for investigation
    
    Uses the diagnostics endpoint with include_chunks=True to get full chunk text.
    Falls back to preliminary query if diagnostics endpoint fails.
    
    Args:
        query: Query text
        project_name: Optional project name filter
        top_k: Number of results
        filters: Optional metadata filters
        
    Returns:
        Tuple of (chunks list with text and metadata, error message if any)
    """
    try:
        # Try diagnostics endpoint with include_chunks=True first
        params = {
            "query": query,
            "top_k": top_k,
            "include_chunks": "true"
        }
        
        if project_name:
            params["project_name"] = project_name
        
        # Note: diagnostics endpoint doesn't support filters yet
        # If filters are provided, we'll fall back to query endpoint
        
        if not filters:
            try:
                diagnostics_response = requests.get(
                    f"{API_BASE}/diagnostics/vector-search",
                    params=params,
                    timeout=30
                )
                
                if diagnostics_response.status_code == 200:
                    diagnostics_data = diagnostics_response.json()
                    if diagnostics_data.get("success") and diagnostics_data.get("chunks"):
                        # Convert diagnostics chunks format to our format
                        chunks = []
                        for chunk_data in diagnostics_data["chunks"]:
                            chunk = {
                                "text": chunk_data.get("text", ""),
                                "metadata": chunk_data.get("metadata", {})
                            }
                            chunks.append(chunk)
                        return chunks, None
            except:
                # Fall through to query endpoint approach
                pass
        
        # Fallback: Make a preliminary non-streaming query to get source information
        # This gives us metadata but not full chunk text
        query_data = {
            "query": query,
            "top_k": top_k
        }
        
        if project_name:
            query_data["project_name"] = project_name
        
        if filters:
            query_data["filters"] = filters
        
        query_response = requests.post(
            f"{API_BASE}/query",
            json=query_data,
            timeout=300
        )
        
        if query_response.status_code != 200:
            return [], f"Failed to get preliminary query results: {query_response.status_code}"
        
        result = query_response.json()
        sources = result.get("sources", [])
        
        # Reconstruct chunks from sources
        # Note: Full chunk text is not available in the query endpoint response
        chunks = []
        for source in sources:
            chunk = {
                "text": "[Full chunk text not available via query endpoint - use diagnostics endpoint with include_chunks=true]",
                "metadata": {
                    "document_id": source.get("document_id", ""),
                    "file_name": source.get("document_name", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "project_name": source.get("project_name", ""),
                    "document_type": source.get("document_type", ""),
                    "relevance_score": source.get("relevance_score"),
                    "row_number": source.get("row_number"),
                    "sheet_name": source.get("sheet_name")
                }
            }
            chunks.append(chunk)
        
        return chunks, None
        
    except requests.exceptions.RequestException as e:
        return [], f"Network error: {str(e)}"
    except Exception as e:
        return [], f"Unexpected error: {str(e)}"


def get_chunks_from_streaming_sources(sources_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract chunk information from streaming sources message
    
    Note: Full chunk text is not included in the sources message.
    This function extracts available metadata from sources.
    
    Args:
        sources_data: Sources data from streaming response
        
    Returns:
        List of chunk dictionaries with metadata
    """
    chunks = []
    sources = sources_data.get("sources", [])
    
    for idx, source in enumerate(sources, 1):
        chunk = {
            "text": "[Full chunk text not available via API - retrieved from vector store during search]",
            "metadata": {
                "document_id": source.get("document_id", ""),
                "file_name": source.get("document_name", ""),
                "chunk_id": source.get("chunk_id", ""),
                "project_name": source.get("project_name", ""),
                "document_type": source.get("document_type", ""),
                "relevance_score": source.get("relevance_score"),
                "row_number": source.get("row_number"),
                "sheet_name": source.get("sheet_name")
            }
        }
        chunks.append(chunk)
    
    return chunks


def display_query_parameters(query_data: Dict[str, Any]):
    """Display query parameters that were sent"""
    print("\n" + "=" * 80)
    print("SECTION 1: QUERY PARAMETERS")
    print("=" * 80)
    print(json.dumps(query_data, indent=2))
    print()


def display_chunks(chunks: List[Dict[str, Any]]):
    """Display full chunks retrieved from vector search"""
    print("\n" + "=" * 80)
    print("SECTION 2: FULL CHUNKS RETRIEVED (for investigation)")
    print("=" * 80)
    
    if not chunks:
        print("No chunks retrieved.")
        return
    
    print(f"Total chunks: {len(chunks)}")
    print("\nNOTE: Full chunk text is not available via API endpoints.")
    print("      Chunk text is retrieved from vector store during search but not exposed in API responses.")
    print("      Metadata below shows what chunks were retrieved and their properties.\n")
    
    for idx, chunk in enumerate(chunks, 1):
        print(f"\n--- Chunk {idx} ---")
        text = chunk.get('text', 'N/A')
        if text.startswith('[') and 'not available' in text:
            print(f"Text: {text}")
        else:
            print(f"Text: {text}")
        print(f"Metadata:")
        metadata = chunk.get("metadata", {})
        for key, value in metadata.items():
            if value is not None:
                print(f"  {key}: {value}")
    print()


def display_prompt(prompt: str):
    """Display final prompt sent to LLM"""
    print("\n" + "=" * 80)
    print("SECTION 3: FINAL PROMPT SENT TO LLM (for investigation)")
    print("=" * 80)
    print(prompt)
    print()


def display_sources(sources: List[Dict[str, Any]]):
    """Display source citations from streaming response"""
    print("\n" + "=" * 80)
    print("SECTION 4: SOURCE CITATIONS")
    print("=" * 80)
    
    if not sources:
        print("No sources found.")
        return
    
    print(f"Total sources: {len(sources)}\n")
    
    for idx, source in enumerate(sources, 1):
        print(f"\n--- Source {idx} ---")
        print(f"Document: {source.get('document_name', 'Unknown')}")
        print(f"Document ID: {source.get('document_id', 'N/A')}")
        print(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
        print(f"Project: {source.get('project_name', 'N/A')}")
        print(f"Document Type: {source.get('document_type', 'N/A')}")
        if source.get('relevance_score') is not None:
            print(f"Relevance Score: {source.get('relevance_score'):.4f}")
        if source.get('row_number') is not None:
            print(f"Row Number: {source.get('row_number')}")
        if source.get('sheet_name'):
            print(f"Sheet Name: {source.get('sheet_name')}")
    print()


def display_response(response_text: str):
    """Display complete LLM response (untrimmed)"""
    print("\n" + "=" * 80)
    print("SECTION 5: LLM RESPONSE (complete, untrimmed)")
    print("=" * 80)
    print(response_text)
    print()


def display_timings(timings: Dict[str, Optional[float]]):
    """Display timing information"""
    print("\n" + "=" * 80)
    print("SECTION 6: TIMING INFORMATION")
    print("=" * 80)
    
    if timings:
        for key, value in timings.items():
            if value is not None:
                print(f"{key.replace('_', ' ').title()}: {value:.3f}s")
            else:
                print(f"{key.replace('_', ' ').title()}: N/A")
    else:
        print("No timing information available.")
    print()


def parse_sse_stream(response) -> tuple[Dict[str, Any], str, Dict[str, Any], Optional[str]]:
    """
    Parse SSE stream and extract all data
    
    Returns:
        Tuple of (sources_data, response_text, done_data, error_message)
    """
    sources_data = None
    response_chunks = []
    done_data = None
    error_message = None
    
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
                        sources_data = data
                    elif msg_type == "chunk":
                        response_chunks.append(data.get("text", ""))
                    elif msg_type == "done":
                        done_data = data
                    elif msg_type == "error":
                        error_message = data.get("message", "Unknown error")
                        break
                        
                except json.JSONDecodeError:
                    print(f"  [WARN] Failed to parse JSON: {data_str[:50]}...")
                    continue
    
    except Exception as e:
        error_message = f"Error reading stream: {str(e)}"
    
    response_text = "".join(response_chunks)
    return sources_data, response_text, done_data, error_message


def get_available_models() -> List[str]:
    """
    Fetch available Ollama models from the API
    
    Returns:
        List of available model names
    """
    try:
        response = requests.get(f"{API_BASE}/diagnostics/ollama-models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("models"):
                return [model.get("name", "") for model in data.get("models", []) if model.get("name")]
        return []
    except Exception as e:
        print(f"[WARN] Could not fetch available models: {e}")
        return []


def test_streaming_query_e2e(
    query: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None
):
    """
    Main test function for end-to-end streaming query test
    
    Args:
        query: User query text
        project_name: Optional project name filter
        top_k: Number of results to return
        filters: Optional metadata filters
    """
    print("\n" + "=" * 80)
    print("END-TO-END STREAMING QUERY TEST")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Build query data
    query_data = {
        "query": query,
        "top_k": top_k
    }
    
    if project_name:
        query_data["project_name"] = project_name
    
    if filters:
        query_data["filters"] = filters
    
    if model:
        query_data["model"] = model
    
    # Display query parameters
    display_query_parameters(query_data)
    
    # Get chunks for investigation (before streaming call)
    print("Retrieving chunks for investigation...")
    chunks, chunks_error = get_vector_search_results(query, project_name, top_k, filters)
    
    if chunks_error:
        print(f"[WARN] Could not retrieve chunks separately: {chunks_error}")
        print("Chunks will be extracted from streaming response sources.\n")
    
    # Make streaming query call
    print("Making streaming query call...")
    try:
        response = requests.post(
            f"{API_BASE}/query/stream",
            json=query_data,
            stream=True,
            headers={"Accept": "text/event-stream"},
            timeout=3100  # Must exceed backend OLLAMA_TIMEOUT (3000s) for long RAG LLM runs
        )
        
        if response.status_code != 200:
            print(f"\n[ERROR] Query failed with status {response.status_code}")
            try:
                error_data = response.json()
                print(f"Error: {json.dumps(error_data, indent=2)}")
            except:
                print(f"Error: {response.text}")
            return
        
        # Parse SSE stream
        sources_data, response_text, done_data, error_message = parse_sse_stream(response)
        
        # Extract chunks from sources if we didn't get them separately
        if not chunks and sources_data:
            chunks = get_chunks_from_streaming_sources(sources_data)
        
        # Display chunks
        display_chunks(chunks)
        
        # Reconstruct and display prompt
        if chunks:
            # Check if we have actual chunk text or just placeholders
            has_actual_text = any(
                chunk.get("text", "").startswith("[") and "not available" not in chunk.get("text", "")
                for chunk in chunks
            ) or all(
                not (chunk.get("text", "").startswith("[") and "not available" in chunk.get("text", ""))
                for chunk in chunks
            )
            
            if has_actual_text or any(not chunk.get("text", "").startswith("[") for chunk in chunks):
                context_text = format_context(chunks, project_name)
                prompt = build_rag_prompt(query, context_text, project_name)
                display_prompt(prompt)
            else:
                # Chunks don't have actual text, show note
                print("\n" + "=" * 80)
                print("SECTION 3: FINAL PROMPT SENT TO LLM (for investigation)")
                print("=" * 80)
                print("NOTE: Full chunk text is not available, so the exact prompt cannot be reconstructed.")
                print("      The prompt would be built using the chunk text retrieved from vector store.")
                print("      Below is a reconstructed prompt using available metadata:\n")
                context_text = format_context(chunks, project_name)
                prompt = build_rag_prompt(query, context_text, project_name)
                print(prompt)
                print()
        else:
            print("\n" + "=" * 80)
            print("SECTION 3: FINAL PROMPT SENT TO LLM (for investigation)")
            print("=" * 80)
            print("[No chunks available to build prompt]")
            print()
        
        # Display sources
        if sources_data:
            display_sources(sources_data.get("sources", []))
        
        # Display response
        display_response(response_text)
        
        # Display timings
        if done_data:
            display_timings(done_data.get("timings", {}))
        else:
            display_timings({})
        
        # Display summary
        print("\n" + "=" * 80)
        print("SECTION 7: SUMMARY")
        print("=" * 80)
        
        if error_message:
            print(f"[ERROR] {error_message}")
            print("Status: FAILED")
        else:
            print("Status: SUCCESS")
            if sources_data:
                print(f"Sources found: {len(sources_data.get('sources', []))}")
            print(f"Response length: {len(response_text)} characters")
            if done_data:
                print(f"Total time: {done_data.get('total_time', 0):.3f}s")
        print()
        
    except requests.exceptions.Timeout:
        print("\n[ERROR] Request timed out (query may take 200+ seconds)")
        print("Status: TIMEOUT")
    except requests.exceptions.ConnectionError:
        print("\n[ERROR] Cannot connect to backend. Is the server running?")
        print("  Start with: docker-compose up -d")
        print("Status: CONNECTION ERROR")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Status: FAILED")


def get_filters_from_args(args) -> Optional[Dict[str, Any]]:
    """Build filters dictionary from command-line arguments"""
    filters = {}
    
    if args.severity:
        filters["severity"] = args.severity
    if args.status:
        filters["status"] = args.status
    if args.category:
        filters["category"] = args.category
    if args.owner:
        filters["owner"] = args.owner
    if args.start_date or args.end_date:
        filters["date_range"] = {}
        if args.start_date:
            filters["date_range"]["start_date"] = args.start_date
        if args.end_date:
            filters["date_range"]["end_date"] = args.end_date
    
    return filters if filters else None


def interactive_mode():
    """Interactive mode for parameter input"""
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - Enter Query Parameters")
    print("=" * 80)
    
    query = input("\nQuery (required): ").strip()
    if not query:
        print("Error: Query is required")
        return
    
    project_name = input("Project name (optional, press Enter to skip): ").strip()
    project_name = project_name if project_name else None
    
    top_k_str = input("Top K (optional, default 5, press Enter for default): ").strip()
    top_k = int(top_k_str) if top_k_str else 5
    
    # Fetch and display available models
    print("\nFetching available Ollama models...")
    available_models = get_available_models()
    model = None
    if available_models:
        print(f"\nAvailable models ({len(available_models)}):")
        for idx, model_name in enumerate(available_models, 1):
            print(f"  {idx}. {model_name}")
        print("  0. Use default model (from configuration)")
        model_choice = input("\nSelect model (enter number or model name, press Enter for default): ").strip()
        if model_choice:
            # Try to parse as number
            try:
                choice_num = int(model_choice)
                if 1 <= choice_num <= len(available_models):
                    model = available_models[choice_num - 1]
                elif choice_num == 0:
                    model = None
            except ValueError:
                # Not a number, treat as model name
                if model_choice in available_models:
                    model = model_choice
                else:
                    print(f"[WARN] Model '{model_choice}' not in available models list, but will try anyway.")
                    model = model_choice
    else:
        model_input = input("Model name (optional, press Enter to use default): ").strip()
        model = model_input if model_input else None
    
    print("\nExcel Filters (optional, press Enter to skip each):")
    severity = input("  Severity: ").strip()
    severity = severity if severity else None
    
    status = input("  Status: ").strip()
    status = status if status else None
    
    category = input("  Category: ").strip()
    category = category if category else None
    
    owner = input("  Owner: ").strip()
    owner = owner if owner else None
    
    start_date = input("  Start date (ISO format, e.g., 2024-01-01T00:00:00Z): ").strip()
    start_date = start_date if start_date else None
    
    end_date = input("  End date (ISO format, e.g., 2024-12-31T23:59:59Z): ").strip()
    end_date = end_date if end_date else None
    
    filters = {}
    if severity:
        filters["severity"] = severity
    if status:
        filters["status"] = status
    if category:
        filters["category"] = category
    if owner:
        filters["owner"] = owner
    if start_date or end_date:
        filters["date_range"] = {}
        if start_date:
            filters["date_range"]["start_date"] = start_date
        if end_date:
            filters["date_range"]["end_date"] = end_date
    
    filters = filters if filters else None
    
    test_streaming_query_e2e(query, project_name, top_k, filters, model)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="End-to-end test script for streaming query endpoint with full investigation details"
    )
    
    parser.add_argument("--query", "-q", required=False, help="User query text")
    parser.add_argument("--project-name", "-p", help="Project name filter")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--model", "-m", help="Ollama model name to use (e.g., 'llama3.2:3b', 'mistral:7b'). Use --list-models to see available models.")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")
    
    # Filter arguments
    parser.add_argument("--severity", help="Filter by severity level")
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--owner", help="Filter by owner/assignee")
    parser.add_argument("--start-date", help="Filter by start date (ISO format)")
    parser.add_argument("--end-date", help="Filter by end date (ISO format)")
    
    args = parser.parse_args()
    
    # Handle --list-models flag
    if args.list_models:
        print("Fetching available Ollama models...")
        available_models = get_available_models()
        if available_models:
            print(f"\nAvailable models ({len(available_models)}):")
            for model_name in available_models:
                print(f"  - {model_name}")
        else:
            print("\n[WARN] Could not fetch available models. Ollama may not be running or accessible.")
            print("You can still specify a model name manually using --model flag.")
        return
    
    # If no query provided, use interactive mode
    if not args.query:
        interactive_mode()
    else:
        filters = get_filters_from_args(args)
        test_streaming_query_e2e(
            query=args.query,
            project_name=args.project_name,
            top_k=args.top_k,
            filters=filters,
            model=args.model
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Test interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

