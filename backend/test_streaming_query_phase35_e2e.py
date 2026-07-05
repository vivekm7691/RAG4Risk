"""End-to-end test script for streaming query with Phase 3.5 (past projects) support.

Uses Phase 3.5 parameters: include_past_projects, context_weighting,
exclude_chunk_ids, force_project. Adds Section 7: Similar projects (other
projects used in retrieval), matching Step 7 in MANUAL_TEST_PHASE3.5.md.
"""

import requests
import json
import sys
import argparse
from typing import Dict, Any, Optional, List
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_BASE = f"{BASE_URL}/api"

# Timeout for streaming request: (connect, read). Read must be >= backend OLLAMA_TIMEOUT (3000s).
STREAM_CONNECT_TIMEOUT = 60
STREAM_READ_TIMEOUT = 3100  # > backend OLLAMA_TIMEOUT (3000s) for RAG LLM


def format_context(chunks: List[Dict[str, Any]], project_name: Optional[str] = None) -> str:
    """Format retrieved chunks as context text (same logic as RAGService._format_context)."""
    context_parts = []
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")
        doc_name = metadata.get("file_name", "Unknown document")
        doc_type = metadata.get("document_type", "")
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


def format_context_like_llm(
    chunks: List[Dict[str, Any]], project_name: Optional[str] = None
) -> str:
    """
    Format chunks exactly as RAGService._format_context does for the LLM.
    Includes Phase 3.5 grouping: "Context from Current Project" and "Context from Similar Past Projects".
    """
    current_chunks = []
    past_chunks = []  # list of (project_label, chunk_text, source_info)
    for idx, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata", {})
        chunk_text = chunk.get("text", "")
        doc_name = metadata.get("file_name", "Unknown document")
        doc_type = metadata.get("document_type", "")
        pname = metadata.get("project_name", "")
        source_info = f"[Source {idx}: {doc_name}"
        if doc_type in ["risk register", "issue log"]:
            row_num = metadata.get("row_number")
            sheet_name = metadata.get("sheet_name")
            if row_num:
                source_info += f", Row {row_num}"
            if sheet_name:
                source_info += f", Sheet: {sheet_name}"
        source_info += "]"
        if chunk.get("is_past_project"):
            past_chunks.append((pname or "Past project", chunk_text, source_info))
        else:
            current_chunks.append(f"{source_info}\n{chunk_text}")

    if not past_chunks:
        return "\n\n---\n\n".join(current_chunks)

    parts = []
    if current_chunks:
        label = f"Context from Current Project ({project_name or 'current'}):"
        parts.append(label)
        parts.append("\n\n".join(current_chunks))
    if past_chunks:
        parts.append("Context from Similar Past Projects:")
        for proj_label, text, src in past_chunks:
            parts.append(f"  [{proj_label}] - {src}\n{text}")
    return "\n\n---\n\n".join(parts)


def build_rag_prompt(query: str, context: str, project_name: Optional[str] = None) -> str:
    """Build the RAG prompt with context injection (same logic as RAGService._build_prompt)."""
    project_context = f" for the project '{project_name}'" if project_name else ""
    return f"""You are a helpful assistant that answers questions based on provided context documents{project_context}.

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


def get_retrieve_preview(
    query: str,
    project_name: Optional[str],
    top_k: int,
    include_past_projects: bool,
    context_weighting: Optional[Dict[str, float]] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
    force_project: Optional[Dict[str, str]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Optional[str]]:
    """
    Call retrieve-preview (Phase 3.5) to get chunks and similar_projects for investigation.
    Returns (chunks with text/metadata, similar_projects with full details, error_message).
    """
    if not include_past_projects or not project_name:
        return [], [], None
    try:
        payload = {
            "query": query,
            "project_name": project_name,
            "top_k": top_k,
            "include_past_projects": True,
        }
        if context_weighting:
            payload["context_weighting"] = context_weighting
        if exclude_chunk_ids:
            payload["exclude_chunk_ids"] = exclude_chunk_ids
        if force_project:
            payload["force_project"] = force_project

        r = requests.post(
            f"{API_BASE}/query/retrieve-preview",
            json=payload,
            timeout=30,
        )
        if r.status_code != 200:
            return [], [], f"retrieve-preview failed: {r.status_code} - {r.text[:200]}"
        data = r.json()
        similar = data.get("similar_projects") or []
        chunks_data = data.get("chunks") or []
        chunks = []
        for c in chunks_data:
            meta = (c.get("metadata") or {}).copy()
            meta["chunk_id"] = c.get("chunk_id", "")
            meta["project_name"] = c.get("project_name", "")
            meta["is_past_project"] = c.get("is_past_project", False)
            chunks.append({"text": c.get("text", ""), "metadata": meta})
        similar_full = [
            {
                "project_name": s.get("project_name", ""),
                "customer": s.get("customer"),
                "similarity_score": s.get("similarity_score"),
                "metadata": s.get("metadata"),
            }
            for s in similar
        ]
        return chunks, similar_full, None
    except requests.exceptions.RequestException as e:
        return [], [], f"Network error: {str(e)}"
    except Exception as e:
        return [], [], f"Unexpected error: {str(e)}"


def get_vector_search_results(
    query: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], Optional[str]]:
    """
    Get full chunks from vector search when not using past projects.
    Uses diagnostics endpoint (no LLM) when possible; only falls back to POST /api/query
    when filters are set. If diagnostics fails and no filters, returns empty and lets
    the stream provide sources (avoids 500 from non-streaming query when Ollama is down).
    """
    try:
        params = {"query": query, "top_k": top_k, "include_chunks": "true"}
        if project_name:
            params["project_name"] = project_name
        if not filters:
            try:
                r = requests.get(
                    f"{API_BASE}/diagnostics/vector-search",
                    params=params,
                    timeout=30,
                )
                if r.status_code == 200:
                    data = r.json()
                    if data.get("success") and data.get("chunks"):
                        chunks = [
                            {"text": c.get("text", ""), "metadata": c.get("metadata", {})}
                            for c in data["chunks"]
                        ]
                        return chunks, None
                    if data.get("success") and not data.get("chunks"):
                        return [], None  # No chunks; stream will show empty or from stream
                    # success False or error
                    err = data.get("error") or f"success={data.get('success')}"
                    return [], f"Diagnostics vector-search failed: {err}"
            except requests.exceptions.RequestException as e:
                return [], f"Diagnostics request failed: {e}"
        # With filters we must use POST /api/query (diagnostics has no filters)
        query_data = {"query": query, "top_k": top_k}
        if project_name:
            query_data["project_name"] = project_name
        if filters:
            query_data["filters"] = filters
        r = requests.post(f"{API_BASE}/query", json=query_data, timeout=300)
        if r.status_code != 200:
            try:
                err_body = r.json()
                detail = err_body.get("detail", r.text)
            except Exception:
                detail = r.text
            return [], f"Preliminary query failed: {r.status_code} – {detail}"
        result = r.json()
        sources = result.get("sources", [])
        chunks = []
        for source in sources:
            chunks.append({
                "text": "[Full chunk text not available via query endpoint]",
                "metadata": {
                    "document_id": source.get("document_id", ""),
                    "file_name": source.get("document_name", ""),
                    "chunk_id": source.get("chunk_id", ""),
                    "project_name": source.get("project_name", ""),
                    "document_type": source.get("document_type", ""),
                    "relevance_score": source.get("relevance_score"),
                    "row_number": source.get("row_number"),
                    "sheet_name": source.get("sheet_name"),
                },
            })
        return chunks, None
    except requests.exceptions.RequestException as e:
        return [], f"Network error: {e}"
    except Exception as e:
        return [], f"Unexpected error: {e}"


def get_chunks_from_streaming_sources(sources_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract chunk information from streaming sources message."""
    chunks = []
    for source in (sources_data.get("sources") or []):
        chunks.append({
            "text": "[Full chunk text not available via API]",
            "metadata": {
                "document_id": source.get("document_id", ""),
                "file_name": source.get("document_name", ""),
                "chunk_id": source.get("chunk_id", ""),
                "project_name": source.get("project_name", ""),
                "document_type": source.get("document_type", ""),
                "relevance_score": source.get("relevance_score"),
                "row_number": source.get("row_number"),
                "sheet_name": source.get("sheet_name"),
                "is_past_project": source.get("is_past_project", False),
            },
        })
    return chunks


def display_query_parameters(query_data: Dict[str, Any]):
    print("\n" + "=" * 80)
    print("SECTION 1: QUERY PARAMETERS")
    print("=" * 80)
    print(json.dumps(query_data, indent=2))
    print()


def display_chunks(chunks: List[Dict[str, Any]], project_name: Optional[str] = None):
    """Display chunks exactly as they are presented to the LLM (same format as RAGService._format_context)."""
    print("\n" + "=" * 80)
    print("SECTION 2: CHUNKS AS PRESENTED TO THE LLM")
    print("=" * 80)
    if not chunks:
        print("No chunks retrieved.")
        return
    print(f"Total chunks: {len(chunks)}")
    print("\nBelow is the exact context string passed to the LLM (source lines + chunk text only):\n")
    context_str = format_context_like_llm(chunks, project_name)
    print(context_str)
    print()


def display_prompt(prompt: str):
    print("\n" + "=" * 80)
    print("SECTION 3: FINAL PROMPT SENT TO LLM (for investigation)")
    print("=" * 80)
    print(prompt)
    print()


def display_sources(sources: List[Dict[str, Any]]):
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
        print(f"Chunk ID: {source.get('chunk_id', 'N/A')}")
        print(f"Project: {source.get('project_name', 'N/A')}")
        print(f"Document Type: {source.get('document_type', 'N/A')}")
        if source.get("is_past_project") is not None:
            print(f"Past project: {source.get('is_past_project')}")
        if source.get("relevance_score") is not None:
            print(f"Relevance Score: {source.get('relevance_score'):.4f}")
        if source.get("row_number") is not None:
            print(f"Row Number: {source.get('row_number')}")
        if source.get("sheet_name"):
            print(f"Sheet Name: {source.get('sheet_name')}")
    print()


def display_response(response_text: str):
    print("\n" + "=" * 80)
    print("SECTION 5: LLM RESPONSE (complete, untrimmed)")
    print("=" * 80)
    print(response_text)
    print()


def display_timings(timings: Dict[str, Optional[float]]):
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


def display_similar_projects(similar_projects: List[Dict[str, Any]], project_name: Optional[str]):
    """Display Section 7: Other similar projects used in retrieval (Step 7 in MANUAL_TEST_PHASE3.5.md)."""
    print("\n" + "=" * 80)
    print("SECTION 7: SIMILAR PROJECTS (OTHER PROJECTS USED IN RETRIEVAL)")
    print("=" * 80)
    if not similar_projects:
        print("No similar past projects were used in this retrieval.")
        if project_name:
            print(f"(Current project: {project_name})")
        print()
        return
    print(f"Current project: {project_name or 'N/A'}")
    print(f"Similar projects used for context: {len(similar_projects)}\n")
    for idx, sp in enumerate(similar_projects, 1):
        print(f"--- Similar project {idx} ---")
        print(f"  project_name: {sp.get('project_name', 'N/A')}")
        print(f"  customer: {sp.get('customer', 'N/A')}")
        score = sp.get("similarity_score")
        if score is not None:
            print(f"  similarity_score: {score:.4f}")
        meta = sp.get("metadata")
        if meta:
            print("  metadata:", json.dumps(meta, indent=4))
    print()


def display_summary(
    error_message: Optional[str],
    sources_data: Optional[Dict[str, Any]],
    response_text: str,
    done_data: Optional[Dict[str, Any]],
):
    print("\n" + "=" * 80)
    print("SECTION 8: SUMMARY")
    print("=" * 80)
    if error_message:
        print(f"[ERROR] {error_message}")
        print("Status: FAILED")
    else:
        print("Status: SUCCESS")
        if sources_data:
            print(f"Sources found: {len(sources_data.get('sources', []))}")
            sp_names = sources_data.get("similar_projects")
            if sp_names:
                print(f"Similar projects (from stream): {sp_names}")
        print(f"Response length: {len(response_text)} characters")
        if done_data:
            print(f"Total time: {done_data.get('total_time', 0):.3f}s")
    print()


def parse_sse_stream(response) -> tuple[Dict[str, Any], str, Dict[str, Any], Optional[str], Optional[List[str]]]:
    """
    Parse SSE stream. Returns (sources_data, response_text, done_data, error_message, similar_projects_names).
    similar_projects_names is the list from the sources message when Phase 3.5 is used.
    """
    sources_data = None
    response_chunks = []
    done_data = None
    error_message = None
    similar_projects_names = None
    try:
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith(b"data: "):
                data_str = line[6:].decode("utf-8")
                try:
                    data = json.loads(data_str)
                    msg_type = data.get("type")
                    if msg_type == "sources":
                        sources_data = data
                        similar_projects_names = data.get("similar_projects")
                    elif msg_type == "chunk":
                        response_chunks.append(data.get("text", ""))
                    elif msg_type == "done":
                        done_data = data
                    elif msg_type == "error":
                        error_message = data.get("message", "Unknown error")
                        break
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        error_message = f"Error reading stream: {str(e)}"
    response_text = "".join(response_chunks)
    return sources_data, response_text, done_data, error_message, similar_projects_names


def get_available_models() -> List[str]:
    try:
        r = requests.get(f"{API_BASE}/diagnostics/ollama-models", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("success") and data.get("models"):
                return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
        return []
    except Exception:
        return []


def build_query_data(
    query: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    include_past_projects: bool = False,
    context_weighting: Optional[Dict[str, float]] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
    force_project: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Build request body for /api/query/stream with Phase 3.5 fields."""
    data = {"query": query, "top_k": top_k}
    if project_name:
        data["project_name"] = project_name
    if filters:
        data["filters"] = filters
    if model:
        data["model"] = model
    if include_past_projects:
        data["include_past_projects"] = True
    if context_weighting:
        data["context_weighting"] = context_weighting
    if exclude_chunk_ids:
        data["exclude_chunk_ids"] = exclude_chunk_ids
    if force_project:
        data["force_project"] = force_project
    return data


def test_streaming_query_phase35_e2e(
    query: str,
    project_name: Optional[str] = None,
    top_k: int = 5,
    filters: Optional[Dict[str, Any]] = None,
    model: Optional[str] = None,
    include_past_projects: bool = False,
    context_weighting: Optional[Dict[str, float]] = None,
    exclude_chunk_ids: Optional[List[str]] = None,
    force_project: Optional[Dict[str, str]] = None,
):
    """Run E2E streaming query test with Phase 3.5 (past projects) support."""
    print("\n" + "=" * 80)
    print("END-TO-END STREAMING QUERY TEST (PHASE 3.5 – PAST PROJECTS)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    query_data = build_query_data(
        query=query,
        project_name=project_name,
        top_k=top_k,
        filters=filters,
        model=model,
        include_past_projects=include_past_projects,
        context_weighting=context_weighting,
        exclude_chunk_ids=exclude_chunk_ids,
        force_project=force_project,
    )
    display_query_parameters(query_data)

    chunks = []
    similar_projects_full = []

    if include_past_projects and project_name:
        print("Retrieving chunks and similar projects (retrieve-preview)...")
        chunks, similar_projects_full, err = get_retrieve_preview(
            query=query,
            project_name=project_name,
            top_k=top_k,
            include_past_projects=True,
            context_weighting=context_weighting,
            exclude_chunk_ids=exclude_chunk_ids,
            force_project=force_project,
        )
        if err:
            print(f"[WARN] retrieve-preview: {err}")
    else:
        print("Retrieving chunks for investigation...")
        chunks, chunks_error = get_vector_search_results(query, project_name, top_k, filters)
        if chunks_error:
            print(f"[WARN] {chunks_error}")

    # --- TEMPORARY: LLM call commented out; only output retrieved chunks. Revert to re-enable. ---
    display_chunks(chunks, project_name)
    display_similar_projects(similar_projects_full, project_name)
    print("\n[INFO] LLM call skipped (temporary). Revert comment block in test_streaming_query_phase35_e2e.py to re-enable.\n")
    return

    # --- COMMENTED OUT: streaming query and LLM-dependent output (revert by uncommenting below and removing early return above) ---
    # print("Making streaming query call...")
    # try:
    #     response = requests.post(
    #         f"{API_BASE}/query/stream",
    #         json=query_data,
    #         stream=True,
    #         headers={"Accept": "text/event-stream"},
    #         timeout=(STREAM_CONNECT_TIMEOUT, STREAM_READ_TIMEOUT),
    #     )
    #     if response.status_code != 200:
    #         print(f"\n[ERROR] Query failed with status {response.status_code}")
    #         try:
    #             print(json.dumps(response.json(), indent=2))
    #         except Exception:
    #             print(response.text)
    #         return
    #
    #     sources_data, response_text, done_data, error_message, _ = parse_sse_stream(response)
    #     if not chunks and sources_data:
    #         chunks = get_chunks_from_streaming_sources(sources_data)
    #
    #     display_chunks(chunks)
    #
    #     if chunks:
    #         context_text = format_context(chunks, project_name)
    #         prompt = build_rag_prompt(query, context_text, project_name)
    #         display_prompt(prompt)
    #     else:
    #         print("\n" + "=" * 80)
    #         print("SECTION 3: FINAL PROMPT SENT TO LLM (for investigation)")
    #         print("=" * 80)
    #         print("[No chunks available to build prompt]")
    #         print()
    #
    #     if sources_data:
    #         display_sources(sources_data.get("sources", []))
    #
    #     display_response(response_text)
    #     if done_data:
    #         display_timings(done_data.get("timings", {}))
    #     else:
    #         display_timings({})
    #
    #     display_similar_projects(similar_projects_full, project_name)
    #     display_summary(error_message, sources_data, response_text, done_data)
    #
    # except requests.exceptions.Timeout:
    #     print("\n[ERROR] Request timed out (connect or stream read)")
    #     print("  If the LLM is slow, increase STREAM_READ_TIMEOUT in this script (backend uses OLLAMA_TIMEOUT).")
    #     print("Status: TIMEOUT")
    # except requests.exceptions.ConnectionError:
    #     print("\n[ERROR] Cannot connect to backend. Is the server running?")
    #     print("  Start with: docker-compose up -d")
    #     print("Status: CONNECTION ERROR")
    # except Exception as e:
    #     print(f"\n[ERROR] Unexpected error: {str(e)}")
    #     import traceback
    #     traceback.print_exc()
    #     print("Status: FAILED")


def get_filters_from_args(args) -> Optional[Dict[str, Any]]:
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
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE – Phase 3.5 streaming query")
    print("=" * 80)

    query = input("\nQuery (required): ").strip()
    if not query:
        print("Error: Query is required")
        return

    project_name = input("Project name (optional, press Enter to skip): ").strip() or None
    top_k_str = input("Top K (default 5): ").strip()
    top_k = int(top_k_str) if top_k_str else 5

    print("\nFetching available Ollama models...")
    available_models = get_available_models()
    model = None
    if available_models:
        print(f"Available models: {', '.join(available_models)}")
        model_choice = input("Model (press Enter for default): ").strip()
        if model_choice:
            model = model_choice if model_choice in available_models else model_choice
    else:
        model = input("Model name (optional): ").strip() or None

    include_past = input("Include past projects? (y/N): ").strip().lower() == "y"
    context_weighting = None
    if include_past:
        cw = input("Current project weight (default 0.7): ").strip() or "0.7"
        pw = input("Past projects weight (default 0.3): ").strip() or "0.3"
        try:
            context_weighting = {
                "current_project_weight": float(cw),
                "past_projects_weight": float(pw),
            }
        except ValueError:
            context_weighting = None
    exclude_ids_str = input("Exclude chunk IDs (comma-separated, optional): ").strip()
    exclude_chunk_ids = [x.strip() for x in exclude_ids_str.split(",") if x.strip()] or None
    force_customer = input("Force project – customer (optional): ").strip() or None
    force_pname = input("Force project – project_name (optional): ").strip() or None
    force_project = None
    if force_customer and force_pname:
        force_project = {"customer": force_customer, "project_name": force_pname}

    print("\nExcel filters (optional):")
    severity = input("  Severity: ").strip() or None
    status = input("  Status: ").strip() or None
    category = input("  Category: ").strip() or None
    owner = input("  Owner: ").strip() or None
    start_date = input("  Start date (ISO): ").strip() or None
    end_date = input("  End date (ISO): ").strip() or None
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

    test_streaming_query_phase35_e2e(
        query=query,
        project_name=project_name,
        top_k=top_k,
        filters=filters,
        model=model,
        include_past_projects=include_past,
        context_weighting=context_weighting,
        exclude_chunk_ids=exclude_chunk_ids,
        force_project=force_project,
    )


def main():
    parser = argparse.ArgumentParser(
        description="E2E streaming query test with Phase 3.5 (past projects). Section 7 = similar projects used in retrieval."
    )
    parser.add_argument("--query", "-q", help="User query text")
    parser.add_argument("--project-name", "-p", help="Project name filter")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of results (default: 5)")
    parser.add_argument("--model", "-m", help="Ollama model name")
    parser.add_argument("--list-models", action="store_true", help="List available Ollama models and exit")

    parser.add_argument("--include-past-projects", action="store_true", help="Include context from similar past projects (Phase 3.5)")
    parser.add_argument("--current-weight", type=float, default=0.7, help="Current project context weight (default: 0.7)")
    parser.add_argument("--past-weight", type=float, default=0.3, help="Past projects context weight (default: 0.3)")
    parser.add_argument("--exclude-chunk-ids", help="Comma-separated chunk IDs to exclude")
    parser.add_argument("--force-project-customer", help="Force past project by customer name")
    parser.add_argument("--force-project-name", help="Force past project by project name")

    parser.add_argument("--severity", help="Filter by severity")
    parser.add_argument("--status", help="Filter by status")
    parser.add_argument("--category", help="Filter by category")
    parser.add_argument("--owner", help="Filter by owner")
    parser.add_argument("--start-date", help="Filter start date (ISO)")
    parser.add_argument("--end-date", help="Filter end date (ISO)")

    args = parser.parse_args()

    if args.list_models:
        models = get_available_models()
        if models:
            for m in models:
                print(m)
        else:
            print("Could not fetch models.")
        return

    if not args.query:
        interactive_mode()
        return

    filters = get_filters_from_args(args)
    context_weighting = None
    if args.include_past_projects:
        context_weighting = {
            "current_project_weight": args.current_weight,
            "past_projects_weight": args.past_weight,
        }
    exclude_chunk_ids = None
    if args.exclude_chunk_ids:
        exclude_chunk_ids = [x.strip() for x in args.exclude_chunk_ids.split(",") if x.strip()]
    force_project = None
    if args.force_project_customer and args.force_project_name:
        force_project = {"customer": args.force_project_customer, "project_name": args.force_project_name}

    test_streaming_query_phase35_e2e(
        query=args.query,
        project_name=args.project_name,
        top_k=args.top_k,
        filters=filters,
        model=args.model,
        include_past_projects=args.include_past_projects,
        context_weighting=context_weighting,
        exclude_chunk_ids=exclude_chunk_ids,
        force_project=force_project,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
