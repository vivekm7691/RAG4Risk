"""Query and search API routes"""

import json
import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.query import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    RetrievalPreviewResponse,
    PreviewChunk,
    SimilarProjectPreview,
)
from app.models.document import DocumentType
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_service import RAGService
from app.services.query_service import QueryService
from app.config import settings

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
rag_service = RAGService()
query_service = QueryService()


def _context_weighting_from_request(request: QueryRequest):
    """Get current/past weights from request or config defaults."""
    if request.context_weighting:
        return request.context_weighting.current_project_weight, request.context_weighting.past_projects_weight
    return settings.DEFAULT_CURRENT_PROJECT_WEIGHT, settings.DEFAULT_PAST_PROJECTS_WEIGHT


@router.post("/retrieve-preview", response_model=RetrievalPreviewResponse, status_code=status.HTTP_200_OK)
async def retrieve_preview(request: QueryRequest):
    """
    Phase 3.5: Retrieval preview when including past projects.
    Returns similar projects (with metadata) and candidate chunks (with chunk_id). No LLM call.
    """
    if not request.include_past_projects:
        return RetrievalPreviewResponse(
            similar_projects=[],
            chunks=[],
            query=request.query,
            project_name=request.project_name,
        )
    cw, pw = _context_weighting_from_request(request)
    force = request.force_project.model_dump() if request.force_project else None
    try:
        result = await query_service.retrieve_with_past_projects(
            query=request.query,
            project_name=request.project_name,
            top_k=request.top_k,
            current_project_weight=cw,
            past_projects_weight=pw,
            exclude_chunk_ids=request.exclude_chunk_ids,
            force_project=force,
            preview_only=True,
        )
    except Exception as e:
        logger.exception("Retrieve preview failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    similar = [
        SimilarProjectPreview(
            project_name=sp["project_name"],
            customer=sp.get("customer"),
            similarity_score=sp.get("similarity_score", 0.0),
            metadata=sp.get("metadata").model_dump() if hasattr(sp.get("metadata"), "model_dump") else (sp.get("metadata") or {}),
        )
        for sp in result["similar_projects"]
    ]
    chunks = [
        PreviewChunk(
            chunk_id=(c["metadata"].get("chunk_id") or ""),
            text=c["text"],
            project_name=c["metadata"].get("project_name", ""),
            customer=c["metadata"].get("_similar_project", {}).get("customer") if c.get("is_past_project") else c["metadata"].get("customer"),
            metadata={k: v for k, v in c["metadata"].items() if not k.startswith("_")},
            is_past_project=c.get("is_past_project", False),
        )
        for c in result["chunks"]
    ]
    return RetrievalPreviewResponse(
        similar_projects=similar,
        chunks=chunks,
        query=request.query,
        project_name=request.project_name,
    )


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(request: QueryRequest):
    """
    Query documents using semantic search and RAG
    
    - **query**: User query text
    - **project_name**: Optional project name filter
    - **top_k**: Number of results to return (default: 5)
    - **filters**: Optional metadata filters for Excel-based queries
        - severity: Filter by severity level
        - status: Filter by status
        - category: Filter by category
        - date_range: Filter by date range (start_date, end_date in ISO format)
        - owner: Filter by owner/assignee
    """
    total_start = time.time()
    
    try:
        embedding_start = time.time()
        search_results = None
        similar_projects_result = None

        if request.include_past_projects and request.project_name:
            cw, pw = _context_weighting_from_request(request)
            force = request.force_project.model_dump() if request.force_project else None
            result = await query_service.retrieve_with_past_projects(
                query=request.query,
                project_name=request.project_name,
                top_k=request.top_k,
                current_project_weight=cw,
                past_projects_weight=pw,
                exclude_chunk_ids=request.exclude_chunk_ids,
                force_project=force,
                preview_only=False,
            )
            similar_projects_result = result["similar_projects"]
            search_results = result["chunks"]
        else:
            query_embedding = embedding_service.generate_embedding(request.query).tolist()
            filters_dict = request.filters.model_dump(exclude_none=True) if request.filters else None
            vector_store = await get_vector_store()
            search_results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                project_name=request.project_name,
                document_type=None,
                filters=filters_dict,
            )

        embedding_time = time.time() - embedding_start
        search_time = 0.0  # folded into embedding_time for past-projects path

        if not search_results:
            total_time = time.time() - total_start
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=request.query,
                project_name=request.project_name,
                similar_projects=[sp.get("project_name") for sp in (similar_projects_result or [])] or None,
            )

        format_start = time.time()
        context_chunks = []
        for r in search_results:
            if isinstance(r, dict):
                text = r.get("text", "")
                meta = r.get("metadata", r) if "metadata" in r else r
            else:
                text, meta = r["text"], r["metadata"]
            context_chunks.append({"text": text, "metadata": meta})
        format_time = time.time() - format_start

        model_to_use = request.model or settings.OLLAMA_MODEL
        rag_service_instance = RAGService(model=model_to_use) if request.model else rag_service
        llm_start = time.time()
        answer = await rag_service_instance.generate_response(
            query=request.query,
            context_chunks=context_chunks,
            project_name=request.project_name,
            stream=False,
        )
        llm_time = time.time() - llm_start

        sources = []
        for r in search_results:
            meta = r.get("metadata", r) if isinstance(r, dict) and "metadata" in r else (r if isinstance(r, dict) else {})
            dist = r.get("distance") if isinstance(r, dict) else None
            relevance_score = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else None
            is_past = r.get("is_past_project", False) if isinstance(r, dict) else False
            sources.append(SourceCitation(
                document_id=meta.get("document_id", ""),
                document_name=meta.get("file_name", "Unknown document"),
                chunk_id=meta.get("chunk_id", ""),
                project_name=meta.get("project_name", ""),
                document_type=meta.get("document_type", ""),
                relevance_score=relevance_score,
                is_past_project=is_past,
                row_number=meta.get("row_number"),
                sheet_name=meta.get("sheet_name"),
            ))

        total_time = time.time() - total_start
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            project_name=request.project_name,
            similar_projects=[sp.get("project_name") for sp in (similar_projects_result or [])] or None,
        )
        
    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"Query failed after {total_time:.3f}s: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/stream", status_code=status.HTTP_200_OK)
async def query_documents_stream(request: QueryRequest):
    """
    Query documents using semantic search and RAG with streaming response
    
    Returns Server-Sent Events (SSE) format with incremental response chunks.
    
    - **query**: User query text
    - **project_name**: Optional project name filter
    - **top_k**: Number of results to return (default: 5)
    - **filters**: Optional metadata filters for Excel-based queries
        - severity: Filter by severity level
        - status: Filter by status
        - category: Filter by category
        - date_range: Filter by date range (start_date, end_date in ISO format)
        - owner: Filter by owner/assignee
    
    Response format (SSE):
    - Initial: {"type": "sources", "sources": [...], "query": "...", "project_name": "..."}
    - Chunks: {"type": "chunk", "text": "partial response"}
    - Done: {"type": "done", "total_time": 5.234}
    - Error: {"type": "error", "message": "error message"}
    """
    total_start = time.time()
    
    async def generate_stream():
        """Generator function for streaming response"""
        try:
            embedding_start = time.time()
            similar_projects_stream = None
            if request.include_past_projects and request.project_name:
                cw, pw = _context_weighting_from_request(request)
                force = request.force_project.model_dump() if request.force_project else None
                result = await query_service.retrieve_with_past_projects(
                    query=request.query,
                    project_name=request.project_name,
                    top_k=request.top_k,
                    current_project_weight=cw,
                    past_projects_weight=pw,
                    exclude_chunk_ids=request.exclude_chunk_ids,
                    force_project=force,
                    preview_only=False,
                )
                similar_projects_stream = result["similar_projects"]
                search_results = result["chunks"]
            else:
                query_embedding = embedding_service.generate_embedding(request.query).tolist()
                filters_dict = request.filters.model_dump(exclude_none=True) if request.filters else None
                vector_store = await get_vector_store()
                search_results = await vector_store.search(
                    query_embedding=query_embedding,
                    top_k=request.top_k,
                    project_name=request.project_name,
                    document_type=None,
                    filters=filters_dict,
                )
            embedding_time = time.time() - embedding_start
            search_time = 0.0
            logger.info(f"Retrieval completed in {embedding_time:.3f}s, found {len(search_results) if search_results else 0} results")
            
            if not search_results:
                total_time = time.time() - total_start
                logger.info(f"Query completed in {total_time:.3f}s (no results found)")
                # Send sources message with empty sources
                sources_data = {
                    "type": "sources",
                    "sources": [],
                    "query": request.query,
                    "project_name": request.project_name
                }
                yield f"data: {json.dumps(sources_data)}\n\n"
                
                # Send answer message
                answer_data = {
                    "type": "chunk",
                    "text": "I couldn't find any relevant information to answer your question."
                }
                yield f"data: {json.dumps(answer_data)}\n\n"
                
                # Send done message
                done_data = {
                    "type": "done",
                    "total_time": round(total_time, 3),
                    "timings": {
                        "embedding": round(embedding_time, 3),
                        "search": round(search_time, 3),
                        "format": 0.0,
                        "llm": None
                    }
                }
                yield f"data: {json.dumps(done_data)}\n\n"
                return
            
            # Build source citations (support both raw search results and query_service chunk dicts)
            sources = []
            for result in search_results:
                meta = result.get("metadata", result) if isinstance(result, dict) else {}
                dist = result.get("distance") if isinstance(result, dict) else None
                relevance_score = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else None
                is_past = result.get("is_past_project", False) if isinstance(result, dict) else False
                citation = SourceCitation(
                    document_id=meta.get("document_id", ""),
                    document_name=meta.get("file_name", "Unknown document"),
                    chunk_id=meta.get("chunk_id", ""),
                    project_name=meta.get("project_name", ""),
                    document_type=meta.get("document_type", ""),
                    relevance_score=relevance_score,
                    is_past_project=is_past,
                    row_number=meta.get("row_number"),
                    sheet_name=meta.get("sheet_name"),
                )
                sources.append(citation)
            
            format_start = time.time()
            context_chunks = []
            for r in search_results:
                text = r.get("text", "")
                meta = r.get("metadata", r) if isinstance(r, dict) and "metadata" in r else (r if isinstance(r, dict) else {})
                context_chunks.append({"text": text, "metadata": meta})
            format_time = time.time() - format_start
            
            sources_data = {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "query": request.query,
                "project_name": request.project_name,
            }
            if similar_projects_stream:
                sources_data["similar_projects"] = [sp.get("project_name") for sp in similar_projects_stream]
            yield f"data: {json.dumps(sources_data)}\n\n"
            
            # Generate streaming response using RAG (now async)
            # Use model from request if provided, otherwise use default
            model_to_use = request.model or settings.OLLAMA_MODEL
            rag_service_instance = RAGService(model=model_to_use) if request.model else rag_service
            
            llm_start = time.time()
            try:
                # Get async generator from RAG service (returns generator, don't await)
                async_gen = rag_service_instance.generate_response(
                    query=request.query,
                    context_chunks=context_chunks,
                    project_name=request.project_name,
                    stream=True
                )
                
                # Await to get the actual generator, then consume it
                generator = await async_gen
                async for chunk in generator:
                    chunk_data = {
                        "type": "chunk",
                        "text": chunk
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"
                
                llm_time = time.time() - llm_start
                logger.info(f"LLM response streamed in {llm_time:.3f}s (model: {model_to_use})")
                
            except Exception as llm_error:
                llm_time = time.time() - llm_start
                logger.error(f"LLM streaming failed after {llm_time:.3f}s: {str(llm_error)}")
                error_data = {
                    "type": "error",
                    "message": f"LLM error: {str(llm_error)}"
                }
                yield f"data: {json.dumps(error_data)}\n\n"
            
            # Send done message
            total_time = time.time() - total_start
            done_data = {
                "type": "done",
                "total_time": round(total_time, 3),
                "timings": {
                    "embedding": round(embedding_time, 3),
                    "search": round(search_time, 3),
                    "format": round(format_time, 3),
                    "llm": round(llm_time, 3) if 'llm_time' in locals() else None
                }
            }
            yield f"data: {json.dumps(done_data)}\n\n"
            
        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"Streaming query failed after {total_time:.3f}s: {str(e)}")
            error_data = {
                "type": "error",
                "message": f"Query failed: {str(e)}",
                "total_time": round(total_time, 3)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

