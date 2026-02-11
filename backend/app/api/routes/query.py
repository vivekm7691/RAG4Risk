"""Query and search API routes"""

import json
import time
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.query import QueryRequest, QueryResponse, SourceCitation
from app.models.document import DocumentType
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_service import RAGService
from app.config import settings

# Set up logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
rag_service = RAGService()
# Note: vector_store is now async, get it per request


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
        # Generate query embedding
        embedding_start = time.time()
        query_embedding = embedding_service.generate_embedding(request.query)
        query_embedding_list = query_embedding.tolist()
        embedding_time = time.time() - embedding_start
        logger.info(f"Query embedding generated in {embedding_time:.3f}s")
        
        # Prepare filters for vector store
        filters_dict = None
        if request.filters:
            filters_dict = request.filters.model_dump(exclude_none=True)
        
        # Search vector store (now async with Qdrant)
        search_start = time.time()
        vector_store = await get_vector_store()
        search_results = await vector_store.search(
            query_embedding=query_embedding_list,
            top_k=request.top_k,
            project_name=request.project_name,
            document_type=None,
            filters=filters_dict
        )
        search_time = time.time() - search_start
        logger.info(f"Vector search completed in {search_time:.3f}s, found {len(search_results) if search_results else 0} results")
        
        if not search_results:
            total_time = time.time() - total_start
            logger.info(f"Query completed in {total_time:.3f}s (no results found)")
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=request.query,
                project_name=request.project_name
            )
        
        # Format chunks for RAG service
        format_start = time.time()
        context_chunks = []
        for result in search_results:
            context_chunks.append({
                "text": result["text"],
                "metadata": result["metadata"]
            })
        format_time = time.time() - format_start
        logger.info(f"Chunks formatted in {format_time:.3f}s")
        
        # Generate response using RAG (now async)
        llm_start = time.time()
        answer = await rag_service.generate_response(
            query=request.query,
            context_chunks=context_chunks,
            project_name=request.project_name,
            stream=False
        )
        llm_time = time.time() - llm_start
        logger.info(f"LLM response generated in {llm_time:.3f}s")
        
        # Build source citations
        sources = []
        for result in search_results:
            metadata = result["metadata"]
            
            # Calculate relevance score from distance (lower distance = higher relevance)
            distance = result.get("distance")
            relevance_score = None
            if distance is not None:
                # Convert distance to similarity score (1 - normalized distance)
                # Assuming cosine distance ranges from 0 to 2, normalize to 0-1
                relevance_score = max(0.0, 1.0 - (distance / 2.0))
            
            citation = SourceCitation(
                document_id=metadata.get("document_id", ""),
                document_name=metadata.get("file_name", "Unknown document"),
                chunk_id=metadata.get("chunk_id", ""),
                project_name=metadata.get("project_name", ""),
                document_type=metadata.get("document_type", ""),
                relevance_score=relevance_score,
                # Excel-specific metadata
                row_number=metadata.get("row_number"),
                sheet_name=metadata.get("sheet_name")
            )
            sources.append(citation)
        
        total_time = time.time() - total_start
        logger.info(
            f"Query completed in {total_time:.3f}s "
            f"(embedding: {embedding_time:.3f}s, search: {search_time:.3f}s, "
            f"format: {format_time:.3f}s, llm: {llm_time:.3f}s)"
        )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            project_name=request.project_name
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
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = embedding_service.generate_embedding(request.query)
            query_embedding_list = query_embedding.tolist()
            embedding_time = time.time() - embedding_start
            logger.info(f"Query embedding generated in {embedding_time:.3f}s")
            
            # Prepare filters for vector store
            filters_dict = None
            if request.filters:
                filters_dict = request.filters.model_dump(exclude_none=True)
            
            # Search vector store (now async with Qdrant)
            search_start = time.time()
            vector_store = await get_vector_store()
            search_results = await vector_store.search(
                query_embedding=query_embedding_list,
                top_k=request.top_k,
                project_name=request.project_name,
                document_type=None,
                filters=filters_dict
            )
            search_time = time.time() - search_start
            logger.info(f"Vector search completed in {search_time:.3f}s, found {len(search_results) if search_results else 0} results")
            
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
            
            # Build source citations
            sources = []
            for result in search_results:
                metadata = result["metadata"]
                
                # Calculate relevance score from distance
                distance = result.get("distance")
                relevance_score = None
                if distance is not None:
                    relevance_score = max(0.0, 1.0 - (distance / 2.0))
                
                citation = SourceCitation(
                    document_id=metadata.get("document_id", ""),
                    document_name=metadata.get("file_name", "Unknown document"),
                    chunk_id=metadata.get("chunk_id", ""),
                    project_name=metadata.get("project_name", ""),
                    document_type=metadata.get("document_type", ""),
                    relevance_score=relevance_score,
                    row_number=metadata.get("row_number"),
                    sheet_name=metadata.get("sheet_name")
                )
                sources.append(citation)
            
            # Format chunks for RAG service
            format_start = time.time()
            context_chunks = []
            for result in search_results:
                context_chunks.append({
                    "text": result["text"],
                    "metadata": result["metadata"]
                })
            format_time = time.time() - format_start
            logger.info(f"Chunks formatted in {format_time:.3f}s")
            
            # Send sources first
            sources_data = {
                "type": "sources",
                "sources": [source.model_dump() for source in sources],
                "query": request.query,
                "project_name": request.project_name
            }
            yield f"data: {json.dumps(sources_data)}\n\n"
            
            # Generate streaming response using RAG (now async)
            llm_start = time.time()
            try:
                # Get async generator from RAG service (returns generator, don't await)
                async_gen = rag_service.generate_response(
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
                logger.info(f"LLM response streamed in {llm_time:.3f}s")
                
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

