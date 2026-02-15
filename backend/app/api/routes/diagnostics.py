"""Diagnostic endpoints for testing system components"""

import time
from typing import Optional, Dict, Any
import httpx
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel

from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_service import RAGService
from app.config import settings

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
# Note: vector_store is now async, get it per request
rag_service = RAGService()


class QueryTimingRequest(BaseModel):
    """Request model for query timing endpoint"""
    query: str
    project_name: Optional[str] = None
    top_k: int = 5


class QueryTimingResponse(BaseModel):
    """Response model for query timing endpoint"""
    query: str
    timings: Dict[str, float]
    embedding_time: float
    search_time: float
    llm_time: Optional[float] = None
    total_time: float
    success: bool
    error: Optional[str] = None


@router.get("/ollama", status_code=status.HTTP_200_OK)
async def test_ollama():
    """
    Test Ollama connectivity and model availability
    
    Returns:
        - connection_status: Whether Ollama is reachable
        - model: Model name being used
        - response_time: Time taken to get a response (seconds)
        - error: Error message if connection failed
    """
    start_time = time.time()
    result = {
        "connection_status": "disconnected",
        "model": settings.OLLAMA_MODEL,
        "base_url": settings.OLLAMA_BASE_URL,
        "response_time": None,
        "error": None
    }
    
    try:
        # Test with a simple prompt
        test_prompt = "Say 'OK' if you can read this."
        url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        
        payload = {
            "model": settings.OLLAMA_MODEL,
            "prompt": test_prompt,
            "stream": False
        }
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            
            result_data = response.json()
            if "response" in result_data:
                result["connection_status"] = "connected"
                result["response_time"] = time.time() - start_time
            else:
                result["error"] = "Unexpected response format from Ollama"
                
    except httpx.TimeoutException:
        result["error"] = f"Connection to Ollama timed out after 30 seconds"
        result["response_time"] = time.time() - start_time
    except httpx.HTTPStatusError as e:
        result["error"] = f"Ollama API returned error: {e.response.status_code} - {e.response.text}"
        result["response_time"] = time.time() - start_time
    except httpx.RequestError as e:
        result["error"] = f"Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}: {str(e)}"
        result["response_time"] = time.time() - start_time
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
        result["response_time"] = time.time() - start_time
    
    return result


@router.get("/ollama-models", status_code=status.HTTP_200_OK)
async def list_ollama_models():
    """
    List all available Ollama models
    
    Returns:
        - success: Whether the request succeeded
        - models: List of available models with metadata (name, size, modified_at)
        - error: Error message if request failed
    """
    result = {
        "success": False,
        "models": [],
        "error": None
    }
    
    try:
        # Call Ollama API to get list of models
        url = f"{settings.OLLAMA_BASE_URL}/api/tags"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            # Parse response - Ollama returns {"models": [...]}
            if "models" in data:
                models_list = []
                for model in data["models"]:
                    model_info = {
                        "name": model.get("name", "unknown"),
                        "size": model.get("size", 0),
                        "modified_at": model.get("modified_at")
                    }
                    models_list.append(model_info)
                
                result["success"] = True
                result["models"] = models_list
            else:
                result["error"] = "Unexpected response format from Ollama API"
                
    except httpx.TimeoutException:
        result["error"] = f"Connection to Ollama timed out after 30 seconds"
    except httpx.HTTPStatusError as e:
        result["error"] = f"Ollama API returned error: {e.response.status_code} - {e.response.text}"
    except httpx.RequestError as e:
        result["error"] = f"Failed to connect to Ollama at {settings.OLLAMA_BASE_URL}: {str(e)}"
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result


@router.get("/embedding", status_code=status.HTTP_200_OK)
async def test_embedding(
    query: str = Query(..., description="Test query text to generate embedding for")
):
    """
    Test embedding generation speed
    
    Args:
        query: Test query text
        
    Returns:
        - query: The input query
        - embedding_time: Time taken to generate embedding (seconds)
        - embedding_dimension: Dimension of the generated embedding
        - success: Whether embedding generation succeeded
    """
    start_time = time.time()
    result = {
        "query": query,
        "embedding_time": None,
        "embedding_dimension": None,
        "success": False,
        "error": None
    }
    
    try:
        embedding = embedding_service.generate_embedding(query)
        embedding_time = time.time() - start_time
        
        result["embedding_time"] = embedding_time
        result["embedding_dimension"] = len(embedding)
        result["success"] = True
        
    except Exception as e:
        result["error"] = f"Failed to generate embedding: {str(e)}"
        result["embedding_time"] = time.time() - start_time
    
    return result


@router.get("/vector-search", status_code=status.HTTP_200_OK)
async def test_vector_search(
    query: str = Query(..., description="Test query text to search for"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    project_name: Optional[str] = Query(None, description="Optional project name filter"),
    include_chunks: bool = Query(False, description="Include full chunks with text in response")
):
    """
    Test vector store search speed
    
    Args:
        query: Test query text
        top_k: Number of results to return
        project_name: Optional project name filter
        include_chunks: If True, include full chunks with text in response (for investigation)
        
    Returns:
        - query: The input query
        - embedding_time: Time to generate embedding (seconds)
        - search_time: Time to search vector store (seconds)
        - total_time: Total time (seconds)
        - results_count: Number of results found
        - chunks: Full chunks with text and metadata (if include_chunks=True)
        - success: Whether search succeeded
    """
    start_time = time.time()
    result = {
        "query": query,
        "embedding_time": None,
        "search_time": None,
        "total_time": None,
        "results_count": 0,
        "chunks": None,
        "success": False,
        "error": None
    }
    
    try:
        # Generate embedding
        embedding_start = time.time()
        query_embedding = embedding_service.generate_embedding(query)
        query_embedding_list = query_embedding.tolist()
        result["embedding_time"] = time.time() - embedding_start
        
        # Search vector store (now async with Qdrant)
        search_start = time.time()
        vector_store = await get_vector_store()
        search_results = await vector_store.search(
            query_embedding=query_embedding_list,
            top_k=top_k,
            project_name=project_name,
            document_type=None,
            filters=None
        )
        result["search_time"] = time.time() - search_start
        result["results_count"] = len(search_results) if search_results else 0
        
        # Include full chunks if requested
        if include_chunks and search_results:
            result["chunks"] = search_results
        
        result["total_time"] = time.time() - start_time
        result["success"] = True
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Vector search failed: {type(e).__name__}: {str(e)}", exc_info=True)
        result["error"] = f"Failed to perform vector search: {type(e).__name__}: {str(e)}"
        result["total_time"] = time.time() - start_time
    
    return result


@router.post("/query-timing", response_model=QueryTimingResponse, status_code=status.HTTP_200_OK)
async def test_query_timing(request: QueryTimingRequest):
    """
    Test full query with detailed timing breakdown
    
    Measures time for:
    - Embedding generation
    - Vector store search
    - LLM response generation
    
    Args:
        request: Query timing request with query text and optional filters
        
    Returns:
        - query: The input query
        - timings: Detailed timing breakdown
        - embedding_time: Time for embedding generation (seconds)
        - search_time: Time for vector search (seconds)
        - llm_time: Time for LLM response (seconds, None if failed)
        - total_time: Total query time (seconds)
        - success: Whether query succeeded
        - error: Error message if query failed
    """
    total_start = time.time()
    timings = {}
    
    result = {
        "query": request.query,
        "timings": {},
        "embedding_time": None,
        "search_time": None,
        "llm_time": None,
        "total_time": None,
        "success": False,
        "error": None
    }
    
    try:
        # Step 1: Generate query embedding
        embedding_start = time.time()
        query_embedding = embedding_service.generate_embedding(request.query)
        query_embedding_list = query_embedding.tolist()
        embedding_time = time.time() - embedding_start
        timings["embedding_generation"] = embedding_time
        result["embedding_time"] = embedding_time
        
        # Step 2: Search vector store (now async with Qdrant)
        search_start = time.time()
        vector_store = await get_vector_store()
        search_results = await vector_store.search(
            query_embedding=query_embedding_list,
            top_k=request.top_k,
            project_name=request.project_name,
            document_type=None,
            filters=None
        )
        search_time = time.time() - search_start
        timings["vector_search"] = search_time
        result["search_time"] = search_time
        
        if not search_results:
            result["error"] = "No search results found"
            result["total_time"] = time.time() - total_start
            result["timings"] = timings
            return result
        
        # Step 3: Format chunks for RAG
        format_start = time.time()
        context_chunks = []
        for search_result in search_results:
            context_chunks.append({
                "text": search_result["text"],
                "metadata": search_result["metadata"]
            })
        format_time = time.time() - format_start
        timings["chunk_formatting"] = format_time
        
        # Step 4: Generate LLM response
        llm_start = time.time()
        try:
            answer = rag_service.generate_response(
                query=request.query,
                context_chunks=context_chunks,
                project_name=request.project_name
            )
            llm_time = time.time() - llm_start
            timings["llm_response"] = llm_time
            result["llm_time"] = llm_time
            result["success"] = True
        except Exception as llm_error:
            llm_time = time.time() - llm_start
            timings["llm_response"] = llm_time
            result["llm_time"] = llm_time
            result["error"] = f"LLM call failed: {str(llm_error)}"
        
        result["total_time"] = time.time() - total_start
        result["timings"] = timings
        
    except Exception as e:
        result["error"] = f"Query failed: {str(e)}"
        result["total_time"] = time.time() - total_start
        result["timings"] = timings
    
    return result


