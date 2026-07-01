"""Query and search API routes"""

import json
import time
import logging
from typing import Optional, List, Any, Dict

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from app.models.query import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    RetrievalPreviewResponse,
    PreviewChunk,
    SimilarProjectPreview,
    QueryIntentInfo,
    PastProjectIntentSlots,
    GraphExpansionInfo,
    metadata_filters_for_vector_search,
)
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store
from app.services.rag_service import RAGService
from app.services.query_service import QueryService, past_project_slot_bases
from app.services.query_intent_service import (
    allocate_top_k_by_weights,
    QueryIntentResult,
    QueryIntentService,
)
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

embedding_service = EmbeddingService()
rag_service = RAGService()
query_service = QueryService()


def _context_weighting_from_request(request: QueryRequest):
    if request.context_weighting:
        return request.context_weighting.current_project_weight, request.context_weighting.past_projects_weight
    return settings.DEFAULT_CURRENT_PROJECT_WEIGHT, settings.DEFAULT_PAST_PROJECTS_WEIGHT


def _filters_dict(request: QueryRequest) -> Optional[Dict[str, Any]]:
    return (
        metadata_filters_for_vector_search(request.filters)
        if request.filters
        else None
    )


def _intent_will_run(request: QueryRequest) -> bool:
    return (
        settings.QUERY_INTENT_ENABLED
        and request.use_query_intent
        and bool(request.project_name)
    )


def _graph_augmentation_will_run(request: QueryRequest) -> bool:
    return (
        settings.GRAPH_ENABLED
        and request.use_graph_augmentation
        and bool(request.project_name)
    )


async def _augment_with_graph_if_enabled(
    request: QueryRequest,
    chunks: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Optional[GraphExpansionInfo], Optional[List[str]]]:
    if not _graph_augmentation_will_run(request) or not chunks:
        return chunks, None, None

    from app.services.graph_retrieval import augment_chunks_with_graph

    try:
        result = await augment_chunks_with_graph(
            chunks,
            project_name=request.project_name or "",
        )
        return (
            result.chunks,
            GraphExpansionInfo(
                enabled=True,
                added_chunk_ids=result.added_chunk_ids,
                paths_summary=result.paths_summary,
            ),
            result.graph_context_lines or None,
        )
    except Exception as exc:
        logger.warning("Graph augmentation failed, using vector-only: %s", exc)
        return chunks, GraphExpansionInfo(enabled=True), None


def _normalize_context_chunks(search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    context_chunks = []
    for r in search_results:
        if isinstance(r, dict):
            text = r.get("text", "")
            meta = r.get("metadata", r) if "metadata" in r else r
            chunk = {"text": text, "metadata": meta}
            if "is_past_project" in r:
                chunk["is_past_project"] = r["is_past_project"]
        else:
            chunk = {"text": r["text"], "metadata": r["metadata"]}
        context_chunks.append(chunk)
    return context_chunks


def _search_results_to_sources(search_results: List[Dict[str, Any]]) -> List[SourceCitation]:
    sources = []
    for r in search_results:
        meta = r.get("metadata", r) if isinstance(r, dict) and "metadata" in r else (r if isinstance(r, dict) else {})
        dist = r.get("distance") if isinstance(r, dict) else None
        relevance_score = max(0.0, 1.0 - (dist / 2.0)) if dist is not None else None
        is_past = r.get("is_past_project", False) if isinstance(r, dict) else False
        sources.append(
            SourceCitation(
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
        )
    return sources


async def _classify_intent(request: QueryRequest) -> QueryIntentResult:
    vector_store = await get_vector_store()
    doc_types = await vector_store.distinct_document_types_for_project(request.project_name)
    return await QueryIntentService().classify_from_ollama(
        request.query,
        document_types_in_project=doc_types if doc_types else None,
    )


def _intent_info_from_result(
    ir: QueryIntentResult,
    request: QueryRequest,
    similar_projects: Optional[List[Dict[str, Any]]],
    *,
    n_current_override: Optional[int] = None,
    n_past_override: Optional[int] = None,
) -> QueryIntentInfo:
    if n_current_override is not None and n_past_override is not None:
        n_current, n_past = n_current_override, n_past_override
    else:
        cw, pw = _context_weighting_from_request(request)
        n_current = max(1, round(request.top_k * cw))
        n_past = max(0, request.top_k - n_current)
    slots_current = allocate_top_k_by_weights(ir.document_weights, n_current)
    past_models: List[PastProjectIntentSlots] = []
    sp_list = similar_projects or []
    if sp_list and n_past > 0:
        bases = past_project_slot_bases(n_past, len(sp_list))
        for sp, take in zip(sp_list, bases):
            pname = sp.get("project_name")
            if not pname or take <= 0:
                continue
            past_models.append(
                PastProjectIntentSlots(
                    project_name=pname,
                    budget=take,
                    slots=allocate_top_k_by_weights(ir.document_weights, take),
                )
            )
    return QueryIntentInfo(
        intent_summary=ir.intent_summary,
        document_weights=dict(ir.document_weights),
        priority_order=ir.priority_order,
        used_fallback=ir.used_fallback,
        n_current_slots=n_current,
        n_past_slots=n_past,
        slots_current_project=slots_current,
        slots_past_by_project=past_models,
    )


def _similar_projects_to_preview(raw: Optional[List[Dict[str, Any]]]) -> Optional[List[SimilarProjectPreview]]:
    if not raw:
        return None
    out: List[SimilarProjectPreview] = []
    for sp in raw:
        meta = sp.get("metadata")
        if meta is not None and hasattr(meta, "model_dump"):
            meta = meta.model_dump(mode="json")
        elif meta is not None and not isinstance(meta, dict):
            meta = None
        out.append(
            SimilarProjectPreview(
                project_name=sp["project_name"],
                customer=sp.get("customer"),
                similarity_score=float(sp.get("similarity_score", 0.0)),
                metadata=meta,
            )
        )
    return out


def _preview_chunks_from_result(result: Dict[str, Any]) -> List[PreviewChunk]:
    return [
        PreviewChunk(
            chunk_id=(c["metadata"].get("chunk_id") or ""),
            text=c["text"],
            project_name=c["metadata"].get("project_name", ""),
            customer=c["metadata"].get("_similar_project", {}).get("customer")
            if c.get("is_past_project")
            else c["metadata"].get("customer"),
            metadata={k: v for k, v in c["metadata"].items() if not k.startswith("_")},
            is_past_project=c.get("is_past_project", False),
        )
        for c in result["chunks"]
    ]


def _similar_from_result(result: Dict[str, Any]) -> List[SimilarProjectPreview]:
    return [
        SimilarProjectPreview(
            project_name=sp["project_name"],
            customer=sp.get("customer"),
            similarity_score=sp.get("similarity_score", 0.0),
            metadata=(
                sp.get("metadata").model_dump(mode="json")
                if hasattr(sp.get("metadata"), "model_dump")
                else (sp.get("metadata") or {})
            ),
        )
        for sp in result["similar_projects"]
    ]


@router.post("/retrieve-preview", response_model=RetrievalPreviewResponse, status_code=status.HTTP_200_OK)
async def retrieve_preview(request: QueryRequest):
    """
    Retrieval preview: candidate chunks (and similar past projects when include_past_projects).
    With use_query_intent + QUERY_INTENT_ENABLED, also returns intent summary and slot breakdown.
    """
    filters_dict = _filters_dict(request)
    intent_ir: Optional[QueryIntentResult] = None
    if _intent_will_run(request):
        try:
            intent_ir = await _classify_intent(request)
        except Exception as e:
            logger.exception("Intent classification failed during retrieve-preview")
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    weights = intent_ir.document_weights if intent_ir else None
    prio = intent_ir.priority_order if intent_ir else None

    try:
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
                preview_only=True,
                document_weights=weights,
                priority_order=prio,
                filters=filters_dict,
            )
            similar = _similar_from_result(result)
            intent_info = _intent_info_from_result(intent_ir, request, result["similar_projects"]) if intent_ir else None
        elif intent_ir:
            result = await query_service.retrieve_with_past_projects(
                query=request.query,
                project_name=request.project_name,
                top_k=request.top_k,
                current_project_weight=1.0,
                past_projects_weight=0.0,
                exclude_chunk_ids=request.exclude_chunk_ids,
                force_project=None,
                preview_only=True,
                document_weights=weights,
                priority_order=prio,
                filters=filters_dict,
            )
            similar = []
            intent_info = _intent_info_from_result(
                intent_ir,
                request,
                [],
                n_current_override=request.top_k,
                n_past_override=0,
            )
        else:
            query_embedding = embedding_service.generate_embedding(request.query).tolist()
            vector_store = await get_vector_store()
            chunks = await vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                project_name=request.project_name,
                document_type=None,
                filters=filters_dict,
            )
            result = {"similar_projects": [], "chunks": chunks}
            similar = []
            intent_info = None
    except Exception as e:
        logger.exception("Retrieve preview failed")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

    search_results, graph_expansion, _ = await _augment_with_graph_if_enabled(request, result["chunks"])
    result = {**result, "chunks": search_results}

    chunks = _preview_chunks_from_result(result)
    return RetrievalPreviewResponse(
        similar_projects=similar,
        chunks=chunks,
        query=request.query,
        project_name=request.project_name,
        query_intent=intent_info,
        graph_expansion=graph_expansion,
    )


@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def query_documents(request: QueryRequest):
    """Query documents using semantic search and RAG (optional Phase 3.75 query intent)."""
    total_start = time.time()

    try:
        embedding_start = time.time()
        search_results = None
        similar_projects_result: Optional[List[Dict[str, Any]]] = None
        filters_dict = _filters_dict(request)

        intent_ir: Optional[QueryIntentResult] = None
        if _intent_will_run(request):
            intent_ir = await _classify_intent(request)
        weights = intent_ir.document_weights if intent_ir else None
        prio = intent_ir.priority_order if intent_ir else None

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
                document_weights=weights,
                priority_order=prio,
                filters=filters_dict,
            )
            similar_projects_result = result["similar_projects"]
            search_results = result["chunks"]
        elif intent_ir:
            result = await query_service.retrieve_with_past_projects(
                query=request.query,
                project_name=request.project_name,
                top_k=request.top_k,
                current_project_weight=1.0,
                past_projects_weight=0.0,
                exclude_chunk_ids=request.exclude_chunk_ids,
                force_project=None,
                preview_only=False,
                document_weights=weights,
                priority_order=prio,
                filters=filters_dict,
            )
            similar_projects_result = []
            search_results = result["chunks"]
        else:
            query_embedding = embedding_service.generate_embedding(request.query).tolist()
            vector_store = await get_vector_store()
            search_results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=request.top_k,
                project_name=request.project_name,
                document_type=None,
                filters=filters_dict,
            )

        intent_info = None
        if intent_ir:
            if request.include_past_projects and request.project_name:
                intent_info = _intent_info_from_result(intent_ir, request, similar_projects_result)
            else:
                intent_info = _intent_info_from_result(
                    intent_ir,
                    request,
                    [],
                    n_current_override=request.top_k,
                    n_past_override=0,
                )

        search_results, graph_expansion, graph_context_lines = await _augment_with_graph_if_enabled(
            request,
            search_results or [],
        )

        embedding_time = time.time() - embedding_start
        search_time = 0.0

        if not search_results:
            total_time = time.time() - total_start
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                query=request.query,
                project_name=request.project_name,
                similar_projects=_similar_projects_to_preview(similar_projects_result),
                query_intent=intent_info,
                graph_expansion=graph_expansion,
            )

        format_start = time.time()
        context_chunks = _normalize_context_chunks(search_results)
        format_time = time.time() - format_start

        model_to_use = request.model or settings.OLLAMA_MODEL
        rag_service_instance = RAGService(model=model_to_use) if request.model else rag_service
        llm_start = time.time()
        answer = await rag_service_instance.generate_response(
            query=request.query,
            context_chunks=context_chunks,
            project_name=request.project_name,
            stream=False,
            graph_context_lines=graph_context_lines,
        )
        llm_time = time.time() - llm_start

        sources = _search_results_to_sources(search_results)

        total_time = time.time() - total_start
        return QueryResponse(
            answer=answer,
            sources=sources,
            query=request.query,
            project_name=request.project_name,
            similar_projects=_similar_projects_to_preview(similar_projects_result),
            query_intent=intent_info,
            graph_expansion=graph_expansion,
        )

    except Exception as e:
        total_time = time.time() - total_start
        logger.error(f"Query failed after {total_time:.3f}s: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}",
        )


@router.post("/stream", status_code=status.HTTP_200_OK)
async def query_documents_stream(request: QueryRequest):
    """Streaming query with SSE; sources event may include query_intent (Phase 3.75)."""
    total_start = time.time()

    async def generate_stream():
        try:
            embedding_start = time.time()
            similar_projects_stream: Optional[List[Dict[str, Any]]] = None
            filters_dict = _filters_dict(request)

            intent_ir: Optional[QueryIntentResult] = None
            if _intent_will_run(request):
                intent_ir = await _classify_intent(request)
            weights = intent_ir.document_weights if intent_ir else None
            prio = intent_ir.priority_order if intent_ir else None

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
                    document_weights=weights,
                    priority_order=prio,
                    filters=filters_dict,
                )
                similar_projects_stream = result["similar_projects"]
                search_results = result["chunks"]
            elif intent_ir:
                result = await query_service.retrieve_with_past_projects(
                    query=request.query,
                    project_name=request.project_name,
                    top_k=request.top_k,
                    current_project_weight=1.0,
                    past_projects_weight=0.0,
                    exclude_chunk_ids=request.exclude_chunk_ids,
                    force_project=None,
                    preview_only=False,
                    document_weights=weights,
                    priority_order=prio,
                    filters=filters_dict,
                )
                similar_projects_stream = []
                search_results = result["chunks"]
            else:
                query_embedding = embedding_service.generate_embedding(request.query).tolist()
                vector_store = await get_vector_store()
                search_results = await vector_store.search(
                    query_embedding=query_embedding,
                    top_k=request.top_k,
                    project_name=request.project_name,
                    document_type=None,
                    filters=filters_dict,
                )

            intent_info = None
            if intent_ir:
                if request.include_past_projects and request.project_name:
                    intent_info = _intent_info_from_result(intent_ir, request, similar_projects_stream)
                else:
                    intent_info = _intent_info_from_result(
                        intent_ir,
                        request,
                        [],
                        n_current_override=request.top_k,
                        n_past_override=0,
                    )

            search_results, graph_expansion, graph_context_lines = await _augment_with_graph_if_enabled(
                request,
                search_results or [],
            )

            embedding_time = time.time() - embedding_start
            search_time = 0.0
            logger.info(
                f"Retrieval completed in {embedding_time:.3f}s, found {len(search_results) if search_results else 0} results"
            )

            if not search_results:
                total_time = time.time() - total_start
                logger.info(f"Query completed in {total_time:.3f}s (no results found)")
                sources_data = {
                    "type": "sources",
                    "sources": [],
                    "query": request.query,
                    "project_name": request.project_name,
                }
                if intent_info:
                    sources_data["query_intent"] = intent_info.model_dump(mode="json")
                if graph_expansion:
                    sources_data["graph_expansion"] = graph_expansion.model_dump(mode="json")
                yield f"data: {json.dumps(sources_data)}\n\n"

                answer_data = {
                    "type": "chunk",
                    "text": "I couldn't find any relevant information to answer your question.",
                }
                yield f"data: {json.dumps(answer_data)}\n\n"

                done_data = {
                    "type": "done",
                    "total_time": round(total_time, 3),
                    "timings": {
                        "embedding": round(embedding_time, 3),
                        "search": round(search_time, 3),
                        "format": 0.0,
                        "llm": None,
                    },
                }
                yield f"data: {json.dumps(done_data)}\n\n"
                return

            sources = _search_results_to_sources(search_results)

            format_start = time.time()
            context_chunks = _normalize_context_chunks(search_results)
            format_time = time.time() - format_start

            sources_data = {
                "type": "sources",
                "sources": [s.model_dump() for s in sources],
                "query": request.query,
                "project_name": request.project_name,
            }
            if similar_projects_stream:
                rich = _similar_projects_to_preview(similar_projects_stream)
                if rich:
                    sources_data["similar_projects"] = [p.model_dump(mode="json") for p in rich]
            if intent_info:
                sources_data["query_intent"] = intent_info.model_dump(mode="json")
            if graph_expansion:
                sources_data["graph_expansion"] = graph_expansion.model_dump(mode="json")
            yield f"data: {json.dumps(sources_data)}\n\n"

            model_to_use = request.model or settings.OLLAMA_MODEL
            rag_service_instance = RAGService(model=model_to_use) if request.model else rag_service

            llm_start = time.time()
            try:
                async_gen = rag_service_instance.generate_response(
                    query=request.query,
                    context_chunks=context_chunks,
                    project_name=request.project_name,
                    stream=True,
                    graph_context_lines=graph_context_lines,
                )

                generator = await async_gen
                async for chunk in generator:
                    chunk_data = {"type": "chunk", "text": chunk}
                    yield f"data: {json.dumps(chunk_data)}\n\n"

                llm_time = time.time() - llm_start
                logger.info(f"LLM response streamed in {llm_time:.3f}s (model: {model_to_use})")

            except Exception as llm_error:
                llm_time = time.time() - llm_start
                logger.error(f"LLM streaming failed after {llm_time:.3f}s: {str(llm_error)}")
                error_data = {"type": "error", "message": f"LLM error: {str(llm_error)}"}
                yield f"data: {json.dumps(error_data)}\n\n"

            total_time = time.time() - total_start
            done_data = {
                "type": "done",
                "total_time": round(total_time, 3),
                "timings": {
                    "embedding": round(embedding_time, 3),
                    "search": round(search_time, 3),
                    "format": round(format_time, 3),
                    "llm": round(llm_time, 3) if "llm_time" in locals() else None,
                },
            }
            yield f"data: {json.dumps(done_data)}\n\n"

        except Exception as e:
            total_time = time.time() - total_start
            logger.error(f"Streaming query failed after {total_time:.3f}s: {str(e)}")
            error_data = {
                "type": "error",
                "message": f"Query failed: {str(e)}",
                "total_time": round(total_time, 3),
            }
            yield f"data: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
