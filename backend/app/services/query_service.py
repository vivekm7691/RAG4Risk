"""Query service for multi-stage retrieval with past projects (Phase 3.5) and intent weights (Phase 3.75)."""

import logging
from typing import List, Dict, Any, Optional, Set

from app.config import settings
from app.models.document import DocumentType
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.project_similarity import ProjectSimilarityService, get_shared_project_similarity_service
from app.services.query_intent_service import allocate_top_k_by_weights

logger = logging.getLogger(__name__)


def _chunk_id(chunk: Dict[str, Any]) -> Optional[str]:
    return (chunk.get("metadata") or {}).get("chunk_id")


def past_project_slot_bases(n_past: int, num_similar_projects: int) -> List[int]:
    """Round-robin split of n_past retrieval slots across similar projects."""
    if num_similar_projects <= 0 or n_past <= 0:
        return []
    bases = [0] * num_similar_projects
    for i in range(n_past):
        bases[i % num_similar_projects] += 1
    return bases


def _chunk_sort_key(c: Dict[str, Any], priority_order: Optional[List[str]]) -> tuple:
    dist = c.get("distance")
    if dist is None:
        dist = 1.0
    if not priority_order:
        return (dist, 0)
    dt = (c.get("metadata") or {}).get("document_type")
    try:
        idx = priority_order.index(dt)
    except ValueError:
        idx = len(priority_order)
    return (dist, idx)


class QueryService:
    """Orchestrates retrieval: current project + similar past projects, with weighting and filtering."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        project_similarity_service: Optional[ProjectSimilarityService] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.similarity_service = project_similarity_service or get_shared_project_similarity_service()

    async def _retrieve_weighted_for_project(
        self,
        vector_store: VectorStore,
        query_embedding: List[float],
        project_name: str,
        n_slots: int,
        document_weights: Dict[str, float],
        exclude_chunk_ids: Set[str],
        filters: Optional[Dict[str, Any]],
        is_past_project: bool,
        similar_project: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Per-type Qdrant searches, dedupe by chunk_id (best distance), same weights as current (Phase 3.75)."""
        logger.info("_retrieve_weighted_for_project: project=%r n_slots=%d weights=%s", project_name, n_slots, document_weights)
        if n_slots <= 0 or not project_name:
            logger.warning("_retrieve_weighted_for_project: early return (n_slots=%d, project_name=%r)", n_slots, project_name)
            return []

        allocation = allocate_top_k_by_weights(document_weights, n_slots)
        logger.info("_retrieve_weighted_for_project: allocation=%s", allocation)
        best_by_id: Dict[str, Dict[str, Any]] = {}

        for doc_type_str, ki in allocation.items():
            if ki <= 0:
                continue
            try:
                dt = DocumentType(doc_type_str)
            except ValueError:
                continue
            results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=ki,
                project_name=project_name,
                document_type=dt,
                filters=filters,
            )
            for r in results:
                cid = (r.get("metadata") or {}).get("chunk_id")
                if cid and cid in exclude_chunk_ids:
                    continue
                dist = r.get("distance")
                if dist is None:
                    dist = 1.0
                meta = dict(r.get("metadata") or {})
                if is_past_project and similar_project is not None:
                    meta["_similar_project"] = similar_project
                chunk = {
                    "text": r["text"],
                    "metadata": meta,
                    "distance": dist,
                    "is_past_project": is_past_project,
                }
                key = cid if cid else f"__noid_{id(r)}__"
                prev = best_by_id.get(key)
                if prev is None or dist < (prev.get("distance") or 1.0):
                    best_by_id[key] = chunk

        logger.info("_retrieve_weighted_for_project: returning %d chunks", len(best_by_id))
        return list(best_by_id.values())

    async def retrieve_with_past_projects(
        self,
        query: str,
        project_name: Optional[str],
        top_k: int = 10,
        current_project_weight: float = 0.7,
        past_projects_weight: float = 0.3,
        exclude_chunk_ids: Optional[List[str]] = None,
        force_project: Optional[Dict[str, str]] = None,
        preview_only: bool = False,
        document_weights: Optional[Dict[str, float]] = None,
        priority_order: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-stage retrieval: current project chunks + past project chunks, with optional preview.

        When ``document_weights`` is set (Phase 3.75), allocates each stage's budget across
        document types and runs filtered vector search per type; same allocation applies to
        each similar past project slice. When unset, behavior matches the original single
        unfiltered search per project.

        Returns:
            - similar_projects: list of {project_name, customer, similarity_score, metadata}
            - chunks: list of {text, metadata, is_past_project} with chunk_id in metadata
            - If preview_only=True, no LLM is called; chunks are candidate context for confirmation.
        """
        _ = preview_only  # API parity with retrieve-preview; retrieval path is identical
        exclude_chunk_ids = set(exclude_chunk_ids or [])
        vector_store = await get_vector_store()
        query_embedding = self.embedding_service.generate_embedding(query).tolist()

        n_current = max(1, round(top_k * current_project_weight))
        n_past = max(0, top_k - n_current)

        similar_projects: List[Dict[str, Any]] = []
        all_chunks: List[Dict[str, Any]] = []

        use_weights = bool(document_weights)
        logger.info("retrieve_with_past_projects: project=%r top_k=%d n_current=%d n_past=%d use_weights=%s", project_name, top_k, n_current, n_past, use_weights)

        # Stage 1: current project chunks
        if use_weights and document_weights is not None:
            current_list = await self._retrieve_weighted_for_project(
                vector_store,
                query_embedding,
                project_name or "",
                n_current,
                document_weights,
                exclude_chunk_ids,
                filters,
                is_past_project=False,
                similar_project=None,
            )
            all_chunks.extend(current_list)
        else:
            current_results = await vector_store.search(
                query_embedding=query_embedding,
                top_k=n_current,
                project_name=project_name,
                document_type=None,
                filters=filters,
            )
            for r in current_results:
                chunk_id = (r.get("metadata") or {}).get("chunk_id")
                if chunk_id and chunk_id in exclude_chunk_ids:
                    continue
                all_chunks.append({
                    "text": r["text"],
                    "metadata": r.get("metadata") or {},
                    "distance": r.get("distance"),
                    "is_past_project": False,
                })

        # Stage 2 & 3: similar projects and their chunks (total past slots = n_past)
        if n_past > 0 and project_name:
            similar_projects = await self.similarity_service.find_similar_projects(
                project_name=project_name,
                top_k=settings.SIMILAR_PROJECTS_COUNT,
                exclude_project_names=[],
                force_project=force_project,
                vector_store=vector_store,
            )
            if similar_projects:
                n_sp = len(similar_projects)
                bases = past_project_slot_bases(n_past, n_sp)
                for sp, take in zip(similar_projects, bases):
                    if take <= 0:
                        continue
                    pname = sp.get("project_name")
                    if not pname:
                        continue
                    if use_weights and document_weights is not None:
                        past_list = await self._retrieve_weighted_for_project(
                            vector_store,
                            query_embedding,
                            pname,
                            take,
                            document_weights,
                            exclude_chunk_ids,
                            filters,
                            is_past_project=True,
                            similar_project=sp,
                        )
                        all_chunks.extend(past_list)
                    else:
                        past_results = await vector_store.search(
                            query_embedding=query_embedding,
                            top_k=take,
                            project_name=pname,
                            document_type=None,
                            filters=filters,
                        )
                        for r in past_results:
                            chunk_id = (r.get("metadata") or {}).get("chunk_id")
                            if chunk_id and chunk_id in exclude_chunk_ids:
                                continue
                            meta = r.get("metadata") or {}
                            meta["_similar_project"] = sp
                            all_chunks.append({
                                "text": r["text"],
                                "metadata": meta,
                                "distance": r.get("distance"),
                                "is_past_project": True,
                            })

        # Backfill from current project if we are short (e.g. exclusions or sparse past results)
        seen_ids: Set[str] = {cid for cid in (_chunk_id(c) for c in all_chunks) if cid}
        if len(all_chunks) < top_k and project_name:
            need = top_k - len(all_chunks)
            fetch_limit = min(max(need * 4, top_k), 50)
            extra = await vector_store.search(
                query_embedding=query_embedding,
                top_k=fetch_limit,
                project_name=project_name,
                document_type=None,
                filters=filters,
            )
            for r in extra:
                if len(all_chunks) >= top_k:
                    break
                chunk_id = (r.get("metadata") or {}).get("chunk_id")
                if chunk_id and chunk_id in exclude_chunk_ids:
                    continue
                if chunk_id and chunk_id in seen_ids:
                    continue
                if not chunk_id:
                    continue
                seen_ids.add(chunk_id)
                all_chunks.append({
                    "text": r["text"],
                    "metadata": r.get("metadata") or {},
                    "distance": r.get("distance"),
                    "is_past_project": False,
                })

        if use_weights and priority_order:
            all_chunks.sort(key=lambda c: _chunk_sort_key(c, priority_order))
        else:
            all_chunks.sort(key=lambda c: c.get("distance") or 1.0)
        all_chunks = all_chunks[:top_k]

        return {
            "similar_projects": similar_projects,
            "chunks": all_chunks,
        }
