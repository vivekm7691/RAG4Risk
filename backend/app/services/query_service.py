"""Query service for multi-stage retrieval with past projects (Phase 3.5)."""

import logging
from typing import List, Dict, Any, Optional

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import get_vector_store, VectorStore
from app.services.project_similarity import ProjectSimilarityService

logger = logging.getLogger(__name__)


class QueryService:
    """Orchestrates retrieval: current project + similar past projects, with weighting and filtering."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        project_similarity_service: Optional[ProjectSimilarityService] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.similarity_service = project_similarity_service or ProjectSimilarityService()

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
    ) -> Dict[str, Any]:
        """
        Multi-stage retrieval: current project chunks + past project chunks, with optional preview.

        Returns:
            - similar_projects: list of {project_name, customer, similarity_score, metadata}
            - chunks: list of {text, metadata, is_past_project} with chunk_id in metadata
            - If preview_only=True, no LLM is called; chunks are candidate context for confirmation.
        """
        exclude_chunk_ids = set(exclude_chunk_ids or [])
        vector_store = await get_vector_store()
        query_embedding = self.embedding_service.generate_embedding(query).tolist()

        n_current = max(1, round(top_k * current_project_weight))
        n_past = max(0, top_k - n_current)

        similar_projects: List[Dict[str, Any]] = []
        all_chunks: List[Dict[str, Any]] = []

        # Stage 1: current project chunks
        current_results = await vector_store.search(
            query_embedding=query_embedding,
            top_k=n_current,
            project_name=project_name,
            document_type=None,
            filters=None,
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

        # Stage 2 & 3: similar projects and their chunks
        if n_past > 0 and project_name:
            similar_projects = await self.similarity_service.find_similar_projects(
                project_name=project_name,
                top_k=settings.SIMILAR_PROJECTS_COUNT,
                exclude_project_names=[],
                force_project=force_project,
                vector_store=vector_store,
            )
            if similar_projects:
                # How many chunks to take from past projects total
                per_project = max(1, n_past // len(similar_projects))
                for sp in similar_projects:
                    pname = sp.get("project_name")
                    if not pname:
                        continue
                    past_results = await vector_store.search(
                        query_embedding=query_embedding,
                        top_k=per_project,
                        project_name=pname,
                        document_type=None,
                        filters=None,
                    )
                    for r in past_results:
                        chunk_id = (r.get("metadata") or {}).get("chunk_id")
                        if chunk_id and chunk_id in exclude_chunk_ids:
                            continue
                        meta = r.get("metadata") or {}
                        meta["_similar_project"] = sp  # for customer/metadata in preview
                        all_chunks.append({
                            "text": r["text"],
                            "metadata": meta,
                            "distance": r.get("distance"),
                            "is_past_project": True,
                        })

        # Sort by distance (lower = more relevant) and cap at top_k
        all_chunks.sort(key=lambda c: c.get("distance") or 1.0)
        all_chunks = all_chunks[:top_k]

        return {
            "similar_projects": similar_projects,
            "chunks": all_chunks,
        }
