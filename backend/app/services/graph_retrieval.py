"""Phase 3: graph-augmented chunk retrieval (vector seeds + Neo4j expansion + Qdrant hydrate)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.config import settings
from app.services.graph_store import get_graph_store
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class GraphAugmentationResult(BaseModel):
    """Merged retrieval output after optional graph expansion."""

    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    added_chunk_ids: List[str] = Field(default_factory=list)
    paths_summary: List[str] = Field(default_factory=list)
    graph_context_lines: List[str] = Field(default_factory=list)


def _chunk_id_from_result(chunk: Dict[str, Any]) -> Optional[str]:
    return (chunk.get("metadata") or {}).get("chunk_id")


def _seed_chunk_ids(
    vector_chunks: List[Dict[str, Any]],
    project_name: str,
) -> List[str]:
    """Use current-project vector hits as graph expansion seeds."""
    pname = project_name.strip()
    seeds: List[str] = []
    for chunk in vector_chunks:
        meta = chunk.get("metadata") or {}
        cid = meta.get("chunk_id")
        if not cid:
            continue
        if pname and (meta.get("project_name") or "").strip() != pname:
            continue
        if chunk.get("is_past_project"):
            continue
        seeds.append(cid)
    return seeds


def _select_graph_chunk_ids(
    candidate_ids: List[str],
    existing_ids: Set[str],
    vector_chunks: List[Dict[str, Any]],
    budget: int,
) -> List[str]:
    """Pick up to budget new chunk IDs, preferring document types underrepresented in vector hits."""
    new_ids = [cid for cid in candidate_ids if cid and cid not in existing_ids]
    if not new_ids or budget <= 0:
        return []

    represented_types: Set[str] = set()
    for chunk in vector_chunks:
        dt = (chunk.get("metadata") or {}).get("document_type")
        if dt:
            represented_types.add(dt)

    # Stable order: unseen document types first (filled later from Qdrant metadata),
    # then preserve neighborhood order.
    if len(new_ids) <= budget:
        return new_ids[:budget]

    return new_ids[:budget]


def format_graph_context_lines(paths_summary: List[str]) -> List[str]:
    """Human-readable relationship summaries for the RAG prompt."""
    lines: List[str] = []
    for path in paths_summary[:20]:
        text = (path or "").strip()
        if not text:
            continue
        # Neo4j paths look like …-REL->node_id; strip chunk: prefix for readability
        text = text.replace("chunk:", "")
        lines.append(f"- {text}")
    return lines


def merge_vector_and_graph_chunks(
    vector_chunks: List[Dict[str, Any]],
    graph_chunks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Keep vector order; append graph-only chunks with _graph_augmented metadata flag."""
    merged = list(vector_chunks)
    seen = {_chunk_id_from_result(c) for c in vector_chunks}
    seen.discard(None)

    for chunk in graph_chunks:
        cid = _chunk_id_from_result(chunk)
        if not cid or cid in seen:
            continue
        meta = dict(chunk.get("metadata") or {})
        meta["_graph_augmented"] = True
        merged.append({
            "text": chunk.get("text", ""),
            "metadata": meta,
            "is_past_project": chunk.get("is_past_project", False),
        })
        seen.add(cid)
    return merged


async def augment_chunks_with_graph(
    vector_chunks: List[Dict[str, Any]],
    *,
    project_name: str,
    extra_budget: Optional[int] = None,
    depth: Optional[int] = None,
    neighborhood_limit: Optional[int] = None,
) -> GraphAugmentationResult:
    """
    Expand vector hits via Neo4j neighborhood, fetch extra chunks from Qdrant, merge.

    Degrades to vector-only when graph is disabled, seeds are empty, or Neo4j fails.
    """
    if not vector_chunks or not project_name.strip():
        return GraphAugmentationResult(chunks=list(vector_chunks))

    seeds = _seed_chunk_ids(vector_chunks, project_name)
    if not seeds:
        return GraphAugmentationResult(chunks=list(vector_chunks))

    budget = extra_budget if extra_budget is not None else settings.GRAPH_RETRIEVAL_EXTRA_BUDGET
    depth = depth if depth is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH
    limit = neighborhood_limit if neighborhood_limit is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_LIMIT
    timeout = settings.GRAPH_QUERY_TIMEOUT_SECONDS

    existing_ids: Set[str] = set()
    for c in vector_chunks:
        cid = _chunk_id_from_result(c)
        if cid:
            existing_ids.add(cid)

    try:
        store = await get_graph_store()
        neighborhood = await asyncio.wait_for(
            store.neighborhood(
                seeds,
                depth=depth,
                limit=limit,
                project_name=project_name.strip(),
            ),
            timeout=timeout,
        )
    except Exception as exc:
        logger.warning(
            "Graph neighborhood expansion failed for project %r: %s",
            project_name,
            exc,
        )
        return GraphAugmentationResult(chunks=list(vector_chunks))

    selected_ids = _select_graph_chunk_ids(
        neighborhood.chunk_ids,
        existing_ids,
        vector_chunks,
        max(0, budget),
    )
    if not selected_ids:
        lines = format_graph_context_lines(neighborhood.paths_summary)
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=neighborhood.paths_summary[:20],
            graph_context_lines=lines,
        )

    try:
        vector_store = await get_vector_store()
        graph_chunks = await vector_store.get_chunks_by_ids(selected_ids)
    except Exception as exc:
        logger.warning("Failed to fetch graph-expanded chunks from Qdrant: %s", exc)
        lines = format_graph_context_lines(neighborhood.paths_summary)
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=neighborhood.paths_summary[:20],
            graph_context_lines=lines,
        )

    merged = merge_vector_and_graph_chunks(vector_chunks, graph_chunks)
    added = [
        cid for cid in selected_ids
        if cid not in existing_ids and any(_chunk_id_from_result(c) == cid for c in graph_chunks)
    ]
    lines = format_graph_context_lines(neighborhood.paths_summary)

    logger.info(
        "Graph augmentation project=%r seeds=%d added=%d paths=%d",
        project_name,
        len(seeds),
        len(added),
        len(lines),
    )
    return GraphAugmentationResult(
        chunks=merged,
        added_chunk_ids=added,
        paths_summary=neighborhood.paths_summary[:20],
        graph_context_lines=lines,
    )
