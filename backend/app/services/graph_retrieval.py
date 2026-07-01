"""Phase 3–4: graph-augmented chunk retrieval (vector seeds + Neo4j expansion + Qdrant hydrate)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field

from app.config import settings
from app.services.graph_store import get_graph_store, is_graph_reachable
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)


class GraphAugmentationResult(BaseModel):
    """Merged retrieval output after optional graph expansion."""

    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    added_chunk_ids: List[str] = Field(default_factory=list)
    paths_summary: List[str] = Field(default_factory=list)
    graph_context_lines: List[str] = Field(default_factory=list)
    vector_chunk_count: int = 0
    graph_added_count: int = 0
    seed_count: int = 0
    degraded: bool = False
    degrade_reason: Optional[str] = None
    timing_ms: Dict[str, float] = Field(default_factory=dict)


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
    max_seeds = max(1, settings.GRAPH_MAX_SEEDS)
    return seeds[:max_seeds]


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


def _vector_only_result(
    vector_chunks: List[Dict[str, Any]],
    *,
    degraded: bool = False,
    degrade_reason: Optional[str] = None,
    timing_ms: Optional[Dict[str, float]] = None,
    paths_summary: Optional[List[str]] = None,
) -> GraphAugmentationResult:
    paths = paths_summary or []
    lines = format_graph_context_lines(paths)
    return GraphAugmentationResult(
        chunks=list(vector_chunks),
        paths_summary=paths[:20],
        graph_context_lines=lines,
        vector_chunk_count=len(vector_chunks),
        graph_added_count=0,
        degraded=degraded,
        degrade_reason=degrade_reason,
        timing_ms=timing_ms or {},
    )


async def _neighborhood_per_seed(
    store: Any,
    seeds: List[str],
    *,
    depth: int,
    limit: int,
    project_name: str,
) -> Any:
    """Expand each seed with a per-seed degree cap, then merge neighborhood results."""
    from app.services.graph_store import GraphNeighborhoodResult

    per_seed_limit = max(1, min(limit, settings.GRAPH_MAX_DEGREE_PER_SEED))
    merged_chunk_ids: List[str] = []
    merged_node_ids: List[str] = []
    merged_paths: List[str] = []
    seen_chunks: Set[str] = set()
    seen_nodes: Set[str] = set()

    for sid in seeds:
        part = await store.neighborhood(
            [sid],
            depth=depth,
            limit=per_seed_limit,
            project_name=project_name,
        )
        for cid in part.chunk_ids:
            if cid not in seen_chunks:
                seen_chunks.add(cid)
                merged_chunk_ids.append(cid)
        for nid in part.node_ids:
            if nid not in seen_nodes:
                seen_nodes.add(nid)
                merged_node_ids.append(nid)
        for path in part.paths_summary:
            if path not in merged_paths:
                merged_paths.append(path)

    return GraphNeighborhoodResult(
        chunk_ids=merged_chunk_ids[:limit],
        node_ids=merged_node_ids[:limit],
        paths_summary=merged_paths[:20],
    )


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

    Degrades to vector-only when graph is disabled, Neo4j unreachable, seeds are empty,
    or graph work times out / fails.
    """
    t_total = time.perf_counter()
    vector_count = len(vector_chunks)

    if not vector_chunks or not project_name.strip():
        return _vector_only_result(vector_chunks)

    if not settings.GRAPH_ENABLED:
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="graph_disabled",
        )

    if not await is_graph_reachable():
        logger.warning(
            "Graph augmentation skipped: Neo4j unreachable for project %r",
            project_name,
        )
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="neo4j_unreachable",
            timing_ms={"total_ms": round((time.perf_counter() - t_total) * 1000, 2)},
        )

    seeds = _seed_chunk_ids(vector_chunks, project_name)
    if not seeds:
        return _vector_only_result(vector_chunks)

    budget = extra_budget if extra_budget is not None else settings.GRAPH_RETRIEVAL_EXTRA_BUDGET
    depth = depth if depth is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH
    limit = neighborhood_limit if neighborhood_limit is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_LIMIT
    timeout = settings.GRAPH_QUERY_TIMEOUT_SECONDS

    existing_ids: Set[str] = set()
    for c in vector_chunks:
        cid = _chunk_id_from_result(c)
        if cid:
            existing_ids.add(cid)

    timing_ms: Dict[str, float] = {}
    neighborhood = None

    try:
        store = await get_graph_store()
        t_neighborhood = time.perf_counter()
        neighborhood = await asyncio.wait_for(
            _neighborhood_per_seed(
                store,
                seeds,
                depth=depth,
                limit=limit,
                project_name=project_name.strip(),
            ),
            timeout=timeout,
        )
        timing_ms["neighborhood_ms"] = round((time.perf_counter() - t_neighborhood) * 1000, 2)
    except asyncio.TimeoutError:
        logger.warning(
            "Graph neighborhood timed out after %.1fs for project %r (seeds=%d)",
            timeout,
            project_name,
            len(seeds),
        )
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="graph_query_timeout",
            timing_ms=timing_ms,
        )
    except Exception as exc:
        logger.warning(
            "Graph neighborhood expansion failed for project %r: %s",
            project_name,
            exc,
        )
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="graph_neighborhood_error",
            timing_ms=timing_ms,
        )

    selected_ids = _select_graph_chunk_ids(
        neighborhood.chunk_ids,
        existing_ids,
        vector_chunks,
        max(0, budget),
    )
    if not selected_ids:
        lines = format_graph_context_lines(neighborhood.paths_summary)
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=neighborhood.paths_summary[:20],
            graph_context_lines=lines,
            vector_chunk_count=vector_count,
            graph_added_count=0,
            seed_count=len(seeds),
            timing_ms=timing_ms,
        )

    try:
        vector_store = await get_vector_store()
        t_hydrate = time.perf_counter()
        graph_chunks = await vector_store.get_chunks_by_ids(selected_ids)
        timing_ms["hydrate_ms"] = round((time.perf_counter() - t_hydrate) * 1000, 2)
    except Exception as exc:
        logger.warning("Failed to fetch graph-expanded chunks from Qdrant: %s", exc)
        lines = format_graph_context_lines(neighborhood.paths_summary)
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=neighborhood.paths_summary[:20],
            graph_context_lines=lines,
            vector_chunk_count=vector_count,
            graph_added_count=0,
            seed_count=len(seeds),
            degraded=True,
            degrade_reason="qdrant_hydrate_error",
            timing_ms=timing_ms,
        )

    merged = merge_vector_and_graph_chunks(vector_chunks, graph_chunks)
    added = [
        cid for cid in selected_ids
        if cid not in existing_ids and any(_chunk_id_from_result(c) == cid for c in graph_chunks)
    ]
    lines = format_graph_context_lines(neighborhood.paths_summary)
    timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

    logger.info(
        "Graph augmentation project=%r seeds=%d vector=%d added=%d degraded=false "
        "timing_ms=%s",
        project_name,
        len(seeds),
        vector_count,
        len(added),
        timing_ms,
    )
    return GraphAugmentationResult(
        chunks=merged,
        added_chunk_ids=added,
        paths_summary=neighborhood.paths_summary[:20],
        graph_context_lines=lines,
        vector_chunk_count=vector_count,
        graph_added_count=len(added),
        seed_count=len(seeds),
        timing_ms=timing_ms,
    )
