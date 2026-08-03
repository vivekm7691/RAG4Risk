"""Phase 3–5.1: graph-augmented chunk retrieval (vector seeds + Neo4j expansion + Qdrant hydrate)."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from app.config import settings
from app.services.graph_store import get_graph_store, is_graph_reachable
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

_VALID_MERGE_STRATEGIES = frozenset({"append", "rrf", "diversity"})
_RRF_K = 60


class GraphAugmentationResult(BaseModel):
    """Merged retrieval output after optional graph expansion."""

    chunks: List[Dict[str, Any]] = Field(default_factory=list)
    added_chunk_ids: List[str] = Field(default_factory=list)
    paths_summary: List[str] = Field(default_factory=list)
    graph_context_lines: List[str] = Field(default_factory=list)
    vector_chunk_count: int = 0
    graph_added_count: int = 0
    seed_count: int = 0
    text_seed_count: int = 0
    degraded: bool = False
    degrade_reason: Optional[str] = None
    timing_ms: Dict[str, float] = Field(default_factory=dict)
    merge_strategy: str = "append"
    chunk_sources: Dict[str, str] = Field(
        default_factory=dict,
        description="chunk_id -> vector | graph | both",
    )


def _chunk_id_from_result(chunk: Dict[str, Any]) -> Optional[str]:
    return (chunk.get("metadata") or {}).get("chunk_id")


def _normalize_merge_strategy(value: Optional[str]) -> str:
    strategy = (value or settings.GRAPH_MERGE_STRATEGY or "append").strip().lower()
    if strategy not in _VALID_MERGE_STRATEGIES:
        logger.warning("Unknown GRAPH_MERGE_STRATEGY=%r; using append", value)
        return "append"
    return strategy


def _seed_chunk_ids(
    vector_chunks: List[Dict[str, Any]],
    project_name: str,
    *,
    seed_chunk_ids: Optional[List[str]] = None,
) -> List[str]:
    """Use current-project vector hits (or explicit override) as graph expansion seeds."""
    max_seeds = max(1, settings.GRAPH_MAX_SEEDS)
    if seed_chunk_ids is not None:
        return [cid for cid in seed_chunk_ids if cid][:max_seeds]

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
    return seeds[:max_seeds]


def _select_graph_chunk_ids(
    candidate_ids: List[str],
    existing_ids: Set[str],
    vector_chunks: List[Dict[str, Any]],
    budget: int,
    *,
    graph_chunk_types: Optional[Dict[str, str]] = None,
) -> List[str]:
    """Pick up to budget new chunk IDs; prefer underrepresented document types when known."""
    new_ids = [cid for cid in candidate_ids if cid and cid not in existing_ids]
    if not new_ids or budget <= 0:
        return []

    if len(new_ids) <= budget:
        return new_ids[:budget]

    represented_types: Set[str] = set()
    for chunk in vector_chunks:
        dt = (chunk.get("metadata") or {}).get("document_type")
        if dt:
            represented_types.add(dt)

    types = graph_chunk_types or {}
    novel: List[str] = []
    familiar: List[str] = []
    for cid in new_ids:
        dt = types.get(cid)
        if dt and dt not in represented_types:
            novel.append(cid)
        else:
            familiar.append(cid)
    ordered = novel + familiar
    return ordered[:budget]


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


def _annotate_chunk(
    chunk: Dict[str, Any],
    *,
    graph_augmented: bool = False,
) -> Dict[str, Any]:
    meta = dict(chunk.get("metadata") or {})
    if graph_augmented:
        meta["_graph_augmented"] = True
    return {
        "text": chunk.get("text", ""),
        "metadata": meta,
        "is_past_project": chunk.get("is_past_project", False),
    }


def merge_vector_and_graph_chunks(
    vector_chunks: List[Dict[str, Any]],
    graph_chunks: List[Dict[str, Any]],
    *,
    strategy: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Merge vector and graph-hydrated chunks.

    Strategies:
    - append: vector order, then graph-only extras
    - rrf: reciprocal rank fusion of vector rank + graph candidate order
    - diversity: append extras preferring underrepresented document_type first
    """
    merge_strategy = _normalize_merge_strategy(strategy)
    vector_ids = [_chunk_id_from_result(c) for c in vector_chunks]
    vector_id_set = {cid for cid in vector_ids if cid}
    graph_by_id: Dict[str, Dict[str, Any]] = {}
    graph_order: List[str] = []
    for chunk in graph_chunks:
        cid = _chunk_id_from_result(chunk)
        if not cid or cid in graph_by_id:
            continue
        graph_by_id[cid] = chunk
        graph_order.append(cid)

    sources: Dict[str, str] = {}
    for cid in vector_id_set:
        sources[cid] = "both" if cid in graph_by_id else "vector"
    for cid in graph_order:
        if cid not in sources:
            sources[cid] = "graph"

    if merge_strategy == "rrf":
        scores: Dict[str, float] = {}
        for rank, cid in enumerate(vector_ids):
            if not cid:
                continue
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        for rank, cid in enumerate(graph_order):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)

        vector_by_id = {
            cid: chunk
            for chunk, cid in zip(vector_chunks, vector_ids)
            if cid
        }
        ranked_ids = sorted(scores.keys(), key=lambda c: (-scores[c], c))
        merged: List[Dict[str, Any]] = []
        for cid in ranked_ids:
            if cid in vector_by_id:
                merged.append(_annotate_chunk(vector_by_id[cid]))
            elif cid in graph_by_id:
                merged.append(_annotate_chunk(graph_by_id[cid], graph_augmented=True))
        return merged, sources

    # append and diversity both keep vector order first
    merged = [_annotate_chunk(c) for c in vector_chunks]
    seen: Set[str] = set(vector_id_set)

    extras = [cid for cid in graph_order if cid not in seen]
    if merge_strategy == "diversity" and extras:
        represented: Set[str] = set()
        for chunk in vector_chunks:
            dt = (chunk.get("metadata") or {}).get("document_type")
            if dt:
                represented.add(dt)
        novel: List[str] = []
        familiar: List[str] = []
        for cid in extras:
            dt = (graph_by_id[cid].get("metadata") or {}).get("document_type")
            if dt and dt not in represented:
                novel.append(cid)
                represented.add(dt)
            else:
                familiar.append(cid)
        extras = novel + familiar

    for cid in extras:
        merged.append(_annotate_chunk(graph_by_id[cid], graph_augmented=True))
        seen.add(cid)

    return merged, sources


def _vector_only_result(
    vector_chunks: List[Dict[str, Any]],
    *,
    degraded: bool = False,
    degrade_reason: Optional[str] = None,
    timing_ms: Optional[Dict[str, float]] = None,
    paths_summary: Optional[List[str]] = None,
    merge_strategy: Optional[str] = None,
) -> GraphAugmentationResult:
    paths = paths_summary or []
    lines = format_graph_context_lines(paths)
    strategy = _normalize_merge_strategy(merge_strategy)
    sources = {
        cid: "vector"
        for cid in (_chunk_id_from_result(c) for c in vector_chunks)
        if cid
    }
    return GraphAugmentationResult(
        chunks=list(vector_chunks),
        paths_summary=paths[:20],
        graph_context_lines=lines,
        vector_chunk_count=len(vector_chunks),
        graph_added_count=0,
        degraded=degraded,
        degrade_reason=degrade_reason,
        timing_ms=timing_ms or {},
        merge_strategy=strategy,
        chunk_sources=sources,
    )


async def _neighborhood_per_seed(
    store: Any,
    seeds: List[str],
    *,
    depth: int,
    limit: int,
    project_name: str,
    relationship_types: Optional[List[str]] = None,
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
            relationship_types=relationship_types,
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


def _cross_link_relationship_types() -> Optional[List[str]]:
    """Bridge + whitelist when GRAPH_CROSS_LINK_ENABLED; else unrestricted (None)."""
    if settings.GRAPH_CROSS_LINK_ENABLED is not True:
        return None
    # Need EXTRACTED_FROM / CONTAINS_CHUNK to hop chunk ↔ entity ↔ chunk across docs
    bridge = {"EXTRACTED_FROM", "CONTAINS_CHUNK"}
    raw = (settings.GRAPH_CROSS_LINK_EDGE_WHITELIST or "").strip()
    if not isinstance(raw, str):
        raw = ""
    whitelist = {p.strip() for p in raw.split(",") if p.strip()}
    return sorted(bridge | whitelist)


async def _text_search_chunk_candidates(
    store: Any,
    query: str,
    *,
    project_name: str,
    timing_ms: Dict[str, float],
) -> Tuple[List[str], List[str], int]:
    """Optional full-text seeds → linked chunk IDs. Returns (chunk_ids, paths, text_seed_count)."""
    if not settings.GRAPH_TEXT_SEARCH_SEED_ENABLED:
        return [], [], 0
    q = (query or "").strip()
    if not q:
        return [], [], 0

    t0 = time.perf_counter()
    try:
        text_limit = max(1, settings.GRAPH_TEXT_SEARCH_SEED_LIMIT)
        text_result = await store.text_search_seed(
            q,
            text_limit,
            project_name=project_name,
        )
        timing_ms["text_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    except Exception as exc:
        timing_ms["text_search_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        logger.warning("Graph text_search_seed failed: %s", exc)
        return [], [], 0

    node_ids = list(text_result.node_ids or [])
    if not node_ids:
        return [], [], 0

    t1 = time.perf_counter()
    try:
        linked = await store.chunks_linked_to_nodes(
            node_ids,
            project_name=project_name,
            limit=settings.GRAPH_NEIGHBORHOOD_DEFAULT_LIMIT,
        )
        timing_ms["text_link_ms"] = round((time.perf_counter() - t1) * 1000, 2)
    except Exception as exc:
        timing_ms["text_link_ms"] = round((time.perf_counter() - t1) * 1000, 2)
        logger.warning("Graph chunks_linked_to_nodes failed: %s", exc)
        return [], [], len(node_ids)

    return list(linked.chunk_ids or []), list(linked.paths_summary or []), len(node_ids)


async def augment_chunks_with_graph(
    vector_chunks: List[Dict[str, Any]],
    *,
    project_name: str,
    query: str = "",
    seed_chunk_ids: Optional[List[str]] = None,
    extra_budget: Optional[int] = None,
    depth: Optional[int] = None,
    neighborhood_limit: Optional[int] = None,
    merge_strategy: Optional[str] = None,
) -> GraphAugmentationResult:
    """
    Expand vector hits via Neo4j neighborhood (and optional text-search seeds),
    fetch extra chunks from Qdrant, merge.

    Degrades to vector-only when graph is disabled, Neo4j unreachable, seeds are empty,
    or graph work times out / fails.
    """
    t_total = time.perf_counter()
    vector_count = len(vector_chunks)
    strategy = _normalize_merge_strategy(merge_strategy)

    if not project_name.strip():
        return _vector_only_result(vector_chunks, merge_strategy=strategy)

    if not settings.GRAPH_ENABLED:
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="graph_disabled",
            merge_strategy=strategy,
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
            merge_strategy=strategy,
        )

    seeds = _seed_chunk_ids(vector_chunks, project_name, seed_chunk_ids=seed_chunk_ids)
    budget = extra_budget if extra_budget is not None else settings.GRAPH_RETRIEVAL_EXTRA_BUDGET
    depth = depth if depth is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH
    if (
        settings.GRAPH_CROSS_LINK_ENABLED is True
        and isinstance(depth, int)
        and isinstance(settings.GRAPH_CROSS_LINK_MAX_DEPTH, int)
        and depth == settings.GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH
    ):
        depth = max(depth, settings.GRAPH_CROSS_LINK_MAX_DEPTH)
    limit = neighborhood_limit if neighborhood_limit is not None else settings.GRAPH_NEIGHBORHOOD_DEFAULT_LIMIT
    timeout = settings.GRAPH_QUERY_TIMEOUT_SECONDS
    pname = project_name.strip()
    relationship_types = _cross_link_relationship_types()

    existing_ids: Set[str] = set()
    for c in vector_chunks:
        cid = _chunk_id_from_result(c)
        if cid:
            existing_ids.add(cid)

    timing_ms: Dict[str, float] = {}
    text_seed_count = 0
    all_candidate_ids: List[str] = []
    all_paths: List[str] = []
    seen_candidates: Set[str] = set()
    seen_paths: Set[str] = set()

    def _absorb_chunk_ids(ids: List[str]) -> None:
        for cid in ids:
            if cid and cid not in seen_candidates:
                seen_candidates.add(cid)
                all_candidate_ids.append(cid)

    def _absorb_paths(paths: List[str]) -> None:
        for path in paths:
            if path and path not in seen_paths:
                seen_paths.add(path)
                all_paths.append(path)

    try:
        store = await get_graph_store()

        text_chunks, text_paths, text_seed_count = await asyncio.wait_for(
            _text_search_chunk_candidates(
                store,
                query,
                project_name=pname,
                timing_ms=timing_ms,
            ),
            timeout=timeout,
        )
        _absorb_chunk_ids(text_chunks)
        _absorb_paths(text_paths)

        if seeds:
            t_neighborhood = time.perf_counter()
            neighborhood = await asyncio.wait_for(
                _neighborhood_per_seed(
                    store,
                    seeds,
                    depth=depth,
                    limit=limit,
                    project_name=pname,
                    relationship_types=relationship_types,
                ),
                timeout=timeout,
            )
            timing_ms["neighborhood_ms"] = round((time.perf_counter() - t_neighborhood) * 1000, 2)
            _absorb_chunk_ids(neighborhood.chunk_ids)
            _absorb_paths(neighborhood.paths_summary)
    except asyncio.TimeoutError:
        logger.warning(
            "Graph expansion timed out after %.1fs for project %r (seeds=%d text_seeds=%d)",
            timeout,
            project_name,
            len(seeds),
            text_seed_count,
        )
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return _vector_only_result(
            vector_chunks,
            degraded=True,
            degrade_reason="graph_query_timeout",
            timing_ms=timing_ms,
            merge_strategy=strategy,
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
            merge_strategy=strategy,
        )

    if not seeds and text_seed_count == 0 and not all_candidate_ids:
        return _vector_only_result(vector_chunks, merge_strategy=strategy)

    selected_ids = _select_graph_chunk_ids(
        all_candidate_ids,
        existing_ids,
        vector_chunks,
        max(0, budget),
    )
    if not selected_ids:
        lines = format_graph_context_lines(all_paths)
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        sources = {
            cid: "vector"
            for cid in (_chunk_id_from_result(c) for c in vector_chunks)
            if cid
        }
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=all_paths[:20],
            graph_context_lines=lines,
            vector_chunk_count=vector_count,
            graph_added_count=0,
            seed_count=len(seeds),
            text_seed_count=text_seed_count,
            timing_ms=timing_ms,
            merge_strategy=strategy,
            chunk_sources=sources,
        )

    try:
        vector_store = await get_vector_store()
        t_hydrate = time.perf_counter()
        graph_chunks = await vector_store.get_chunks_by_ids(selected_ids)
        timing_ms["hydrate_ms"] = round((time.perf_counter() - t_hydrate) * 1000, 2)
    except Exception as exc:
        logger.warning("Failed to fetch graph-expanded chunks from Qdrant: %s", exc)
        lines = format_graph_context_lines(all_paths)
        timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)
        return GraphAugmentationResult(
            chunks=list(vector_chunks),
            paths_summary=all_paths[:20],
            graph_context_lines=lines,
            vector_chunk_count=vector_count,
            graph_added_count=0,
            seed_count=len(seeds),
            text_seed_count=text_seed_count,
            degraded=True,
            degrade_reason="qdrant_hydrate_error",
            timing_ms=timing_ms,
            merge_strategy=strategy,
        )

    # For diversity selection of extras we already ordered candidates; merge applies strategy to full lists
    merged, sources = merge_vector_and_graph_chunks(
        vector_chunks,
        graph_chunks,
        strategy=strategy,
    )
    added = [
        cid for cid in selected_ids
        if cid not in existing_ids and any(_chunk_id_from_result(c) == cid for c in graph_chunks)
    ]
    lines = format_graph_context_lines(all_paths)
    timing_ms["total_ms"] = round((time.perf_counter() - t_total) * 1000, 2)

    logger.info(
        "Graph augmentation project=%r query_len=%d seeds=%d text_seeds=%d vector=%d "
        "added=%d strategy=%s degraded=false timing_ms=%s",
        project_name,
        len(query or ""),
        len(seeds),
        text_seed_count,
        vector_count,
        len(added),
        strategy,
        timing_ms,
    )
    return GraphAugmentationResult(
        chunks=merged,
        added_chunk_ids=added,
        paths_summary=all_paths[:20],
        graph_context_lines=lines,
        vector_chunk_count=vector_count,
        graph_added_count=len(added),
        seed_count=len(seeds),
        text_seed_count=text_seed_count,
        timing_ms=timing_ms,
        merge_strategy=strategy,
        chunk_sources=sources,
    )
