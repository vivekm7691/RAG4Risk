"""Phase 5: offline recall@k evaluation for graph-augmented retrieval."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from app.config import settings
from app.models.graph_eval import (
    GraphEvalCase,
    GraphEvalCaseResult,
    GraphEvalReport,
    GraphEvalSet,
    GraphEvalSummary,
)
from app.services.embeddings import EmbeddingService
from app.services.graph_retrieval import GraphAugmentationResult, augment_chunks_with_graph
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

DEFAULT_EVAL_SET_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "graph_eval_set.json"


def recall_at_k(
    retrieved_ids: Sequence[str],
    must_include: Set[str],
    k: int,
) -> Tuple[float, bool, List[str]]:
    """
    Compute recall@k and hit@k for must-include chunk IDs.

    Returns (recall_fraction, hit_all_present, missing_ids_in_top_k).
    """
    if not must_include:
        return 1.0, True, []

    k = max(1, k)
    top = set(retrieved_ids[:k])
    found = must_include & top
    missing = sorted(must_include - top)
    recall = len(found) / len(must_include)
    hit = len(missing) == 0
    return recall, hit, missing


def chunk_ids_from_results(chunks: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for chunk in chunks:
        cid = (chunk.get("metadata") or {}).get("chunk_id")
        if cid:
            ids.append(cid)
    return ids


def load_eval_set(path: Optional[Path] = None) -> GraphEvalSet:
    """Load eval cases from JSON file."""
    p = path or DEFAULT_EVAL_SET_PATH
    raw = json.loads(p.read_text(encoding="utf-8"))
    return GraphEvalSet.model_validate(raw)


async def retrieve_for_eval(
    query: str,
    project_name: str,
    top_k: int,
    *,
    use_graph: bool,
    depth: Optional[int] = None,
    extra_budget: Optional[int] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> Tuple[List[str], Optional[GraphAugmentationResult]]:
    """Run vector search (+ optional graph augmentation); return ordered chunk IDs."""
    emb = embedding_service or EmbeddingService()
    query_embedding = emb.generate_embedding(query).tolist()
    vector_store = await get_vector_store()
    vector_chunks = await vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        project_name=project_name,
        document_type=None,
        filters=None,
    )

    if not use_graph or not settings.GRAPH_ENABLED:
        return chunk_ids_from_results(vector_chunks), None

    graph_result = await augment_chunks_with_graph(
        vector_chunks,
        project_name=project_name,
        depth=depth,
        extra_budget=extra_budget,
    )
    return chunk_ids_from_results(graph_result.chunks), graph_result


def _case_skip_reason(case: GraphEvalCase) -> Optional[str]:
    if not case.enabled:
        return "disabled"
    if not case.must_include_chunk_ids:
        return "no must_include_chunk_ids (fill after ingest)"
    return None


def _result_for_case(
    case: GraphEvalCase,
    *,
    mode: str,
    retrieved_ids: List[str],
    graph_result: Optional[GraphAugmentationResult],
    skip_reason: Optional[str],
) -> GraphEvalCaseResult:
    if skip_reason:
        return GraphEvalCaseResult(
            case_id=case.id,
            query=case.query,
            project_name=case.project_name,
            top_k=case.top_k,
            mode=mode,
            recall_at_k=0.0,
            hit_at_k=False,
            skipped=True,
            skip_reason=skip_reason,
        )

    recall, hit, missing = recall_at_k(
        retrieved_ids,
        set(case.must_include_chunk_ids),
        case.top_k,
    )
    return GraphEvalCaseResult(
        case_id=case.id,
        query=case.query,
        project_name=case.project_name,
        top_k=case.top_k,
        mode=mode,
        recall_at_k=round(recall, 4),
        hit_at_k=hit,
        retrieved_chunk_ids=retrieved_ids[: case.top_k],
        missing_chunk_ids=missing,
        graph_degraded=graph_result.degraded if graph_result else None,
        graph_added_count=graph_result.graph_added_count if graph_result else None,
    )


def _build_summary(
    vector_results: List[GraphEvalCaseResult],
    graph_results: List[GraphEvalCaseResult],
) -> GraphEvalSummary:
    active_v = [r for r in vector_results if not r.skipped]
    active_g = [r for r in graph_results if not r.skipped]
    skipped = sum(1 for r in vector_results if r.skipped)

    def _mean_recall(rows: List[GraphEvalCaseResult]) -> float:
        if not rows:
            return 0.0
        return round(sum(r.recall_at_k for r in rows) / len(rows), 4)

    improved = 0
    regressed = 0
    by_id_g = {r.case_id: r for r in active_g}
    for vr in active_v:
        gr = by_id_g.get(vr.case_id)
        if not gr:
            continue
        if gr.recall_at_k > vr.recall_at_k:
            improved += 1
        elif gr.recall_at_k < vr.recall_at_k:
            regressed += 1

    return GraphEvalSummary(
        cases_run=len(active_v),
        cases_skipped=skipped,
        mean_recall_vector=_mean_recall(active_v),
        mean_recall_graph=_mean_recall(active_g),
        hits_vector=sum(1 for r in active_v if r.hit_at_k),
        hits_graph=sum(1 for r in active_g if r.hit_at_k),
        graph_improved_count=improved,
        graph_regressed_count=regressed,
    )


async def run_eval_set(
    eval_set: GraphEvalSet,
    *,
    compare_graph: bool = True,
    depth: Optional[int] = None,
    extra_budget: Optional[int] = None,
    embedding_service: Optional[EmbeddingService] = None,
) -> GraphEvalReport:
    """
    Run all enabled cases: vector-only vs graph-augmented (when GRAPH_ENABLED).

    Cases without must_include_chunk_ids are skipped (template placeholders).
    """
    vector_results: List[GraphEvalCaseResult] = []
    graph_results: List[GraphEvalCaseResult] = []
    k_values: Set[int] = set()

    for case in eval_set.cases:
        k_values.add(case.top_k)
        skip = _case_skip_reason(case)

        if skip:
            vector_results.append(
                _result_for_case(case, mode="vector_only", retrieved_ids=[], graph_result=None, skip_reason=skip)
            )
            graph_results.append(
                _result_for_case(case, mode="graph_augmented", retrieved_ids=[], graph_result=None, skip_reason=skip)
            )
            continue

        vec_ids, _ = await retrieve_for_eval(
            case.query,
            case.project_name,
            case.top_k,
            use_graph=False,
            embedding_service=embedding_service,
        )
        vector_results.append(
            _result_for_case(case, mode="vector_only", retrieved_ids=vec_ids, graph_result=None, skip_reason=None)
        )

        if compare_graph and settings.GRAPH_ENABLED:
            graph_ids, graph_result = await retrieve_for_eval(
                case.query,
                case.project_name,
                case.top_k,
                use_graph=True,
                depth=depth,
                extra_budget=extra_budget,
                embedding_service=embedding_service,
            )
            graph_results.append(
                _result_for_case(
                    case,
                    mode="graph_augmented",
                    retrieved_ids=graph_ids,
                    graph_result=graph_result,
                    skip_reason=None,
                )
            )
        else:
            graph_results.append(
                _result_for_case(
                    case,
                    mode="graph_augmented",
                    retrieved_ids=vec_ids,
                    graph_result=None,
                    skip_reason="graph_disabled_or_compare_off",
                )
            )

    return GraphEvalReport(
        eval_set_name=eval_set.name,
        k_values=sorted(k_values),
        vector_results=vector_results,
        graph_results=graph_results,
        summary=_build_summary(vector_results, graph_results),
    )


def format_report_text(report: GraphEvalReport) -> str:
    """Human-readable summary for CLI output."""
    lines = [
        f"Eval set: {report.eval_set_name}",
        f"Cases run: {report.summary.cases_run} (skipped: {report.summary.cases_skipped})",
        f"Mean recall@k  vector={report.summary.mean_recall_vector:.2%}  "
        f"graph={report.summary.mean_recall_graph:.2%}",
        f"Hit@k (all must-include in top-k): vector={report.summary.hits_vector}  "
        f"graph={report.summary.hits_graph}",
        f"Graph vs vector: improved={report.summary.graph_improved_count}  "
        f"regressed={report.summary.graph_regressed_count}",
        "",
        "Per case:",
    ]
    graph_by_id = {r.case_id: r for r in report.graph_results}
    for vr in report.vector_results:
        if vr.skipped:
            lines.append(f"  [{vr.case_id}] SKIPPED ({vr.skip_reason})")
            continue
        gr = graph_by_id.get(vr.case_id)
        graph_part = ""
        if gr and not gr.skipped:
            delta = gr.recall_at_k - vr.recall_at_k
            sign = "+" if delta >= 0 else ""
            graph_part = (
                f" | graph recall={gr.recall_at_k:.0%} hit={gr.hit_at_k}"
                f" added={gr.graph_added_count or 0} ({sign}{delta:.0%})"
            )
        lines.append(
            f"  [{vr.case_id}] vector recall={vr.recall_at_k:.0%} hit={vr.hit_at_k}{graph_part}"
        )
        if vr.missing_chunk_ids:
            lines.append(f"    missing (vector): {', '.join(vr.missing_chunk_ids[:5])}")
        if gr and gr.missing_chunk_ids and not gr.skipped:
            lines.append(f"    missing (graph):  {', '.join(gr.missing_chunk_ids[:5])}")
    return "\n".join(lines)
