"""Run Phase 5 graph retrieval recall@k evaluation against live Qdrant (+ optional Neo4j).

Usage:
    docker compose exec backend python run_graph_eval.py

    docker compose exec backend python run_graph_eval.py --eval-set eval/graph_eval_set.json --json-out report.json

    docker compose exec backend python run_graph_eval.py --depth 2 --extra-budget 8

Requires GRAPH_ENABLED=true and ingested documents matching eval case project names.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.services.graph_eval_service import (  # noqa: E402
    DEFAULT_EVAL_SET_PATH,
    format_report_text,
    load_eval_set,
    run_eval_set,
)


async def _main_async(args: argparse.Namespace) -> int:
    eval_path = Path(args.eval_set) if args.eval_set else DEFAULT_EVAL_SET_PATH
    if not eval_path.is_file():
        print(f"Eval set not found: {eval_path}", file=sys.stderr)
        return 1

    eval_set = load_eval_set(eval_path)
    report = await run_eval_set(
        eval_set,
        compare_graph=not args.vector_only,
        depth=args.depth,
        extra_budget=args.extra_budget,
    )

    text = format_report_text(report)
    print(text)

    if args.json_out:
        out = Path(args.json_out)
        out.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {out}")

    if report.summary.cases_run == 0:
        print(
            "\nNo cases ran — fill must_include_chunk_ids in the eval set for your ingested documents.",
            file=sys.stderr,
        )
        return 2

    if args.vector_only:
        return 0

    if not settings.GRAPH_ENABLED:
        print("\nGRAPH_ENABLED=false — graph column mirrors vector-only.", file=sys.stderr)
        return 0

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 graph retrieval recall@k eval")
    parser.add_argument(
        "--eval-set",
        default=None,
        help=f"Path to eval JSON (default: {DEFAULT_EVAL_SET_PATH.name})",
    )
    parser.add_argument(
        "--vector-only",
        action="store_true",
        help="Skip graph augmentation comparison",
    )
    parser.add_argument("--depth", type=int, default=None, help="Override GRAPH_NEIGHBORHOOD_DEFAULT_DEPTH")
    parser.add_argument("--extra-budget", type=int, default=None, help="Override GRAPH_RETRIEVAL_EXTRA_BUDGET")
    parser.add_argument("--json-out", default=None, help="Write full report JSON to this path")
    args = parser.parse_args()
    raise SystemExit(asyncio.run(_main_async(args)))


if __name__ == "__main__":
    main()
