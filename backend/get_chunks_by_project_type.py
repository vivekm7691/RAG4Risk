"""Fetch all chunk text from Qdrant for a project and document type.

Usage:
    # Inside Docker (recommended):
    docker compose exec backend python get_chunks_by_project_type.py \\
        --project-name "My Project" \\
        --document-type "risk register"

    # Write JSON to file:
    docker compose exec backend python get_chunks_by_project_type.py \\
        --project-name "My Project" \\
        --document-type "statement of work" \\
        --format json \\
        --output chunks.json

    # Locally (from backend/ with deps installed):
    python get_chunks_by_project_type.py --project-name "My Project" --document-type "issue log"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.models.document import DocumentType
    from app.services.vector_store import get_vector_store
except ImportError as e:
    print("=" * 60)
    print("ERROR: Cannot import required modules")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nRun inside the backend container:")
    print("  docker compose exec backend python get_chunks_by_project_type.py --help")
    sys.exit(1)

DOCUMENT_TYPE_CHOICES = [dt.value for dt in DocumentType]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get all chunk text from Qdrant for a project and document type.",
    )
    parser.add_argument(
        "--project-name",
        required=True,
        help="Project name (exact match, as stored in Qdrant payload project_name)",
    )
    parser.add_argument(
        "--document-type",
        required=True,
        choices=DOCUMENT_TYPE_CHOICES,
        help="Document type payload value",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum chunks to return (default: 10000)",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json", "both"),
        default="text",
        help="Output format: concatenated text, JSON array, or both (default: text)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Optional file path to write output (stdout if omitted)",
    )
    parser.add_argument(
        "--separator",
        default="\n\n---\n\n",
        help="Separator between chunks when --format is text or both",
    )
    return parser.parse_args()


def _sort_chunks(chunks: list) -> list:
    """Stable order: document_id, chunk_index, row_number, chunk_id."""
    def key(c: dict):
        m = c.get("metadata") or {}
        return (
            m.get("document_id") or "",
            m.get("chunk_index") if m.get("chunk_index") is not None else -1,
            m.get("row_number") if m.get("row_number") is not None else -1,
            m.get("chunk_id") or "",
        )

    return sorted(chunks, key=key)


async def main() -> int:
    args = parse_args()
    doc_type = DocumentType(args.document_type)

    print(f"Connecting to Qdrant...", file=sys.stderr)
    vector_store = await get_vector_store()
    print(
        f"Collection: {vector_store.collection_name} | "
        f"project={args.project_name!r} | document_type={doc_type.value!r}",
        file=sys.stderr,
    )

    chunks = await vector_store.get_chunks_by_project_and_document_type(
        project_name=args.project_name.strip(),
        document_type=doc_type,
        limit=args.limit,
    )
    chunks = _sort_chunks(chunks)
    print(f"Found {len(chunks)} chunk(s).", file=sys.stderr)

    if args.format in ("json", "both"):
        payload = [
            {
                "text": c.get("text", ""),
                "metadata": c.get("metadata") or {},
            }
            for c in chunks
        ]
        json_out = json.dumps(payload, indent=2, ensure_ascii=False)

    if args.format in ("text", "both"):
        texts = [(c.get("text") or "").strip() for c in chunks]
        text_out = args.separator.join(t for t in texts if t)

    lines: list[str] = []
    if args.format == "json":
        lines.append(json_out)
    elif args.format == "text":
        lines.append(text_out)
    else:
        lines.append("=== JSON ===")
        lines.append(json_out)
        lines.append("=== TEXT ===")
        lines.append(text_out)

    body = "\n".join(lines)
    if args.output:
        args.output.write_text(body, encoding="utf-8")
        print(f"Wrote {len(body)} characters to {args.output}", file=sys.stderr)
    else:
        print(body)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
