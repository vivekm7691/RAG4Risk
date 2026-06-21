"""Re-run knowledge graph sync for an existing document (chunks loaded from Qdrant).

Use after graph ingest logic changes or when LLM extraction was skipped/failed.

Usage:
    docker compose exec backend python resync_document_graph.py --document-id <uuid>

    docker compose exec backend python resync_document_graph.py \\
        --project-name "Freedom Mobile" \\
        --document-type "statement of work" \\
        --file-name "sow.docx"
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.models.document import DocumentType
    from app.services.graph_sync_service import sync_document_to_graph
    from app.services.vector_store import get_vector_store
except ImportError as e:
    print(f"Import error: {e}", file=sys.stderr)
    print("Run inside backend container: docker compose exec backend python resync_document_graph.py --help")
    sys.exit(1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Re-sync Neo4j graph for a document from Qdrant chunks.")
    p.add_argument("--document-id", help="Document UUID (preferred)")
    p.add_argument("--project-name", help="Find document by project + file name if id unknown")
    p.add_argument("--file-name", help="File name match (with --project-name)")
    p.add_argument(
        "--document-type",
        choices=[dt.value for dt in DocumentType],
        help="Optional filter when resolving by project",
    )
    return p.parse_args()


async def _resolve_document_id(args: argparse.Namespace, vector_store) -> tuple[str, list]:
    if args.document_id:
        chunks = await vector_store.get_document_chunks(args.document_id)
        if not chunks:
            raise SystemExit(f"No chunks in Qdrant for document_id={args.document_id}")
        return args.document_id, chunks

    if not args.project_name:
        raise SystemExit("Provide --document-id or --project-name (and --file-name).")

    doc_type = DocumentType(args.document_type) if args.document_type else None
    if doc_type:
        chunks = await vector_store.get_chunks_by_project_and_document_type(
            args.project_name.strip(),
            doc_type,
        )
    else:
        chunks = await vector_store.get_chunks_by_project(args.project_name.strip())

    if not chunks:
        raise SystemExit(f"No chunks for project={args.project_name!r}")

    by_doc: dict[str, list] = {}
    for c in chunks:
        doc_id = (c.get("metadata") or {}).get("document_id")
        if doc_id:
            by_doc.setdefault(doc_id, []).append(c)

    if args.file_name:
        fn = args.file_name.strip().lower()
        for doc_id, doc_chunks in by_doc.items():
            meta = doc_chunks[0].get("metadata") or {}
            if (meta.get("file_name") or "").lower() == fn:
                return doc_id, doc_chunks
        raise SystemExit(f"No document with file_name={args.file_name!r} under project.")

    if len(by_doc) == 1:
        doc_id = next(iter(by_doc))
        return doc_id, by_doc[doc_id]

    raise SystemExit(
        f"Multiple documents ({len(by_doc)}) under project. Use --document-id or --file-name."
    )


async def main() -> int:
    args = parse_args()
    vector_store = await get_vector_store()
    document_id, chunks = await _resolve_document_id(args, vector_store)
    meta = chunks[0].get("metadata") or {}

    print(
        f"Re-syncing graph: document_id={document_id} "
        f"project={meta.get('project_name')!r} "
        f"type={meta.get('document_type')!r} "
        f"chunks={len(chunks)}",
        file=sys.stderr,
    )

    stats = await sync_document_to_graph(
        chunks,
        document_id=document_id,
        project_name=meta.get("project_name") or "",
        document_type=meta.get("document_type") or "",
        file_name=meta.get("file_name") or "",
        title=meta.get("title"),
    )
    print(f"Done: {stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
