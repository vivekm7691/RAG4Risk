"""Diagnostic: test _retrieve_weighted_for_project directly."""
import asyncio
import logging
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

from app.services.query_service import QueryService
from app.services.vector_store import get_vector_store
from app.services.embeddings import EmbeddingService


async def main():
    emb = EmbeddingService()
    vec = emb.generate_embedding("What risks exist in the solution").tolist()
    print(f"Embedding dim: {len(vec)}")

    vs = await get_vector_store()
    qs = QueryService(embedding_service=emb)

    doc_weights = {
        "statement of work": 0.25,
        "solution description document": 0.75,
    }

    print("\n--- Test 1: _retrieve_weighted_for_project ---")
    results = await qs._retrieve_weighted_for_project(
        vector_store=vs,
        query_embedding=vec,
        project_name="Freedom Encompass Modernization",
        n_slots=10,
        document_weights=doc_weights,
        exclude_chunk_ids=set(),
        filters=None,
        is_past_project=False,
        similar_project=None,
    )
    print(f"Weighted retrieval: {len(results)} chunks")
    for r in results[:5]:
        meta = r.get("metadata", {})
        print(f"  dist={r.get('distance'):.4f}  doc_type={meta.get('document_type')}  chunk_id={meta.get('chunk_id', '?')[:20]}")

    print("\n--- Test 2: retrieve_with_past_projects (weighted, no past) ---")
    result2 = await qs.retrieve_with_past_projects(
        query="What risks exist in the solution",
        project_name="Freedom Encompass Modernization",
        top_k=10,
        current_project_weight=1.0,
        past_projects_weight=0.0,
        document_weights=doc_weights,
        priority_order=["solution description document", "statement of work"],
    )
    print(f"retrieve_with_past_projects: {len(result2['chunks'])} chunks")
    for c in result2["chunks"][:5]:
        meta = c.get("metadata", {})
        print(f"  dist={c.get('distance'):.4f}  doc_type={meta.get('document_type')}  chunk_id={meta.get('chunk_id', '?')[:20]}")

    print("\n--- Test 3: plain search (no weights) ---")
    result3 = await qs.retrieve_with_past_projects(
        query="What risks exist in the solution",
        project_name="Freedom Encompass Modernization",
        top_k=5,
        current_project_weight=1.0,
        past_projects_weight=0.0,
        document_weights=None,
    )
    print(f"Plain retrieval: {len(result3['chunks'])} chunks")
    for c in result3["chunks"][:5]:
        meta = c.get("metadata", {})
        print(f"  dist={c.get('distance'):.4f}  doc_type={meta.get('document_type')}  chunk_id={meta.get('chunk_id', '?')[:20]}")


asyncio.run(main())
