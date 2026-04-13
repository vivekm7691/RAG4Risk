"""Quick diagnostic: check what chunks exist in Qdrant for a project."""
import sys
import asyncio

sys.path.insert(0, ".")

PROJECT = "Freedom Encompass Modernization"


async def main():
    from app.services.vector_store import get_vector_store
    from app.services.embeddings import EmbeddingService

    vs = await get_vector_store()

    # 1. Check distinct document types for project
    types = await vs.distinct_document_types_for_project(PROJECT)
    print(f"Document types for '{PROJECT}': {types}")

    # 2. Count chunks for this project
    chunks = await vs.get_chunks_by_project(PROJECT, limit=10)
    print(f"First 10 chunks for '{PROJECT}': {len(chunks)} found")
    for i, c in enumerate(chunks[:3]):
        meta = c.get("metadata", {})
        print(f"  [{i}] project_name={meta.get('project_name')!r}  "
              f"document_type={meta.get('document_type')!r}  "
              f"chunk_id={meta.get('chunk_id', '')[:30]!r}")

    # 3. Try an unfiltered vector search with the project name
    emb = EmbeddingService()
    qe = emb.generate_embedding("risks in the solution").tolist()
    results = await vs.search(
        query_embedding=qe, top_k=5, project_name=PROJECT, document_type=None
    )
    print(f"\nUnfiltered search (project_name='{PROJECT}'): {len(results)} hits")
    for i, r in enumerate(results[:3]):
        meta = r.get("metadata", {})
        print(f"  [{i}] dist={r.get('distance'):.4f}  "
              f"doc_type={meta.get('document_type')!r}  "
              f"chunk_id={meta.get('chunk_id', '')[:30]!r}")

    # 4. Try search WITHOUT project_name filter
    results2 = await vs.search(
        query_embedding=qe, top_k=5, project_name=None, document_type=None
    )
    print(f"\nUnfiltered search (no project filter): {len(results2)} hits")
    for i, r in enumerate(results2[:3]):
        meta = r.get("metadata", {})
        print(f"  [{i}] project={meta.get('project_name')!r}  "
              f"doc_type={meta.get('document_type')!r}  "
              f"dist={r.get('distance'):.4f}")

    # 5. Per-type filtered search (exactly what weighted path does)
    from app.models.document import DocumentType
    for dt in [DocumentType.STATEMENT_OF_WORK, DocumentType.SOLUTION_DESCRIPTION]:
        r3 = await vs.search(
            query_embedding=qe, top_k=3, project_name=PROJECT, document_type=dt
        )
        print(f"\nFiltered search (project + document_type={dt.value!r}): {len(r3)} hits")
        for i, r in enumerate(r3[:2]):
            meta = r.get("metadata", {})
            print(f"  [{i}] dist={r.get('distance'):.4f}  "
                  f"doc_type={meta.get('document_type')!r}")

    # 6. Simulate _retrieve_weighted_for_project
    from app.services.query_intent_service import allocate_top_k_by_weights
    weights = {"statement of work": 0.143, "solution description document": 0.857}
    alloc = allocate_top_k_by_weights(weights, 10)
    print(f"\nAllocation for top_k=10: {alloc}")
    total_found = 0
    for doc_type_str, ki in alloc.items():
        if ki <= 0:
            continue
        try:
            dt = DocumentType(doc_type_str)
        except ValueError as e:
            print(f"  ValueError for {doc_type_str!r}: {e}")
            continue
        r4 = await vs.search(
            query_embedding=qe, top_k=ki, project_name=PROJECT, document_type=dt,
            filters=None,
        )
        print(f"  {doc_type_str!r} (k={ki}): {len(r4)} hits")
        total_found += len(r4)
    print(f"  Total from weighted search: {total_found}")


asyncio.run(main())
