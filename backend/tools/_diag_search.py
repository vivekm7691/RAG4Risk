"""Diagnostic: verify Qdrant has data and search returns results."""
import asyncio
from app.services.vector_store import VectorStore
from app.services.embeddings import EmbeddingService
from app.config import settings
from qdrant_client.models import Filter, FieldCondition, MatchValue


async def main():
    vs = VectorStore()
    client = await vs._get_client()

    info = await client.get_collection(settings.QDRANT_COLLECTION_NAME)
    print(f"Collection: {settings.QDRANT_COLLECTION_NAME}")
    print(f"Points count: {info.points_count}")
    try:
        vec_cfg = info.config.params.vectors
        if hasattr(vec_cfg, "size"):
            print(f"Vector size: {vec_cfg.size}")
        else:
            print(f"Vector config: {vec_cfg}")
    except Exception as e:
        print(f"Vector config error: {e}")
    print()

    scroll_result = await client.scroll(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        limit=3,
        with_payload=True,
        with_vectors=False,
    )
    points = scroll_result[0]
    print(f"Scroll returned {len(points)} points")
    for p in points[:3]:
        pn = p.payload.get("project_name", "?")
        dt = p.payload.get("document_type", "?")
        print(f"  id={p.id}  project={pn}  doc_type={dt}")
    print()

    emb = EmbeddingService()
    vec = emb.generate_embedding("What risks exist in the solution").tolist()
    print(f"Embedding dim: {len(vec)}, first 5: {vec[:5]}")

    resp1 = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=vec,
        limit=5,
    )
    print(f"Unfiltered search: {len(resp1.points)} results")
    for p in resp1.points[:3]:
        pn = p.payload.get("project_name", "?")
        dt = p.payload.get("document_type", "?")
        print(f"  score={p.score:.4f}  project={pn}  doc_type={dt}")

    filt = Filter(must=[
        FieldCondition(key="project_name", match=MatchValue(value="Freedom Encompass Modernization"))
    ])
    resp2 = await client.query_points(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        query=vec,
        query_filter=filt,
        limit=5,
    )
    print(f"Project-filtered search: {len(resp2.points)} results")
    for p in resp2.points[:3]:
        pn = p.payload.get("project_name", "?")
        dt = p.payload.get("document_type", "?")
        print(f"  score={p.score:.4f}  project={pn}  doc_type={dt}")

    # Also test via the VectorStore.search method (same path as API)
    print()
    print("--- VectorStore.search (same path as API) ---")
    results = await vs.search(
        query_embedding=vec,
        top_k=5,
        project_name="Freedom Encompass Modernization",
        document_type=None,
        filters=None,
    )
    print(f"VectorStore.search: {len(results)} results")
    for r in results[:3]:
        meta = r.get("metadata", {})
        print(f"  dist={r.get('distance'):.4f}  project={meta.get('project_name')}  doc_type={meta.get('document_type')}")


asyncio.run(main())
