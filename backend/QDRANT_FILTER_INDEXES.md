# Qdrant filter payload indexes

## Why

Filtered vector search (`query_points` with `query_filter`) can return **no hits** while `scroll` with the same filter still finds points, if payload fields used in filters lack **keyword indexes** or data was indexed before indexes existed. The backend now creates keyword indexes for fields used in [`vector_store._build_qdrant_filter`](app/services/vector_store.py) (e.g. `project_name`, `document_type`).

## When indexes are created

1. **Application startup** — [`main.py`](app/main.py) calls `ensure_filter_payload_indexes()` if the collection already exists.
2. **First or subsequent upload** — [`_ensure_collection_exists`](app/services/vector_store.py) runs index creation after ensuring the collection exists.
3. **Manual migration** — from the `backend` directory:

   ```bash
   set PYTHONPATH=.
   python tools/ensure_qdrant_payload_indexes.py
   ```

   (On Linux/macOS: `export PYTHONPATH=.`)

## Search tuning

`QDRANT_SEARCH_HNSW_EF` (default `128`) is passed to `query_points` as `SearchParams(hnsw_ef=...)`. Higher values can improve recall under filters at some CPU cost. Set to `0` in `.env` to omit custom search params.

## Optional HNSW re-optimization

If you added indexes **after** ingesting a large corpus and recall is still poor, Qdrant recommends rebuilding the graph so HNSW aligns with payload indexing. See the [Qdrant collection parameters](https://qdrant.tech/documentation/concepts/collections/#update-collection-parameters) docs (e.g. temporarily adjusting HNSW `m`, waiting for optimization, then restoring). Alternatively reindex into a new collection with indexes created **before** bulk insert.

## Client request filters

`QueryRequest.filters` fields are **AND**-combined with `project_name` (and per-type filters when using query intent). Non-matching metadata yields empty search results; see [`QueryFilters`](app/models/query.py) docstring.
