#!/usr/bin/env python3
"""One-off: create Qdrant keyword payload indexes for filter fields (migration helper).

Run from `backend/` with:

  set PYTHONPATH=.
  python tools/ensure_qdrant_payload_indexes.py

Uses QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION_NAME from environment / .env.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main() -> None:
    from app.services.vector_store import get_vector_store

    vs = await get_vector_store()
    await vs.ensure_filter_payload_indexes()
    print("ensure_filter_payload_indexes finished.")


if __name__ == "__main__":
    asyncio.run(main())
