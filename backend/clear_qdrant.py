"""Script to clear Qdrant collection for testing

This script clears all documents from the Qdrant vector store collection.

Usage:
    # Run inside Docker container (recommended):
    docker-compose exec backend python clear_qdrant.py
    
    # Or run locally if dependencies are installed:
    python backend/clear_qdrant.py
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path to import app modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Check if running in Docker or locally
try:
    from app.services.vector_store import get_vector_store
except ImportError as e:
    print("=" * 60)
    print("ERROR: Cannot import required modules")
    print("=" * 60)
    print(f"Error: {e}")
    print("\nThis script must be run inside the Docker container where dependencies are installed.")
    print("\nTo run this script:")
    print("  docker-compose exec backend python clear_qdrant.py")
    print("\nOr if running locally, ensure all dependencies are installed:")
    print("  pip install -r backend/requirements.txt")
    sys.exit(1)


async def main():
    """Clear all documents from Qdrant collection"""
    print("=" * 60)
    print("Clear Qdrant Vector Store Collection")
    print("=" * 60)
    print("Connecting to Qdrant...")
    try:
        vector_store = await get_vector_store()
        print(f"Collection: {vector_store.collection_name}")
        
        # Get approximate count before clearing
        try:
            client = await vector_store._get_client()
            scroll_result = await client.scroll(
                collection_name=vector_store.collection_name,
                limit=10000  # Get all points to count
            )
            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            count_before = len(points) if points else 0
            print(f"Points in collection before clearing: {count_before}")
            
            if count_before == 0:
                print("Collection is already empty.")
                return
        except Exception as e:
            print(f"Note: Could not check collection status: {e}")
            print("Proceeding with clear operation...")
        
        # Clear the collection
        print("\nClearing collection (deleting and recreating)...")
        success = await vector_store.clear_collection()
        
        if success:
            print("[SUCCESS] Collection cleared successfully!")
            print("The collection has been deleted and recreated (empty).")
            print("\nYou can now upload documents again.")
        else:
            print("[WARNING] Clear operation may not have completed successfully")
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

