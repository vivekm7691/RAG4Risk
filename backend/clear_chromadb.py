"""Script to clear Qdrant collection for testing

NOTE: This script needs to be updated to use async VectorStore methods.
It currently uses synchronous code which won't work with the new Qdrant implementation.
"""

import sys
from pathlib import Path

# Add parent directory to path to import app modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import VectorStore

def main():
    """Clear all documents from ChromaDB collection"""
    print("Connecting to ChromaDB...")
    try:
        store = VectorStore()
        print(f"Collection: {store.collection.name}")
        
        # Get count before clearing
        results = store.collection.get()
        count_before = len(results["ids"]) if results["ids"] else 0
        print(f"Documents in collection before clearing: {count_before}")
        
        if count_before == 0:
            print("Collection is already empty.")
            return
        
        # Clear the collection
        print("Clearing collection...")
        store.clear_collection()
        
        # Verify it's empty
        results_after = store.collection.get()
        count_after = len(results_after["ids"]) if results_after["ids"] else 0
        print(f"Documents in collection after clearing: {count_after}")
        
        if count_after == 0:
            print("[SUCCESS] Collection cleared successfully!")
        else:
            print(f"[WARNING] {count_after} documents still remain")
            
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

