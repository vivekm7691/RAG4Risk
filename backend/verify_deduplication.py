"""Script to verify deduplication in ChromaDB"""

import sys
from pathlib import Path
from collections import Counter, defaultdict

# Add parent directory to path to import app modules
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.vector_store import VectorStore


def verify_deduplication():
    """Verify that deduplication is working correctly in ChromaDB"""
    print("=" * 70)
    print("  ChromaDB Deduplication Verification")
    print("=" * 70)
    print()
    
    try:
        # Connect to ChromaDB
        print("Connecting to ChromaDB...")
        store = VectorStore()
        print(f"Collection: {store.collection.name}")
        print()
        
        # Get all chunks
        print("Retrieving all chunks from collection...")
        results = store.collection.get()
        
        if not results["ids"]:
            print("Collection is empty. No chunks to verify.")
            return
        
        total_chunks = len(results["ids"])
        print(f"Total chunks in collection: {total_chunks}")
        print()
        
        # Analyze chunks
        document_types = Counter()
        content_hashes = Counter()
        content_hash_to_chunks = defaultdict(list)
        document_ids = set()
        
        for idx in range(len(results["ids"])):
            chunk_id = results["ids"][idx]
            metadata = results["metadatas"][idx]
            
            # Count by document_type
            doc_type = metadata.get("document_type", "unknown")
            document_types[doc_type] += 1
            
            # Track content_hashes
            content_hash = metadata.get("content_hash")
            if content_hash:
                content_hashes[content_hash] += 1
                content_hash_to_chunks[content_hash].append({
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", "unknown"),
                    "document_type": doc_type
                })
            
            # Track document_ids
            doc_id = metadata.get("document_id")
            if doc_id:
                document_ids.add(doc_id)
        
        # Print statistics
        print("=" * 70)
        print("  Statistics")
        print("=" * 70)
        print()
        
        print(f"Total chunks: {total_chunks}")
        print(f"Unique document IDs: {len(document_ids)}")
        print(f"Unique content hashes: {len(content_hashes)}")
        print()
        
        # Chunks by document_type
        print("Chunks by document type:")
        for doc_type, count in document_types.most_common():
            print(f"  - {doc_type}: {count}")
        print()
        
        # Check for duplicate content_hashes
        duplicate_hashes = {h: count for h, count in content_hashes.items() if count > 1}
        
        if duplicate_hashes:
            print("=" * 70)
            print("  WARNING: Duplicate Content Hashes Found!")
            print("=" * 70)
            print()
            print(f"Found {len(duplicate_hashes)} content hashes that appear multiple times:")
            print()
            
            for content_hash, count in duplicate_hashes.items():
                print(f"Content hash: {content_hash[:16]}... (appears {count} times)")
                chunks_with_hash = content_hash_to_chunks[content_hash]
                for chunk_info in chunks_with_hash:
                    print(f"  - Chunk ID: {chunk_info['chunk_id']}")
                    print(f"    Document ID: {chunk_info['document_id']}")
                    print(f"    Document Type: {chunk_info['document_type']}")
                print()
        else:
            print("=" * 70)
            print("  SUCCESS: No Duplicate Content Hashes!")
            print("=" * 70)
            print()
            print("All content hashes are unique. Deduplication is working correctly.")
            print()
        
        # Summary
        print("=" * 70)
        print("  Summary")
        print("=" * 70)
        print()
        
        if duplicate_hashes:
            total_duplicates = sum(count - 1 for count in duplicate_hashes.values())
            print(f"Status: FAILED - Found {len(duplicate_hashes)} duplicate content hashes")
            print(f"Total duplicate chunks: {total_duplicates}")
            print(f"Expected unique chunks: {total_chunks - total_duplicates}")
            print(f"Actual unique chunks: {len(content_hashes)}")
        else:
            print("Status: PASSED - No duplicates found")
            print(f"Total chunks: {total_chunks}")
            print(f"Unique content hashes: {len(content_hashes)}")
            print("All chunks have unique content hashes.")
        
        print()
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    verify_deduplication()

