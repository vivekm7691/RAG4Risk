"""Manual testing script for Phase 2: Document Ingestion Pipeline

This script tests all Phase 2 services:
1. Document Parser (Word documents)
2. Excel Parser (risk registers, issue logs)
3. Chunker (Word and Excel)
4. Embedding Service
5. Vector Store (Chroma integration)

Usage:
    python test_phase2.py

Prerequisites:
    - Install dependencies: pip install -r requirements.txt
    - Start Chroma: docker-compose up chroma -d
    - Place test files in Samples/ directory:
      - test_document.docx (Word document)
      - test_risk_register.xlsx (Excel risk register)
      - test_issue_log.xlsx (Excel issue log)
"""

import sys
import os
from pathlib import Path
from typing import Optional
import uuid
from datetime import datetime

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

# Get project root (one level up from backend/)
PROJECT_ROOT = Path(__file__).parent.parent

from app.services.document_parser import DocumentParser
from app.services.excel_parser import ExcelParser
from app.services.chunker import Chunker
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore
from app.models.document import DocumentType as DocType


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(label: str, value: any, max_length: int = 200):
    """Print a formatted result"""
    if isinstance(value, str) and len(value) > max_length:
        value = value[:max_length] + "..."
    print(f"  {label}: {value}")


def test_document_parser(file_path: Optional[str] = None):
    """Test Word document parsing"""
    print_section("Test 1: Word Document Parser")
    
    if not file_path:
        file_path = str(PROJECT_ROOT / "Samples" / "test_document.docx")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        print("  💡 Create a test Word document or update the file path")
        return None
    
    try:
        parser = DocumentParser()
        result = parser.parse(file_path, DocType.STATEMENT_OF_WORK)
        
        print_result("File", file_path)
        print_result("Text length", f"{len(result['text'])} characters")
        print_result("Metadata", result['metadata'])
        print_result("First 200 chars", result['text'][:200])
        
        if result['metadata'].get('table_count', 0) > 0:
            print_result("Tables found", result['metadata']['table_count'])
        
        print("  ✅ Document parser test passed!")
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None


def test_excel_parser(file_path: Optional[str] = None, doc_type: DocType = DocType.RISK_REGISTER):
    """Test Excel file parsing"""
    print_section(f"Test 2: Excel Parser ({doc_type.value})")
    
    if not file_path:
        if doc_type == DocType.RISK_REGISTER:
            file_path = str(PROJECT_ROOT / "Samples" / "test_risk_register.xlsx")
        else:
            file_path = str(PROJECT_ROOT / "Samples" / "test_issue_log.xlsx")
    
    if not os.path.exists(file_path):
        print(f"  ⚠️  File not found: {file_path}")
        print("  💡 Create a test Excel file or update the file path")
        return None
    
    try:
        parser = ExcelParser()
        result = parser.parse(file_path, doc_type)
        
        print_result("File", file_path)
        print_result("Total rows", result['metadata']['total_rows'])
        print_result("Sheets", result['metadata']['sheet_names'])
        print_result("Row count by sheet", result['metadata'].get('row_count_by_sheet', {}))
        
        if result['rows']:
            first_row = result['rows'][0]
            print_result("First row text", first_row['text'][:200])
            print_result("First row metadata", first_row['metadata'])
        
        print("  ✅ Excel parser test passed!")
        return result
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None


def test_chunker_word(doc_result: dict):
    """Test chunking for Word documents"""
    print_section("Test 3: Chunker (Word Documents)")
    
    if not doc_result:
        print("  ⚠️  Skipping: No document result from previous test")
        return None
    
    try:
        chunker = Chunker()
        
        # Add document metadata
        doc_metadata = {
            **doc_result['metadata'],
            "document_id": str(uuid.uuid4()),
            "project_name": "Test Project",
            "upload_date": datetime.now().isoformat()
        }
        
        chunks = chunker.chunk_document(
            doc_result['text'],
            doc_metadata,
            DocType.STATEMENT_OF_WORK
        )
        
        print_result("Number of chunks", len(chunks))
        if chunks:
            print_result("First chunk size", f"{chunks[0]['metadata'].get('chunk_size', 0)} characters")
            print_result("First chunk text", chunks[0]['text'][:200])
            print_result("First chunk metadata keys", list(chunks[0]['metadata'].keys()))
        
        print("  ✅ Word chunker test passed!")
        return chunks
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None


def test_chunker_excel(excel_result: dict):
    """Test chunking for Excel rows"""
    print_section("Test 4: Chunker (Excel Rows)")
    
    if not excel_result:
        print("  ⚠️  Skipping: No Excel result from previous test")
        return None
    
    try:
        chunker = Chunker()
        
        doc_metadata = {
            "document_id": str(uuid.uuid4()),
            "project_name": "Test Project",
            "upload_date": datetime.now().isoformat(),
            "document_type": excel_result['metadata']['document_type']
        }
        
        chunks = chunker.chunk_excel_rows(
            excel_result['rows'],
            doc_metadata
        )
        
        print_result("Number of chunks", len(chunks))
        if chunks:
            print_result("First chunk", chunks[0]['text'][:200])
            print_result("First chunk metadata", {
                k: v for k, v in chunks[0]['metadata'].items() 
                if k in ['chunk_id', 'row_number', 'sheet_name', 'severity', 'status']
            })
        
        print("  ✅ Excel chunker test passed!")
        return chunks
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return None


def test_embeddings():
    """Test embedding generation"""
    print_section("Test 5: Embedding Service")
    
    try:
        service = EmbeddingService()
        
        # Test single embedding
        test_text = "This is a test sentence for embedding generation."
        embedding = service.generate_embedding(test_text)
        
        print_result("Embedding dimension", len(embedding))
        print_result("Embedding shape (single)", embedding.shape)
        print_result("First 5 values", embedding[:5].tolist())
        
        # Test batch embeddings
        test_texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence."
        ]
        embeddings = service.generate_embeddings(test_texts)
        
        print_result("Batch embeddings shape", embeddings.shape)
        print_result("Number of texts", len(test_texts))
        
        # Test embedding dimension
        dim = service.get_embedding_dimension()
        print_result("Model embedding dimension", dim)
        
        print("  ✅ Embedding service test passed!")
        return service
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        print("  💡 Make sure sentence-transformers is installed and model can be downloaded")
        return None


def test_vector_store_connection():
    """Test Chroma vector store connection"""
    print_section("Test 6: Vector Store Connection")
    
    try:
        store = VectorStore()
        print("  ✅ Connected to Chroma successfully!")
        return store
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        print("  💡 Make sure Chroma is running: docker-compose up chroma -d")
        return None


def test_vector_store_add_and_search(store: VectorStore, chunks: list, embedding_service: EmbeddingService, doc_type_label: str = ""):
    """Test adding documents to vector store and searching
    
    Args:
        store: VectorStore instance
        chunks: List of chunks to add
        embedding_service: EmbeddingService instance
        doc_type_label: Optional label for document type (e.g., "Word" or "Excel") for reporting
    """
    if not store or not chunks or not embedding_service:
        print("  ⚠️  Skipping: Missing prerequisites")
        return
    
    try:
        # Generate embeddings for all chunks (not just first 5)
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        if doc_type_label:
            print(f"  Processing {doc_type_label} chunks...")
        print(f"  Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_service.generate_embeddings(chunk_texts).tolist()
        
        # Add to vector store
        print("  Adding chunks to vector store...")
        chunks_to_add = len(chunks)
        chunk_ids = store.add_documents(chunks, embeddings)
        chunks_added = len(chunk_ids)
        chunks_skipped = chunks_to_add - chunks_added
        print_result("Chunks attempted", chunks_to_add)
        print_result("Chunks added (new)", chunks_added)
        print_result("Chunks skipped (duplicates)", chunks_skipped)
        
        if chunks_skipped > 0:
            print(f"  [INFO] {chunks_skipped} chunks were skipped as duplicates (same content hash)")
        
        # Test search (only for first batch to avoid duplicate output)
        if doc_type_label == "Word" or not doc_type_label:
            query_text = "what is the statement of work about"
            print(f"\n  Searching for: '{query_text}'")
            query_embedding = embedding_service.generate_embedding(query_text).tolist()
            
            results = store.search(
                query_embedding,
                top_k=3,
                project_name="Test Project"
            )
            
            print_result("Search results found", len(results))
            for i, result in enumerate(results, 1):
                print(f"\n  Result {i}:")
                print(f"    Distance: {result.get('distance', 'N/A')}")
                print(f"    Text: {result['text'][:150]}...")
                print(f"    Metadata keys: {list(result.get('metadata', {}).keys())}")
            
            # Test search with filters (for Excel)
            if chunks and 'severity' in chunks[0].get('metadata', {}):
                print("\n  Testing search with severity filter...")
                filtered_results = store.search(
                    query_embedding,
                    top_k=3,
                    project_name="Test Project",
                    filters={"severity": chunks[0]['metadata'].get('severity')}
                )
                print_result("Filtered results", len(filtered_results))
        
        if doc_type_label:
            print(f"  ✅ {doc_type_label} chunks added successfully!")
        else:
            print("  ✅ Vector store add & search test passed!")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def test_end_to_end():
    """Test complete end-to-end pipeline"""
    print_section("Test 8: End-to-End Pipeline")
    
    try:
        # Create a simple test document text
        test_text = """
        This is a test document for the RAG4Risk system.
        It contains multiple paragraphs to test chunking.
        
        The system should be able to parse this document,
        chunk it into smaller pieces, generate embeddings,
        and store them in the vector database.
        
        Then we can search for relevant information using
        semantic search capabilities.
        """
        
        # 1. Parse (simulated - using text directly)
        doc_metadata = {
            "title": "Test Document",
            "author": "Test Author",
            "document_type": DocType.STATEMENT_OF_WORK.value,
            "document_id": str(uuid.uuid4()),
            "project_name": "E2E Test Project",
            "upload_date": datetime.now().isoformat()
        }
        
        # 2. Chunk
        chunker = Chunker()
        chunks = chunker.chunk_document(test_text, doc_metadata, DocType.STATEMENT_OF_WORK)
        print_result("Chunks created", len(chunks))
        
        # 3. Generate embeddings
        embedding_service = EmbeddingService()
        chunk_texts = [chunk['text'] for chunk in chunks]
        embeddings = embedding_service.generate_embeddings(chunk_texts).tolist()
        print_result("Embeddings generated", len(embeddings))
        
        # 4. Store in vector database
        vector_store = VectorStore()
        chunks_to_store = len(chunks)
        chunk_ids = vector_store.add_documents(chunks, embeddings)
        chunks_stored = len(chunk_ids)
        chunks_skipped = chunks_to_store - chunks_stored
        print_result("Chunks attempted", chunks_to_store)
        print_result("Chunks stored (new)", chunks_stored)
        if chunks_skipped > 0:
            print_result("Chunks skipped (duplicates)", chunks_skipped)
        
        # 5. Search
        query = "What is the system about?"
        query_embedding = embedding_service.generate_embedding(query).tolist()
        results = vector_store.search(
            query_embedding,
            top_k=2,
            project_name="E2E Test Project"
        )
        print_result("Search results", len(results))
        
        print("  ✅ End-to-end pipeline test passed!")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main test runner"""
    print("\n" + "=" * 70)
    print("  Phase 2: Document Ingestion Pipeline - Manual Testing")
    print("=" * 70)
    
    # Test results storage
    doc_result = None
    excel_result = None
    word_chunks = None
    excel_chunks = None
    embedding_service = None
    vector_store = None
    
    # Run tests
    try:
        # Test 1: Document Parser
        doc_result = test_document_parser()
        
        # Test 2: Excel Parser
        excel_result = test_excel_parser()
        
        # Test 3: Chunker (Word)
        if doc_result:
            word_chunks = test_chunker_word(doc_result)
        
        # Test 4: Chunker (Excel)
        if excel_result:
            excel_chunks = test_chunker_excel(excel_result)
        
        # Test 5: Embeddings
        embedding_service = test_embeddings()
        
        # Test 6: Vector Store Connection
        vector_store = test_vector_store_connection()
        
        # Test 7: Vector Store Add & Search
        if vector_store and embedding_service:
            print_section("Test 7: Vector Store - Add Documents & Search")
            
            # Add Word chunks if available
            if word_chunks:
                test_vector_store_add_and_search(vector_store, word_chunks, embedding_service, doc_type_label="Word")
            
            # Add Excel chunks if available
            if excel_chunks:
                test_vector_store_add_and_search(vector_store, excel_chunks, embedding_service, doc_type_label="Excel")
            
            if not word_chunks and not excel_chunks:
                print("  ⚠️  Skipping: No chunks available to add")
        
        # Test 8: End-to-End
        test_end_to_end()
        
    except KeyboardInterrupt:
        print("\n\n  ⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n  ❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print_section("Test Summary")
    print("  All Phase 2 component tests completed!")
    print("\n  Next steps:")
    print("    - Review test results above")
    print("    - Fix any errors encountered")
    print("    - Proceed to Phase 3: Backend API Development")
    print()


if __name__ == "__main__":
    main()


