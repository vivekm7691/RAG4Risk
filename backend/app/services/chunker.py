"""Chunking service for documents - semantic chunking for Word, row-based for Excel"""

import uuid
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.document import DocumentType as DocType
from app.config import settings


class Chunker:
    """Service for chunking documents into smaller pieces"""
    
    def __init__(self):
        """Initialize the chunker"""
        self._text_splitter = None
    
    def chunk_document(
        self,
        text: str,
        metadata: Dict[str, Any],
        document_type: DocType
    ) -> List[Dict[str, Any]]:
        """
        Chunk a document based on its type
        
        Args:
            text: Text content to chunk
            metadata: Document metadata
            document_type: Type of document
            
        Returns:
            List of chunk dictionaries, each containing:
                - text: Chunk text content
                - metadata: Chunk metadata (including chunk_id)
        """
        if document_type in [DocType.RISK_REGISTER, DocType.ISSUE_LOG]:
            # Excel files are already chunked row-by-row by the parser
            # This method is called for Word documents
            raise ValueError("Excel files should be chunked row-by-row in the parser, not here")
        
        # Word documents: use semantic chunking
        return self._chunk_word_document(text, metadata)
    
    def chunk_excel_rows(
        self,
        rows: List[Dict[str, Any]],
        document_metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Process Excel rows - each row is already a chunk
        
        Args:
            rows: List of row objects from Excel parser
            document_metadata: Document-level metadata
            
        Returns:
            List of chunk dictionaries (one per row)
        """
        chunks = []
        
        for idx, row in enumerate(rows):
            # Generate a UUID for each chunk (Qdrant requires UUID or integer, not string with suffix)
            chunk_id = str(uuid.uuid4())
            
            # Combine row metadata with document metadata
            chunk_metadata = {
                **document_metadata,
                **row.get("metadata", {}),
                "chunk_id": chunk_id,
                "chunk_index": idx
            }
            
            chunks.append({
                "text": row.get("text", ""),
                "metadata": chunk_metadata
            })
        
        return chunks
    
    def _chunk_word_document(
        self,
        text: str,
        metadata: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Chunk a Word document using recursive character text splitting
        
        Args:
            text: Document text content
            metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        # Get text splitter
        text_splitter = self._get_text_splitter()
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        
        # Convert to our format with metadata
        result_chunks = []
        for idx, chunk_text in enumerate(chunks):
            # Generate a UUID for each chunk (Qdrant requires UUID or integer, not string with suffix)
            chunk_id = str(uuid.uuid4())
            
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_index": idx,
                "chunk_size": len(chunk_text)
            }
            
            result_chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })
        
        return result_chunks
    
    def _get_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Get or create text splitter instance
        
        Returns:
            RecursiveCharacterTextSplitter instance
        """
        if self._text_splitter is None:
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.CHUNK_SIZE,
                chunk_overlap=settings.CHUNK_OVERLAP,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
        
        return self._text_splitter
    

