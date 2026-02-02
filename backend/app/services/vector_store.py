"""Vector store service for Chroma database integration"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from app.config import settings
from app.models.document import DocumentType as DocType


class VectorStore:
    """Service for managing vector storage in Chroma"""
    
    def __init__(self):
        """Initialize the vector store service"""
        self.client: Optional[chromadb.ClientAPI] = None
        self.collection: Optional[chromadb.Collection] = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Chroma client and collection"""
        try:
            # Create Chroma client
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            # Use default embedding function (will be overridden by our embeddings)
            embedding_fn = embedding_functions.DefaultEmbeddingFunction()
            
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                embedding_function=embedding_fn,
                metadata={"description": "RAG4Risk document collection"}
            )
            
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Chroma database: {str(e)}")
    
    def add_documents(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> List[str]:
        """
        Add document chunks to the vector store
        
        Checks for existing chunks by ID and content hash to prevent duplicates.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata'
            embeddings: List of embedding vectors (one per chunk)
            
        Returns:
            List of chunk IDs that were actually added (new chunks only)
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        if not chunks:
            return []
        
        # Extract chunk IDs and validate
        ids = []
        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id")
            if not chunk_id:
                raise ValueError("Chunk metadata must contain 'chunk_id'")
            ids.append(chunk_id)
        
        # Generate content hashes for all chunks
        content_hashes = {}
        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id")
            content_hash = self._generate_content_hash(chunk)
            content_hashes[chunk_id] = content_hash
        
        # Check for existing chunk IDs
        existing_ids = set()
        if ids:
            try:
                existing_results = self.collection.get(ids=ids)
                if existing_results["ids"]:
                    existing_ids = set(existing_results["ids"])
            except Exception:
                # If get() fails, assume no existing IDs (safe to proceed)
                existing_ids = set()
        
        # Check for existing content hashes (query by content_hash metadata)
        existing_content_hashes = set()
        try:
            # Query for chunks with matching content hashes
            unique_content_hashes = set(content_hashes.values())
            for content_hash in unique_content_hashes:
                results = self.collection.get(
                    where={"content_hash": content_hash}
                )
                if results["ids"]:
                    existing_content_hashes.add(content_hash)
        except Exception:
            # If query fails, proceed without content hash check
            pass
        
        # Filter out chunks that already exist (by ID or content hash)
        new_chunks = []
        new_embeddings = []
        new_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk["metadata"].get("chunk_id")
            content_hash = content_hashes[chunk_id]
            
            # Skip if ID exists OR content hash exists
            if chunk_id not in existing_ids and content_hash not in existing_content_hashes:
                # Add content hash to metadata before storing
                chunk["metadata"]["content_hash"] = content_hash
                new_chunks.append(chunk)
                new_embeddings.append(embedding)
                new_ids.append(chunk_id)
        
        # Only add new chunks
        if not new_chunks:
            return []
        
        # Prepare data for Chroma
        texts = []
        metadatas = []
        
        for chunk in new_chunks:
            texts.append(chunk["text"])
            # Prepare metadata (Chroma requires string values)
            metadata = self._prepare_metadata(chunk["metadata"])
            metadatas.append(metadata)
        
        # Add to collection
        try:
            self.collection.add(
                ids=new_ids,
                embeddings=new_embeddings,
                documents=texts,
                metadatas=metadatas
            )
        except Exception as e:
            raise RuntimeError(f"Failed to add documents to vector store: {str(e)}")
        
        return new_ids
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        project_name: Optional[str] = None,
        document_type: Optional[DocType] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents in the vector store
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return (defaults to settings.TOP_K)
            project_name: Filter by project name (optional)
            document_type: Filter by document type (optional)
            filters: Additional metadata filters for Excel-based queries (optional)
                    Supported filters:
                    - severity: Filter by severity level
                    - status: Filter by status
                    - category: Filter by category
                    - date_range: Dict with 'start_date' and 'end_date' (ISO format strings)
                    - owner: Filter by owner/assignee
        
        Returns:
            List of result dictionaries, each containing:
                - text: Chunk text
                - metadata: Chunk metadata
                - distance: Similarity distance (lower is more similar)
        """
        top_k = top_k or settings.TOP_K
        
        # Build where clause for filtering
        where_clause = self._build_where_clause(
            project_name=project_name,
            document_type=document_type,
            filters=filters
        )
        
        try:
            # Perform search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where_clause if where_clause else None
            )
            
            # Format results
            formatted_results = []
            
            if results["ids"] and len(results["ids"][0]) > 0:
                for idx in range(len(results["ids"][0])):
                    result = {
                        "text": results["documents"][0][idx],
                        "metadata": results["metadatas"][0][idx],
                        "distance": results["distances"][0][idx] if "distances" in results else None
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search vector store: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deletion was successful
        """
        try:
            # Find all chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results["ids"]:
                # Delete chunks
                self.collection.delete(ids=results["ids"])
                return True
            
            return False
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from vector store: {str(e)}")
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        try:
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            chunks = []
            if results["ids"]:
                for idx in range(len(results["ids"])):
                    chunk = {
                        "text": results["documents"][idx],
                        "metadata": results["metadatas"][idx]
                    }
                    chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Failed to get document chunks: {str(e)}")
    
    def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if clearing was successful
        """
        try:
            # Get all IDs in the collection
            results = self.collection.get()
            
            if results["ids"]:
                # Delete all documents
                self.collection.delete(ids=results["ids"])
                return True
            
            return True  # Collection is already empty
            
        except Exception as e:
            raise RuntimeError(f"Failed to clear collection: {str(e)}")
    
    def _generate_content_hash(self, chunk: Dict[str, Any]) -> str:
        """
        Generate a hash from chunk content (text + metadata excluding document_id)
        This allows detecting duplicates even when document_id changes
        
        Args:
            chunk: Chunk dictionary with 'text' and 'metadata'
            
        Returns:
            SHA256 hash as hexadecimal string
        """
        # Create a copy of metadata without document_id, chunk_id, upload_date for hashing
        # These fields can change between runs but don't affect content uniqueness
        hash_metadata = {
            k: v for k, v in chunk["metadata"].items() 
            if k not in ["document_id", "chunk_id", "upload_date"]
        }
        
        # Create hash from text + metadata
        content_str = json.dumps({
            "text": chunk["text"],
            "metadata": hash_metadata
        }, sort_keys=True)
        
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """
        Prepare metadata for Chroma (all values must be strings)
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Metadata dictionary with all string values
        """
        prepared = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            
            # Convert to string
            if isinstance(value, (datetime,)):
                prepared[key] = value.isoformat()
            elif isinstance(value, (list, dict)):
                # Convert complex types to JSON string
                import json
                prepared[key] = json.dumps(value)
            elif isinstance(value, DocType):
                prepared[key] = value.value
            else:
                prepared[key] = str(value)
        
        return prepared
    
    def _build_where_clause(
        self,
        project_name: Optional[str] = None,
        document_type: Optional[DocType] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Build Chroma where clause for filtering
        
        Args:
            project_name: Filter by project name
            document_type: Filter by document type
            filters: Additional filters for Excel-based queries
        
        Returns:
            Chroma where clause dictionary or None
        """
        where_clauses = []
        
        # Project name filter
        if project_name:
            where_clauses.append({"project_name": project_name})
        
        # Document type filter
        if document_type:
            where_clauses.append({"document_type": document_type.value})
        
        # Additional filters for Excel-based queries
        if filters:
            # Severity filter
            if "severity" in filters:
                where_clauses.append({"severity": filters["severity"]})
            
            # Status filter
            if "status" in filters:
                where_clauses.append({"status": filters["status"]})
            
            # Category filter
            if "category" in filters:
                where_clauses.append({"category": filters["category"]})
            
            # Owner filter
            if "owner" in filters:
                where_clauses.append({"owner": filters["owner"]})
            
            # Date range filter (requires Chroma's $gte and $lte operators
            if "date_range" in filters:
                date_range = filters["date_range"]
                if "start_date" in date_range and "end_date" in date_range:
                    # Note: Chroma date filtering may require custom handling
                    # For now, we'll store dates as strings and do string comparison
                    where_clauses.append({
                        "$and": [
                            {"date": {"$gte": date_range["start_date"]}},
                            {"date": {"$lte": date_range["end_date"]}}
                        ]
                    })
        
        # Combine clauses with AND
        if len(where_clauses) == 1:
            return where_clauses[0]
        elif len(where_clauses) > 1:
            return {"$and": where_clauses}
        else:
            return None


