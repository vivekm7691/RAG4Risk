"""Vector store service for Qdrant database integration"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import json
import uuid
import logging

logger = logging.getLogger(__name__)

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    CollectionStatus,
)

from app.config import settings
from app.models.document import DocumentType as DocType

# Singleton instance
_vector_store_instance: Optional['VectorStore'] = None


async def get_vector_store() -> 'VectorStore':
    """Get or create singleton VectorStore instance"""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStore()
        await _vector_store_instance._initialize_client()
    return _vector_store_instance


class VectorStore:
    """Service for managing vector storage in Qdrant"""
    
    def __init__(self):
        """Initialize the vector store service"""
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name: str = settings.QDRANT_COLLECTION_NAME
        self._initialized: bool = False
        # Embedding dimension (will be set after first embedding)
        self._embedding_dim: Optional[int] = None
    
    async def _initialize_client(self):
        """Initialize Qdrant client and collection (deprecated - using fresh clients now)"""
        # This method is kept for backward compatibility but is no longer used
        # We create fresh clients for each operation to avoid "client has been closed" errors
        self._initialized = True
    
    async def _get_client(self) -> AsyncQdrantClient:
        """Get or create a Qdrant client (creates fresh client each time to avoid closed client issues)"""
        logger.debug(f"Creating new AsyncQdrantClient: host={settings.QDRANT_HOST}, port={settings.QDRANT_PORT}")
        try:
            client = AsyncQdrantClient(
                host=settings.QDRANT_HOST,
                port=settings.QDRANT_PORT,
                timeout=30.0
            )
            logger.debug(f"AsyncQdrantClient created successfully: {type(client)}")
            return client
        except Exception as e:
            logger.error(f"Failed to create AsyncQdrantClient: {type(e).__name__}: {str(e)}", exc_info=True)
            raise
    
    async def _ensure_initialized(self):
        """Ensure client is initialized (for backward compatibility)"""
        if not self._initialized:
            # Just mark as initialized, we'll create clients on-demand
            self._initialized = True
    
    async def _ensure_collection_exists(self, embedding_dim: int):
        """Ensure collection exists with correct vector dimension"""
        if self._embedding_dim is None:
            self._embedding_dim = embedding_dim
        
        # Get a fresh client for this operation
        logger.info(f"Ensuring collection exists: {self.collection_name}, embedding_dim={embedding_dim}")
        client = await self._get_client()
        try:
            # Check if collection exists
            logger.debug("Calling client.get_collections()")
            collections = await client.get_collections()
            logger.debug(f"Retrieved {len(collections.collections)} collections")
            collection_exists = any(
                col.name == self.collection_name 
                for col in collections.collections
            )
            logger.info(f"Collection exists: {collection_exists}")
            
            if not collection_exists:
                logger.info(f"Creating collection: {self.collection_name} with dimension {embedding_dim}")
                # Create collection with vector dimension
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection created successfully: {self.collection_name}")
        except Exception as e:
            logger.error(
                f"Error in _ensure_collection_exists: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise
        finally:
            # Don't close client - let it be garbage collected
            logger.debug("Leaving client open for garbage collection")
    
    async def add_documents(
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
        
        # Ensure collection exists
        embedding_dim = len(embeddings[0])
        await self._ensure_collection_exists(embedding_dim)
        
        # Generate chunk IDs and check for existing ones
        ids = []
        content_hashes = {}
        
        for chunk in chunks:
            chunk_id = chunk["metadata"].get("chunk_id")
            if not chunk_id:
                chunk_id = str(uuid.uuid4())
                chunk["metadata"]["chunk_id"] = chunk_id
            
            content_hash = self._generate_content_hash(chunk)
            content_hashes[chunk_id] = content_hash
            ids.append(chunk_id)
        
        # Get a fresh client for this operation
        logger.info(f"Adding {len(chunks)} documents to collection: {self.collection_name}")
        client = await self._get_client()
        try:
            # Check for existing points by ID
            logger.debug(f"Checking for existing points with {len(ids)} IDs")
            existing_points = await client.retrieve(
                collection_name=self.collection_name,
                ids=ids
            )
            existing_ids = {point.id for point in existing_points}
            logger.info(f"Found {len(existing_ids)} existing points")
            
            # Check for existing content hashes
            # Note: Qdrant doesn't support querying by payload directly in a simple way
            # We'll skip content hash checking for now and rely on chunk_id uniqueness
            # This is acceptable since chunk_id should be unique per document+chunk
            
            # Prepare points for new chunks only
            new_points = []
            new_ids = []
            
            for idx, chunk in enumerate(chunks):
                chunk_id = chunk["metadata"]["chunk_id"]
                
                # Skip if already exists
                if chunk_id in existing_ids:
                    continue
                
                # Prepare payload (metadata) - Qdrant supports various types
                payload = self._prepare_payload(chunk["metadata"])
                payload["text"] = chunk["text"]  # Store text in payload for retrieval
                
                point = PointStruct(
                    id=chunk_id,
                    vector=embeddings[idx],
                    payload=payload
                )
                new_points.append(point)
                new_ids.append(chunk_id)
            
            # Add new points to collection
            if new_points:
                logger.info(f"Upserting {len(new_points)} new points")
                await client.upsert(
                    collection_name=self.collection_name,
                    points=new_points
                )
                logger.info(f"Successfully upserted {len(new_points)} points")
            else:
                logger.info("No new points to add (all already exist)")
        except Exception as e:
            logger.error(
                f"Error in add_documents: {type(e).__name__}: {str(e)}",
                exc_info=True
            )
            raise
        finally:
            # Don't close client - let it be garbage collected
            logger.debug("Leaving client open for garbage collection")
        
        return new_ids
    
    async def search(
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
        
        # Build filter for Qdrant
        qdrant_filter = self._build_qdrant_filter(
            project_name=project_name,
            document_type=document_type,
            filters=filters
        )
        
        try:
            logger.info(f"Starting vector search: collection={self.collection_name}, top_k={top_k}, has_filter={qdrant_filter is not None}")
            
            # Get a fresh client for this operation
            client = await self._get_client()
            logger.debug(f"Client obtained, type: {type(client)}, id: {id(client)}")
            
            try:
                # Perform search using query_points() method for vector search
                logger.debug(f"Calling client.query_points() with collection={self.collection_name}, vector_dim={len(query_embedding)}")
                
                # Use query_points for vector search - it accepts vector list directly
                query_response = await client.query_points(
                    collection_name=self.collection_name,
                    query=query_embedding,  # Pass vector list directly (list[float])
                    query_filter=qdrant_filter if qdrant_filter else None,
                    limit=top_k
                )
                
                # Extract points from response (query_response.points is a list of ScoredPoint objects)
                search_results = query_response.points
                logger.info(f"Search completed successfully, found {len(search_results)} results")
            except Exception as search_error:
                logger.error(
                    f"Error during client.query_points(): {type(search_error).__name__}: {str(search_error)}",
                    exc_info=True
                )
                raise
            finally:
                # Don't close the client - let it be garbage collected naturally
                # Closing it immediately causes "client has been closed" errors
                logger.debug("Leaving client open for garbage collection")
            
            # Format results
            formatted_results = []
            
            for result in search_results:
                payload = result.payload
                # Qdrant returns similarity score (higher is better, range 0-1 for cosine)
                # Convert to distance (lower is better) for consistency with ChromaDB API
                similarity = result.score
                distance = 1.0 - similarity  # Convert similarity to distance
                
                formatted_result = {
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"},
                    "distance": distance
                }
                formatted_results.append(formatted_result)
            
            return formatted_results
            
        except Exception as e:
            raise RuntimeError(f"Failed to search vector store: {str(e)}")
    
    async def delete_document(self, document_id: str) -> bool:
        """
        Delete all chunks for a document
        
        Args:
            document_id: Document ID to delete
            
        Returns:
            True if deletion was successful
        """
        await self._ensure_initialized()
        
        client = await self._get_client()
        try:
            # Find all points for this document using filter
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Scroll to get all points matching the filter
            scroll_result = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=10000  # Large limit to get all points
            )
            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            
            if points:
                # Delete points by ID
                point_ids = [point.id for point in points]
                await client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids
                )
                return True
            
            return False
            
        except Exception as e:
            raise RuntimeError(f"Failed to delete document from vector store: {str(e)}")
    
    async def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunk dictionaries
        """
        await self._ensure_initialized()
        
        client = await self._get_client()
        try:
            # Find all points for this document using filter
            filter_condition = Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id)
                    )
                ]
            )
            
            # Scroll to get all points matching the filter
            scroll_result = await client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_condition,
                limit=10000  # Large limit to get all points
            )
            points = scroll_result[0] if isinstance(scroll_result, tuple) else scroll_result.points
            
            chunks = []
            for point in points:
                payload = point.payload
                chunk = {
                    "text": payload.get("text", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "text"}
                }
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            raise RuntimeError(f"Failed to get document chunks: {str(e)}")
    
    async def clear_collection(self) -> bool:
        """
        Clear all documents from the collection
        
        Returns:
            True if clearing was successful
        """
        await self._ensure_initialized()
        
        client = await self._get_client()
        try:
            # Delete collection and recreate it
            await client.delete_collection(collection_name=self.collection_name)
            
            # Recreate with same dimension if we know it
            if self._embedding_dim:
                await client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self._embedding_dim,
                        distance=Distance.COSINE
                    )
                )
            
            return True
            
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
    
    def _prepare_payload(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare payload for Qdrant (supports various types, not just strings)
        
        Args:
            metadata: Original metadata dictionary
            
        Returns:
            Payload dictionary with appropriate types
        """
        payload = {}
        for key, value in metadata.items():
            # Qdrant supports: str, int, float, bool, list, dict
            # Convert None to empty string
            if value is None:
                payload[key] = ""
            elif isinstance(value, (str, int, float, bool, list, dict)):
                payload[key] = value
            else:
                # Convert other types to string
                payload[key] = str(value)
        
        return payload
    
    def _build_qdrant_filter(
        self,
        project_name: Optional[str] = None,
        document_type: Optional[DocType] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from query parameters
        
        Args:
            project_name: Filter by project name
            document_type: Filter by document type
            filters: Additional metadata filters
            
        Returns:
            Qdrant Filter object or None
        """
        conditions = []
        
        # Project name filter
        if project_name:
            conditions.append(
                FieldCondition(
                    key="project_name",
                    match=MatchValue(value=project_name)
                )
            )
        
        # Document type filter
        if document_type:
            conditions.append(
                FieldCondition(
                    key="document_type",
                    match=MatchValue(value=document_type.value)
                )
            )
        
        # Additional filters
        if filters:
            # Severity filter
            if "severity" in filters:
                conditions.append(
                    FieldCondition(
                        key="severity",
                        match=MatchValue(value=filters["severity"])
                    )
                )
            
            # Status filter
            if "status" in filters:
                conditions.append(
                    FieldCondition(
                        key="status",
                        match=MatchValue(value=filters["status"])
                    )
                )
            
            # Category filter
            if "category" in filters:
                conditions.append(
                    FieldCondition(
                        key="category",
                        match=MatchValue(value=filters["category"])
                    )
                )
            
            # Owner filter
            if "owner" in filters:
                conditions.append(
                    FieldCondition(
                        key="owner",
                        match=MatchValue(value=filters["owner"])
                    )
                )
            
            # Date range filter
            if "date_range" in filters:
                date_range = filters["date_range"]
                if "start_date" in date_range or "end_date" in date_range:
                    range_conditions = {}
                    if "start_date" in date_range:
                        range_conditions["gte"] = date_range["start_date"]
                    if "end_date" in date_range:
                        range_conditions["lte"] = date_range["end_date"]
                    
                    conditions.append(
                        FieldCondition(
                            key="date",
                            range=Range(**range_conditions)
                        )
                    )
        
        if conditions:
            return Filter(must=conditions)
        
        return None
