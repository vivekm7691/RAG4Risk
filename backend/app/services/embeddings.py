"""Embedding generation service using sentence-transformers"""

from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


class EmbeddingService:
    """Service for generating embeddings from text"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embedding service
        
        Args:
            model_name: Name of the sentence-transformer model to use.
                       Defaults to settings.EMBEDDING_MODEL
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy loading)
        
        Returns:
            SentenceTransformer model instance
        """
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing (default: 32)
            
        Returns:
            NumPy array of embeddings (normalized for cosine similarity)
            Shape: (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Text string to embed
            
        Returns:
            NumPy array of embedding (normalized for cosine similarity)
            Shape: (embedding_dim,)
        """
        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True
        )
        
        return embedding[0]  # Return first (and only) embedding
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by the model
        
        Returns:
            Embedding dimension
        """
        # Generate a dummy embedding to get dimension
        dummy_embedding = self.generate_embedding("test")
        return len(dummy_embedding)


