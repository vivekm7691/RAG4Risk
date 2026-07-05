"""Embedding generation service using sentence-transformers"""

import os
import logging
import threading
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)

# Lock to prevent concurrent model loading
_model_load_lock = threading.Lock()


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
    
    def _load_model(self) -> SentenceTransformer:
        """
        Load the embedding model synchronously (called at startup or on first access)
        Uses thread lock to prevent concurrent loading and httpx client conflicts
        
        Returns:
            SentenceTransformer model instance
        """
        # Double-check pattern with lock to prevent concurrent loading
        if self._model is not None:
            return self._model
        
        with _model_load_lock:
            # Check again after acquiring lock
            if self._model is not None:
                return self._model
                
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Configure SSL verification for HuggingFace downloads
            disable_ssl = (
                os.getenv("HF_HUB_DISABLE_SSL_VERIFY", "").lower() in ("true", "1", "yes") or
                settings.HF_HUB_DISABLE_SSL_VERIFY
            )
            
            if disable_ssl:
                os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
                os.environ["CURL_CA_BUNDLE"] = ""
                os.environ["REQUESTS_CA_BUNDLE"] = ""
                import ssl
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                ssl._create_default_https_context = ssl._create_unverified_context
            
            # Clear any stale httpx clients from huggingface_hub before loading
            try:
                import huggingface_hub
                # Clear cached clients to force new ones
                if hasattr(huggingface_hub, '_CACHED_CLIENTS'):
                    huggingface_hub._CACHED_CLIENTS.clear()
                # Also try to clear httpx client cache
                try:
                    import huggingface_hub.utils._http
                    if hasattr(huggingface_hub.utils._http, '_http_client'):
                        huggingface_hub.utils._http._http_client = None
                except:
                    pass
            except Exception as e:
                logger.debug(f"Could not clear huggingface_hub clients: {e}")
            
            # Set cache directory
            cache_dir = os.getenv("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
            logger.info(f"Using HuggingFace cache directory: {cache_dir}")
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    logger.info(f"Initializing SentenceTransformer (attempt {attempt + 1}/{max_retries})")
                    # Load model - should use cached files if available
                    # Note: Even with cached files, huggingface_hub may make HTTP requests
                    # to check for updates/adapter configs. The model should be fully cached from Docker build.
                    self._model = SentenceTransformer(
                        self.model_name,
                        cache_folder=cache_dir
                    )
                    logger.info(f"Model loaded successfully: {self.model_name}")
                    return self._model
                except RuntimeError as e:
                    error_str = str(e).lower()
                    if "client has been closed" in error_str and attempt < max_retries - 1:
                        logger.warning(f"httpx client closed error (attempt {attempt + 1}/{max_retries}), retrying...")
                        import time
                        time.sleep(2)
                        # Clear clients again before retry
                        try:
                            import huggingface_hub
                            if hasattr(huggingface_hub, '_CACHED_CLIENTS'):
                                huggingface_hub._CACHED_CLIENTS.clear()
                        except:
                            pass
                        continue
                    else:
                        logger.error(f"Failed to load model after {max_retries} attempts: {e}", exc_info=True)
                        raise
                except Exception as e:
                    logger.error(f"Unexpected error loading model: {type(e).__name__}: {str(e)}", exc_info=True)
                    # If SSL verification fails and not already disabled, try with disabled verification
                    if ("certificate" in str(e).lower() or "ssl" in str(e).lower()) and \
                       not disable_ssl and attempt < max_retries - 1:
                        logger.warning("SSL error detected, retrying with SSL verification disabled")
                        os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
                        os.environ["CURL_CA_BUNDLE"] = ""
                        os.environ["REQUESTS_CA_BUNDLE"] = ""
                        import ssl
                        import urllib3
                        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                        ssl._create_default_https_context = ssl._create_unverified_context
                        disable_ssl = True
                        continue
                    raise
    
    @property
    def model(self) -> SentenceTransformer:
        """
        Get or load the embedding model (lazy loading)
        
        Returns:
            SentenceTransformer model instance
        """
        if self._model is None:
            self._load_model()
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








