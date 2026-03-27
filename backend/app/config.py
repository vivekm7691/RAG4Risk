"""Configuration management for RAG4Risk"""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT: float = 1200.0  # 20 minutes timeout for LLM responses
    OLLAMA_STREAMING_ENABLED: bool = True  # Enable streaming responses by default
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    HF_HUB_DISABLE_SSL_VERIFY: bool = False  # Disable SSL verification for HuggingFace downloads (for corporate proxies)
    
    # Qdrant Settings
    # Note: In Docker, QDRANT_HOST is set to "qdrant" (service name) via environment variables
    # For local development (outside Docker), use localhost:6333
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag4risk_documents"
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5

    # Phase 3.5: Past Projects Enhancement
    DEFAULT_CURRENT_PROJECT_WEIGHT: float = 0.7
    DEFAULT_PAST_PROJECTS_WEIGHT: float = 0.3
    SIMILAR_PROJECTS_COUNT: int = 3
    PROJECT_SIMILARITY_SEMANTIC_WEIGHT: float = 0.5
    PROJECT_SIMILARITY_METADATA_WEIGHT: float = 0.5
    # Per-field weights for metadata half of hybrid score (should sum to 1.0)
    PROJECT_SIMILARITY_META_WEIGHT_CSG_PRODUCTS: float = 0.20
    PROJECT_SIMILARITY_META_WEIGHT_INTEGRATION_COMPLEXITY: float = 0.15
    PROJECT_SIMILARITY_META_WEIGHT_PROJECT_SIZE: float = 0.10
    PROJECT_SIMILARITY_META_WEIGHT_PROJECT_COMPLEXITY: float = 0.15
    PROJECT_SIMILARITY_META_WEIGHT_CSG_ROLE: float = 0.15
    PROJECT_SIMILARITY_META_WEIGHT_CLIENT_TYPE: float = 0.15
    PROJECT_SIMILARITY_META_WEIGHT_DATE_RANGE: float = 0.10
    # Cache TTL for find_similar_projects (seconds); cleared on metadata writes
    PROJECT_SIMILARITY_CACHE_TTL_SECONDS: float = 300.0
    PROJECT_METADATA_DB_PATH: str = "data/project_metadata.db"
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: List[str] = [".docx", ".xlsx"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields (e.g., old CHROMA_* variables)


settings = Settings()

