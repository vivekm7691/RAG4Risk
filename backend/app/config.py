"""Configuration management for RAG4Risk"""

from typing import List

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    API_V1_PREFIX: str = "/api"
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:3b"
    OLLAMA_TIMEOUT: float = 3000.0  # 50 minutes — httpx timeout for RAG Ollama /api/generate
    OLLAMA_STREAMING_ENABLED: bool = True  # Enable streaming responses by default
    # Context window tokens passed to Ollama as options.num_ctx (RAG /api/generate and intent /api/chat). Use 0 to omit.
    OLLAMA_NUM_CTX: int = 16384

    # Phase 3.75: query intent LLM (Ollama /api/chat). Empty base URL falls back to OLLAMA_BASE_URL.
    INTENT_LLM_BASE_URL: str = ""
    INTENT_LLM_MODEL: str = "deepseek-r1"
    # Accept INTENT_LLM_TIMEOUT (plan name) or INTENT_LLM_TIMEOUT_SECONDS
    INTENT_LLM_TIMEOUT_SECONDS: float = Field(
        default=1200.0,
        validation_alias=AliasChoices("INTENT_LLM_TIMEOUT_SECONDS", "INTENT_LLM_TIMEOUT"),
    )
    INTENT_LLM_MAX_TOKENS: int = 4096
    INTENT_LLM_TEMPERATURE: float = 0.1
    # When False, ignore client use_query_intent (no intent LLM / weighted retrieval).
    QUERY_INTENT_ENABLED: bool = False
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    HF_HUB_DISABLE_SSL_VERIFY: bool = False  # Disable SSL verification for HuggingFace downloads (for corporate proxies)
    
    # Qdrant Settings
    # Note: In Docker, QDRANT_HOST is set to "qdrant" (service name) via environment variables
    # For local development (outside Docker), use localhost:6333
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "rag4risk_documents"
    # Higher ef improves recall for filtered ANN; omit when unset or <= 0
    QDRANT_SEARCH_HNSW_EF: int = 128
    
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

