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
    OLLAMA_MODEL: str = "llama3.1"
    
    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Chroma Settings
    # Note: In Docker, CHROMA_HOST is set to "chroma" (service name) via environment variables
    # For local development (outside Docker), use localhost:8001
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001  # Host port mapping (container port 8000 is mapped to host 8001)
    CHROMA_COLLECTION_NAME: str = "rag4risk_documents"
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 5
    
    # File Upload Settings
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".docx", ".xlsx"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

