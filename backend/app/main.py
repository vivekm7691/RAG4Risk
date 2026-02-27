"""FastAPI application entry point for RAG4Risk"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG4Risk API",
    description="RAG system for Word document ingestion, vector embeddings, and semantic Q&A",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "RAG4Risk API", "version": "1.0.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Register API routes
from app.api.routes import documents, query, diagnostics, projects
app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(diagnostics.router, prefix="/api/diagnostics", tags=["diagnostics"])
app.include_router(projects.router, prefix="/api/projects", tags=["projects"])


@app.on_event("startup")
async def startup_event():
    """Preload models and services at startup to avoid httpx client closure issues"""
    logger.info("Preloading embedding model at startup...")
    try:
        import asyncio
        from app.services.embeddings import EmbeddingService
        
        # Load model in a thread to avoid blocking the event loop
        # The model should already be cached from Docker build or download_model.py
        embedding_service = EmbeddingService()
        
        # Run model loading in a thread to avoid async context issues with httpx
        def load_model_sync():
            """Load model synchronously in a thread"""
            try:
                logger.info("Loading embedding model from cache in background thread...")
                # Access model property to trigger loading (should use cached files)
                # This happens synchronously in a thread, avoiding httpx client closure issues
                model = embedding_service.model
                logger.info(f"Embedding model loaded successfully: {type(model).__name__}")
            except Exception as e:
                logger.error(f"Failed to load model in background thread: {e}", exc_info=True)
                # Don't raise - let it fail on first use with better error message
                logger.warning("Model loading failed at startup, will retry on first request")
        
        # Run in thread pool to avoid blocking and httpx client conflicts
        await asyncio.to_thread(load_model_sync)
        logger.info("Embedding model preload completed")
    except Exception as e:
        logger.error(f"Failed to preload embedding model: {e}", exc_info=True)
        # Don't fail startup - model will be loaded on first use
        logger.warning("Model will be loaded on first request, which may cause httpx client closure issues")

