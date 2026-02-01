"""FastAPI application entry point for RAG4Risk"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings

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


# TODO: Register API routes when implemented
# from app.api.routes import documents, query
# app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
# app.include_router(query.router, prefix="/api/query", tags=["query"])

