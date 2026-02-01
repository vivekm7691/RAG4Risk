"""Query models for RAG4Risk"""

from typing import List, Optional
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Source citation for query response"""
    document_id: str
    document_name: str
    chunk_id: str
    project_name: str
    document_type: str
    relevance_score: Optional[float] = None


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    project_name: Optional[str] = Field(None, max_length=100, description="Filter by project name")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[SourceCitation]
    query: str
    project_name: Optional[str] = None

