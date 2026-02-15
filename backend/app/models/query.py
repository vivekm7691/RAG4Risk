"""Query models for RAG4Risk"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceCitation(BaseModel):
    """Source citation for query response"""
    document_id: str
    document_name: str
    chunk_id: str
    project_name: str
    document_type: str
    relevance_score: Optional[float] = None
    # Excel-specific metadata
    row_number: Optional[int] = None
    sheet_name: Optional[str] = None


class QueryFilters(BaseModel):
    """Metadata filters for Excel-based queries"""
    severity: Optional[str] = Field(None, description="Filter by severity level")
    status: Optional[str] = Field(None, description="Filter by status")
    category: Optional[str] = Field(None, description="Filter by category")
    date_range: Optional[Dict[str, str]] = Field(
        None,
        description="Filter by date range with 'start_date' and 'end_date' (ISO format)"
    )
    owner: Optional[str] = Field(None, description="Filter by owner/assignee")


class QueryRequest(BaseModel):
    """Query request model"""
    query: str = Field(..., min_length=1, max_length=1000, description="User query text")
    project_name: Optional[str] = Field(None, max_length=100, description="Filter by project name")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    filters: Optional[QueryFilters] = Field(
        None,
        description="Metadata filters for Excel-based queries (severity, status, category, date_range, owner)"
    )
    model: Optional[str] = Field(
        None,
        max_length=100,
        description="Optional Ollama model name to use for this query (e.g., 'llama3.2:3b', 'mistral:7b'). If not specified, uses default from configuration."
    )


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[SourceCitation]
    query: str
    project_name: Optional[str] = None


# Streaming response models (for documentation and type hints)
class StreamingChunk(BaseModel):
    """Streaming response chunk model"""
    type: str = "chunk"
    text: str


class StreamingSources(BaseModel):
    """Streaming sources message model"""
    type: str = "sources"
    sources: List[SourceCitation]
    query: str
    project_name: Optional[str] = None


class StreamingDone(BaseModel):
    """Streaming completion message model"""
    type: str = "done"
    total_time: float
    timings: Optional[Dict[str, Optional[float]]] = None


class StreamingError(BaseModel):
    """Streaming error message model"""
    type: str = "error"
    message: str
    total_time: Optional[float] = None

