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
    is_past_project: bool = False
    # Excel-specific metadata
    row_number: Optional[int] = None
    sheet_name: Optional[str] = None


class ForceProjectSelection(BaseModel):
    """Force-select a past project by customer and project name at query time."""
    customer: str = Field(..., min_length=1, max_length=200)
    project_name: str = Field(..., min_length=1, max_length=200)


class ContextWeighting(BaseModel):
    """Weights for current vs past project context (Phase 3.5)."""
    current_project_weight: float = Field(0.7, ge=0.0, le=1.0)
    past_projects_weight: float = Field(0.3, ge=0.0, le=1.0)


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
    # Phase 3.5: Past projects
    include_past_projects: bool = Field(False, description="Include context from similar past projects")
    context_weighting: Optional[ContextWeighting] = Field(
        None,
        description="Weights for current vs past project context (default 70% current, 30% past)"
    )
    exclude_chunk_ids: Optional[List[str]] = Field(
        None,
        description="Chunk IDs to exclude from context (user dropped from preview)"
    )
    force_project: Optional[ForceProjectSelection] = Field(
        None,
        description="Force-select this past project (customer + project_name) at query time"
    )


class SimilarProjectPreview(BaseModel):
    """Similar project entry in preview or final query response."""
    project_name: str
    customer: Optional[str] = None
    similarity_score: float
    metadata: Optional[Dict[str, Any]] = None


class QueryResponse(BaseModel):
    """Query response model"""
    answer: str
    sources: List[SourceCitation]
    query: str
    project_name: Optional[str] = None
    similar_projects: Optional[List[SimilarProjectPreview]] = Field(
        None,
        description="Similar past projects used for context (scores + metadata when available)",
    )


# Phase 3.5: Retrieval preview (no LLM call)
class PreviewChunk(BaseModel):
    """Candidate chunk for retrieval preview."""
    chunk_id: str
    text: str
    project_name: str
    customer: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    is_past_project: bool = False


class RetrievalPreviewResponse(BaseModel):
    """Response from retrieve-preview endpoint (no LLM)."""
    similar_projects: List[SimilarProjectPreview]
    chunks: List[PreviewChunk]
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
    similar_projects: Optional[List[SimilarProjectPreview]] = None


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

