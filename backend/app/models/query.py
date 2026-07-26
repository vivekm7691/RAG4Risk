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
    """Metadata filters for Excel-based queries.

    When set on QueryRequest, each non-null field is AND-combined with project_name
    (and document_type when using intent-weighted retrieval). Sending filters that
    do not match stored chunk metadata returns no vector hits even if the project exists.
    """
    severity: Optional[str] = Field(None, description="Filter by severity level")
    status: Optional[str] = Field(None, description="Filter by status")
    category: Optional[str] = Field(None, description="Filter by category")
    date_range: Optional[Dict[str, str]] = Field(
        None,
        description="Filter by date range with 'start_date' and 'end_date' (ISO format)"
    )
    owner: Optional[str] = Field(None, description="Filter by owner/assignee")


def metadata_filters_for_vector_search(filters: QueryFilters) -> Optional[Dict[str, Any]]:
    """
    Dict for Qdrant filter building. Drops empty strings and common OpenAPI/Swagger example
    values so literal OpenAPI example "string" placeholders are not AND-ed with project/doc_type
    (which would yield zero hits for chunks without Excel metadata).
    """
    raw = filters.model_dump(exclude_none=True)
    if not raw:
        return None
    out: Dict[str, Any] = {}
    skip_str = frozenset(("", "string"))

    for key, val in raw.items():
        if key == "date_range" and isinstance(val, dict):
            dr: Dict[str, str] = {}
            for dk, dv in val.items():
                if dv is None or not isinstance(dv, str):
                    continue
                s = dv.strip()
                if not s or s.lower() in skip_str:
                    continue
                dr[dk] = s
            if "start_date" in dr or "end_date" in dr:
                out["date_range"] = dr
            continue
        if isinstance(val, str):
            s = val.strip()
            if not s or s.lower() in skip_str:
                continue
            out[key] = s
        else:
            out[key] = val
    return out if out else None


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
    use_query_intent: bool = Field(
        False,
        description="When true and server QUERY_INTENT_ENABLED, run intent LLM and weighted per-type retrieval (requires project_name)",
    )
    use_graph_augmentation: bool = Field(
        False,
        description="When true and server GRAPH_ENABLED, expand vector hits via Neo4j neighborhood (requires project_name)",
    )


class GraphExpansionInfo(BaseModel):
    """Phase 3–5.1: graph-augmented retrieval metadata."""

    enabled: bool = False
    added_chunk_ids: List[str] = Field(default_factory=list)
    paths_summary: List[str] = Field(default_factory=list)
    vector_chunk_count: int = 0
    graph_added_count: int = 0
    seed_count: int = 0
    text_seed_count: int = 0
    degraded: bool = False
    degrade_reason: Optional[str] = None
    timing_ms: Optional[Dict[str, float]] = None
    merge_strategy: Optional[str] = None
    chunk_sources: Optional[Dict[str, str]] = Field(
        None,
        description="chunk_id -> vector | graph | both",
    )


class QueryTimings(BaseModel):
    """Phase 4: per-stage latency (seconds) for observability."""

    retrieval: Optional[float] = None
    graph: Optional[float] = None
    format: Optional[float] = None
    llm: Optional[float] = None
    total: Optional[float] = None


class PastProjectIntentSlots(BaseModel):
    """Per similar project: slot budget and document-type allocation (Phase 3.75)."""

    project_name: str
    budget: int = Field(..., ge=0, description="Retrieval slots assigned to this past project")
    slots: Dict[str, int] = Field(default_factory=dict, description="Per document_type top_k split for this budget")


class QueryIntentInfo(BaseModel):
    """Intent LLM output and resolved per-type top_k allocations for preview and responses."""

    intent_summary: str = ""
    document_weights: Dict[str, float] = Field(default_factory=dict)
    priority_order: Optional[List[str]] = None
    used_fallback: bool = False
    n_current_slots: int = Field(0, description="Rounded current-project slice of top_k")
    n_past_slots: int = Field(0, description="Rounded past-project slice of top_k")
    slots_current_project: Dict[str, int] = Field(default_factory=dict)
    slots_past_by_project: List[PastProjectIntentSlots] = Field(default_factory=list)


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
    query_intent: Optional[QueryIntentInfo] = Field(
        None,
        description="Phase 3.75: intent summary, weights, and per-type slot counts when use_query_intent was applied",
    )
    graph_expansion: Optional[GraphExpansionInfo] = Field(
        None,
        description="Phase 3: graph-augmented chunks and relationship paths when use_graph_augmentation was applied",
    )
    timings: Optional[QueryTimings] = Field(
        None,
        description="Phase 4: per-stage latency in seconds",
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
    query_intent: Optional[QueryIntentInfo] = None
    graph_expansion: Optional[GraphExpansionInfo] = None


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
    query_intent: Optional[QueryIntentInfo] = None
    graph_expansion: Optional[GraphExpansionInfo] = None


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

