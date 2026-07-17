"""Phase 5: offline graph retrieval evaluation models."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class GraphEvalCase(BaseModel):
    """One curated question with expected chunk IDs for recall measurement."""

    id: str = Field(..., min_length=1, description="Stable case identifier")
    query: str = Field(..., min_length=1)
    project_name: str = Field(..., min_length=1)
    must_include_chunk_ids: List[str] = Field(
        default_factory=list,
        description="All listed chunk IDs must appear in top-k for a hit",
    )
    top_k: int = Field(10, ge=1, le=50)
    enabled: bool = True
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class GraphEvalSet(BaseModel):
    """Collection of eval cases (loaded from JSON)."""

    name: str = "default"
    description: Optional[str] = None
    cases: List[GraphEvalCase] = Field(default_factory=list)


class GraphEvalCaseResult(BaseModel):
    """Per-case recall outcome for one retrieval mode."""

    case_id: str
    query: str
    project_name: str
    top_k: int
    mode: str  # vector_only | graph_augmented
    recall_at_k: float
    hit_at_k: bool
    retrieved_chunk_ids: List[str] = Field(default_factory=list)
    missing_chunk_ids: List[str] = Field(default_factory=list)
    graph_degraded: Optional[bool] = None
    graph_added_count: Optional[int] = None
    skipped: bool = False
    skip_reason: Optional[str] = None


class GraphEvalSummary(BaseModel):
    """Aggregate metrics across enabled cases."""

    cases_run: int = 0
    cases_skipped: int = 0
    mean_recall_vector: float = 0.0
    mean_recall_graph: float = 0.0
    hits_vector: int = 0
    hits_graph: int = 0
    graph_improved_count: int = 0
    graph_regressed_count: int = 0


class GraphEvalReport(BaseModel):
    """Full eval output: per-case results + summary."""

    eval_set_name: str
    k_values: List[int] = Field(default_factory=list)
    vector_results: List[GraphEvalCaseResult] = Field(default_factory=list)
    graph_results: List[GraphEvalCaseResult] = Field(default_factory=list)
    summary: GraphEvalSummary = Field(default_factory=GraphEvalSummary)
