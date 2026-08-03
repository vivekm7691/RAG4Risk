"""Knowledge graph ontology for RAG4Risk (Phase 0 + enterprise SOW extensions).

Canonical node/edge types, stable ID rules, and validation for ingest and relation extraction.
Aligns with Qdrant payload keys (document_id, chunk_id, project_name) and Excel row metadata.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

# Minimum confidence to persist LLM-extracted semantic nodes/edges
DEFAULT_MIN_CONFIDENCE = 0.75

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# document_type payload value for statement of work (matches DocumentType enum)
SOW_DOCUMENT_TYPE = "statement of work"


class NodeType(str, Enum):
    """Graph node labels (Neo4j :Label or RDF class)."""

    PROJECT = "Project"
    CUSTOMER = "Customer"
    DOCUMENT = "Document"
    CHUNK = "Chunk"
    RISK = "Risk"
    ISSUE = "Issue"
    REQUIREMENT = "Requirement"
    CONTROL = "Control"
    ASSUMPTION = "Assumption"
    PRODUCT = "Product"
    # Enterprise / delivery ontology (extracted from SOW and related docs)
    STATEMENT_OF_WORK = "StatementOfWork"  # 1:1 with SOW Document on ingest
    DELIVERABLE = "Deliverable"
    MILESTONE = "Milestone"
    ACTIVITY = "Activity"
    SERVICE = "Service"
    CHANGE_REQUEST = "ChangeRequest"
    CHANGE_CONTROL = "ChangeControl"
    ROLE = "Role"
    ORGANIZATION = "Organization"
    # Phase 5.2: project-scoped hub bridging SOW ↔ solution wording
    SYSTEM_COMPONENT = "SystemComponent"
    # Deferred
    PERSON_OR_ROLE = "PersonOrRole"


class EdgeType(str, Enum):
    """Directed relationship types."""

    # Structural (ingest)
    HAS_DOCUMENT = "HAS_DOCUMENT"
    CONTAINS_CHUNK = "CONTAINS_CHUNK"
    FROM_ROW = "FROM_ROW"
    REPRESENTS_SOW = "REPRESENTS_SOW"  # Document (SOW file) -> StatementOfWork node
    # Extraction / semantic (Word + LLM)
    EXTRACTED_FROM = "EXTRACTED_FROM"
    DESCRIBES = "DESCRIBES"
    MITIGATES = "MITIGATES"
    REFERENCES = "REFERENCES"
    RELATES_TO = "RELATES_TO"
    USES_PRODUCT = "USES_PRODUCT"
    IMPLIES_ASSUMPTION = "IMPLIES_ASSUMPTION"
    # Enterprise table (SOW / delivery)
    IS_PART_OF = "IS_PART_OF"
    IS_RESPONSIBLE_FOR = "IS_RESPONSIBLE_FOR"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    HAS_DELIVERABLE = "HAS_DELIVERABLE"
    REQUIRES_ACCEPTANCE_OF = "REQUIRES_ACCEPTANCE_OF"
    DEFINES = "DEFINES"
    PRODUCES = "PRODUCES"
    HAS_MILESTONE = "HAS_MILESTONE"
    HAS_RISK = "HAS_RISK"
    INCLUDES = "INCLUDES"
    HAS_DEPENDENCY_ON = "HAS_DEPENDENCY_ON"
    REFERENCES_REQUIREMENT = "REFERENCES_REQUIREMENT"
    RECORDED_IN = "RECORDED_IN"
    COVERS = "COVERS"
    MANAGED_VIA = "MANAGED_VIA"
    OUTLINES_OBJECTIVE_FOR = "OUTLINES_OBJECTIVE_FOR"
    # Phase 5.2: cross-artifact (SOW ↔ solution)
    DESCRIBES_CURRENT_STATE_OF = "DESCRIBES_CURRENT_STATE_OF"
    IMPLEMENTS = "IMPLEMENTS"
    ADDRESSES = "ADDRESSES"
    TRACES_TO = "TRACES_TO"
    GAPS = "GAPS"
    SAME_AS = "SAME_AS"
    # Deferred
    SIMILAR_TO = "SIMILAR_TO"
    DUPLICATES = "DUPLICATES"


# Node types that may stand in for "StatementOfWork" / "SOW" in the enterprise table
SOW_SOURCE_TYPES: FrozenSet[NodeType] = frozenset(
    {NodeType.STATEMENT_OF_WORK, NodeType.DOCUMENT}
)

# Multiple (source, target) pairs allowed per edge type
ALLOWED_EDGE_PAIRS: Dict[EdgeType, List[Tuple[NodeType, NodeType]]] = {
    EdgeType.HAS_DOCUMENT: [(NodeType.PROJECT, NodeType.DOCUMENT)],
    EdgeType.CONTAINS_CHUNK: [(NodeType.DOCUMENT, NodeType.CHUNK)],
    EdgeType.REPRESENTS_SOW: [(NodeType.DOCUMENT, NodeType.STATEMENT_OF_WORK)],
    EdgeType.FROM_ROW: [
        (NodeType.RISK, NodeType.DOCUMENT),
        (NodeType.ISSUE, NodeType.DOCUMENT),
    ],
    EdgeType.RECORDED_IN: [(NodeType.RISK, NodeType.DOCUMENT)],
    EdgeType.EXTRACTED_FROM: [],  # validated via EXTRACTOR_SEMANTIC_NODE_TYPES -> Chunk
    EdgeType.DESCRIBES: [(NodeType.CHUNK, NodeType.REQUIREMENT)],
    EdgeType.MITIGATES: [(NodeType.CONTROL, NodeType.RISK)],
    EdgeType.REFERENCES: [(NodeType.RISK, NodeType.REQUIREMENT)],
    EdgeType.RELATES_TO: [(NodeType.RISK, NodeType.ISSUE)],
    EdgeType.USES_PRODUCT: [(NodeType.PROJECT, NodeType.PRODUCT)],
    EdgeType.IMPLIES_ASSUMPTION: [(NodeType.CHUNK, NodeType.ASSUMPTION)],
    # Enterprise extensions (from delivery / SOW relationship table)
    EdgeType.IS_PART_OF: [
        (NodeType.DELIVERABLE, NodeType.PROJECT),
        (NodeType.MILESTONE, NodeType.PROJECT),
        (NodeType.RISK, NodeType.PROJECT),
        (NodeType.CHANGE_REQUEST, NodeType.PROJECT),
        (NodeType.REQUIREMENT, NodeType.PROJECT),
        (NodeType.ASSUMPTION, NodeType.PROJECT),
        (NodeType.STATEMENT_OF_WORK, NodeType.PROJECT),
    ],
    EdgeType.IS_RESPONSIBLE_FOR: [
        (NodeType.ROLE, NodeType.DELIVERABLE),
        (NodeType.ORGANIZATION, NodeType.DELIVERABLE),
    ],
    EdgeType.ASSOCIATED_WITH: [(NodeType.MILESTONE, NodeType.DELIVERABLE)],
    EdgeType.HAS_DELIVERABLE: [
        (NodeType.STATEMENT_OF_WORK, NodeType.DELIVERABLE),
        (NodeType.SERVICE, NodeType.DELIVERABLE),
    ],
    EdgeType.REQUIRES_ACCEPTANCE_OF: [
        (NodeType.STATEMENT_OF_WORK, NodeType.DELIVERABLE),
    ],
    EdgeType.DEFINES: [
        (NodeType.STATEMENT_OF_WORK, NodeType.DELIVERABLE),
        (NodeType.STATEMENT_OF_WORK, NodeType.MILESTONE),
        (NodeType.STATEMENT_OF_WORK, NodeType.REQUIREMENT),
        (NodeType.STATEMENT_OF_WORK, NodeType.ASSUMPTION),
    ],
    EdgeType.PRODUCES: [(NodeType.ACTIVITY, NodeType.DELIVERABLE)],
    EdgeType.HAS_MILESTONE: [(NodeType.STATEMENT_OF_WORK, NodeType.MILESTONE)],
    EdgeType.HAS_RISK: [(NodeType.PROJECT, NodeType.RISK)],
    EdgeType.INCLUDES: [
        (NodeType.PROJECT, NodeType.MILESTONE),
        (NodeType.PROJECT, NodeType.CHANGE_REQUEST),
    ],
    EdgeType.HAS_DEPENDENCY_ON: [(NodeType.PROJECT, NodeType.PROJECT)],
    EdgeType.REFERENCES_REQUIREMENT: [
        (NodeType.PROJECT, NodeType.REQUIREMENT),
        (NodeType.STATEMENT_OF_WORK, NodeType.REQUIREMENT),
    ],
    EdgeType.COVERS: [(NodeType.STATEMENT_OF_WORK, NodeType.PROJECT)],
    EdgeType.MANAGED_VIA: [(NodeType.STATEMENT_OF_WORK, NodeType.CHANGE_CONTROL)],
    EdgeType.OUTLINES_OBJECTIVE_FOR: [(NodeType.STATEMENT_OF_WORK, NodeType.PROJECT)],
    # Phase 5.2 cross-artifact
    EdgeType.DESCRIBES_CURRENT_STATE_OF: [
        (NodeType.ACTIVITY, NodeType.SYSTEM_COMPONENT),
        (NodeType.REQUIREMENT, NodeType.SYSTEM_COMPONENT),
    ],
    EdgeType.IMPLEMENTS: [
        (NodeType.ACTIVITY, NodeType.DELIVERABLE),
        (NodeType.REQUIREMENT, NodeType.DELIVERABLE),
    ],
    EdgeType.ADDRESSES: [
        (NodeType.RISK, NodeType.DELIVERABLE),
        (NodeType.RISK, NodeType.MILESTONE),
        (NodeType.CONTROL, NodeType.DELIVERABLE),
        (NodeType.CONTROL, NodeType.MILESTONE),
    ],
    EdgeType.TRACES_TO: [(NodeType.REQUIREMENT, NodeType.REQUIREMENT)],
    EdgeType.GAPS: [
        (NodeType.RISK, NodeType.DELIVERABLE),
        (NodeType.RISK, NodeType.REQUIREMENT),
    ],
    EdgeType.SAME_AS: [
        (NodeType.DELIVERABLE, NodeType.SYSTEM_COMPONENT),
        (NodeType.ACTIVITY, NodeType.SYSTEM_COMPONENT),
        (NodeType.REQUIREMENT, NodeType.SYSTEM_COMPONENT),
        (NodeType.PRODUCT, NodeType.SYSTEM_COMPONENT),
        (NodeType.DELIVERABLE, NodeType.DELIVERABLE),
        (NodeType.ACTIVITY, NodeType.ACTIVITY),
        (NodeType.REQUIREMENT, NodeType.REQUIREMENT),
        (NodeType.SYSTEM_COMPONENT, NodeType.SYSTEM_COMPONENT),
    ],
}

# Back-compat: first pair per edge (deprecated for multi-pair types)
ALLOWED_EDGES: Dict[EdgeType, Tuple[NodeType, NodeType]] = {
    et: pairs[0] for et, pairs in ALLOWED_EDGE_PAIRS.items() if pairs
}

# Semantic node types the relation extractor may emit (Word documents)
EXTRACTOR_SEMANTIC_NODE_TYPES: FrozenSet[NodeType] = frozenset(
    {
        NodeType.REQUIREMENT,
        NodeType.CONTROL,
        NodeType.ASSUMPTION,
        NodeType.RISK,
        NodeType.PRODUCT,
        NodeType.DELIVERABLE,
        NodeType.MILESTONE,
        NodeType.ACTIVITY,
        NodeType.SERVICE,
        NodeType.CHANGE_REQUEST,
        NodeType.CHANGE_CONTROL,
        NodeType.ROLE,
        NodeType.ORGANIZATION,
        NodeType.SYSTEM_COMPONENT,
    }
)

# Per document_type (payload value), which semantic node types the LLM may propose
SEMANTIC_NODES_BY_DOCUMENT_TYPE: Dict[str, FrozenSet[NodeType]] = {
    "statement of work": frozenset(
        {
            NodeType.REQUIREMENT,
            NodeType.DELIVERABLE,
            NodeType.MILESTONE,
            NodeType.ACTIVITY,
            NodeType.SERVICE,
            NodeType.CHANGE_REQUEST,
            NodeType.CHANGE_CONTROL,
            NodeType.ROLE,
            NodeType.ORGANIZATION,
            NodeType.ASSUMPTION,
        }
    ),
    "solution description document": frozenset(
        {
            NodeType.REQUIREMENT,
            NodeType.RISK,
            NodeType.CONTROL,
            NodeType.ASSUMPTION,
            NodeType.DELIVERABLE,
            NodeType.ACTIVITY,
            NodeType.SYSTEM_COMPONENT,
        }
    ),
    "proposal document": frozenset(
        {NodeType.REQUIREMENT, NodeType.PRODUCT, NodeType.DELIVERABLE, NodeType.MILESTONE}
    ),
    "risk register": frozenset({NodeType.RISK}),
    "issue log": frozenset({NodeType.ISSUE}),
}

# Enterprise edges the extractor may emit when document_type is SOW
SOW_EXTRACTABLE_EDGES: FrozenSet[EdgeType] = frozenset(
    {
        EdgeType.DEFINES,
        EdgeType.HAS_DELIVERABLE,
        EdgeType.HAS_MILESTONE,
        EdgeType.REQUIRES_ACCEPTANCE_OF,
        EdgeType.REFERENCES_REQUIREMENT,
        EdgeType.IS_PART_OF,
        EdgeType.INCLUDES,
        EdgeType.IS_RESPONSIBLE_FOR,
        EdgeType.ASSOCIATED_WITH,
        EdgeType.PRODUCES,
        EdgeType.COVERS,
        EdgeType.OUTLINES_OBJECTIVE_FOR,
        EdgeType.MANAGED_VIA,
        EdgeType.EXTRACTED_FROM,
    }
)

# Phase 5.2: edges the solution-doc extractor may emit (within-doc only; no SOW IDs)
SOLUTION_EXTRACTABLE_EDGES: FrozenSet[EdgeType] = frozenset(
    {
        EdgeType.EXTRACTED_FROM,
        EdgeType.DESCRIBES,
        EdgeType.DESCRIBES_CURRENT_STATE_OF,
        EdgeType.GAPS,
        EdgeType.MITIGATES,
        EdgeType.REFERENCES,
        EdgeType.RELATES_TO,
        EdgeType.IMPLIES_ASSUMPTION,
        EdgeType.PRODUCES,
    }
)

# Cross-document edge types written by entity_linking_service (not per-chunk extract)
CROSS_LINK_EDGE_TYPES: FrozenSet[EdgeType] = frozenset(
    {
        EdgeType.IMPLEMENTS,
        EdgeType.ADDRESSES,
        EdgeType.TRACES_TO,
        EdgeType.SAME_AS,
        EdgeType.DESCRIBES_CURRENT_STATE_OF,
        EdgeType.GAPS,
    }
)

CROSS_ARTIFACT_ONTOLOGY_VERSION = "0.3.0-cross-artifact"

# Deliverable / milestone ID tokens shared across SOW and solution text
DELIVERABLE_ID_RE = re.compile(r"\b([Dd]-\d+)\b")
MILESTONE_ID_RE = re.compile(r"\b([Mm]-\d+)\b")


def normalize_slug(label: str, max_len: int = 48) -> str:
    """Lowercase slug for derived semantic node IDs."""
    s = _SLUG_RE.sub("_", (label or "").strip().lower()).strip("_")
    return (s[:max_len] if s else "unknown")


def graph_id_project(project_name: str) -> str:
    return f"project:{project_name.strip()}"


def graph_id_document(document_id: str) -> str:
    return f"document:{document_id}"


def graph_id_chunk(chunk_id: str) -> str:
    return f"chunk:{chunk_id}"


def graph_id_statement_of_work(document_id: str) -> str:
    """1:1 StatementOfWork node for a SOW file."""
    return f"sow:{document_id}"


def graph_id_customer(name: str) -> str:
    return f"customer:{normalize_slug(name, 80)}"


def graph_id_product(name: str) -> str:
    return f"product:{normalize_slug(name, 80)}"


def graph_id_system_component(project_name: str, slug: str) -> str:
    """Project-scoped SystemComponent hub: comp:{project_slug}:{component_slug}."""
    proj = normalize_slug(project_name, 64)
    return f"comp:{proj}:{normalize_slug(slug, 64)}"


def graph_id_risk_row(document_id: str, row_number: int) -> str:
    return f"risk:{document_id}:row:{row_number}"


def graph_id_issue_row(document_id: str, row_number: int) -> str:
    return f"issue:{document_id}:row:{row_number}"


def graph_id_semantic(
    prefix: str,
    document_id: str,
    chunk_id: str,
    slug: str,
) -> str:
    """Semantic entities: req/del/mls/...:{document_id}:{chunk_id}:{slug}."""
    return f"{prefix}:{document_id}:{chunk_id}:{normalize_slug(slug)}"


SEMANTIC_ID_PREFIX: Dict[NodeType, str] = {
    NodeType.REQUIREMENT: "req",
    NodeType.CONTROL: "ctrl",
    NodeType.ASSUMPTION: "asm",
    NodeType.RISK: "risk_sem",
    NodeType.PRODUCT: "prod",
    NodeType.DELIVERABLE: "del",
    NodeType.MILESTONE: "mls",
    NodeType.ACTIVITY: "act",
    NodeType.SERVICE: "svc",
    NodeType.CHANGE_REQUEST: "chg",
    NodeType.CHANGE_CONTROL: "cc",
    NodeType.ROLE: "role",
    NodeType.ORGANIZATION: "org",
    NodeType.SYSTEM_COMPONENT: "comp",
}


def validate_edge_endpoints(
    edge_type: EdgeType,
    source_type: NodeType,
    target_type: NodeType,
    *,
    source_document_type: Optional[str] = None,
) -> bool:
    """
    Return True if source_type -> target_type is allowed for edge_type.

    When source is Document and the edge expects StatementOfWork, pass
    source_document_type='statement of work' (or use STATEMENT_OF_WORK node).
    """
    pairs = ALLOWED_EDGE_PAIRS.get(edge_type)
    if pairs is None:
        return False

    if edge_type == EdgeType.EXTRACTED_FROM:
        return (
            source_type in EXTRACTOR_SEMANTIC_NODE_TYPES
            or source_type == NodeType.SYSTEM_COMPONENT
        ) and target_type == NodeType.CHUNK

    src = source_type
    if src == NodeType.DOCUMENT and any(p[0] == NodeType.STATEMENT_OF_WORK for p in pairs):
        if source_document_type == SOW_DOCUMENT_TYPE:
            src = NodeType.STATEMENT_OF_WORK
        else:
            return False

    return (src, target_type) in pairs


class GraphNode(BaseModel):
    """Node upsert payload for GraphStore."""

    node_id: str = Field(..., min_length=1)
    node_type: NodeType
    label: str = Field(..., min_length=1)
    project_name: str = Field(..., min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)

    @field_validator("project_name")
    @classmethod
    def strip_project(cls, v: str) -> str:
        return v.strip()


class GraphEdge(BaseModel):
    """Edge upsert payload for GraphStore."""

    edge_type: EdgeType
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    project_name: str = Field(..., min_length=1)
    evidence_text: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("project_name")
    @classmethod
    def strip_project(cls, v: str) -> str:
        return v.strip()


class ExtractedNode(BaseModel):
    """Single node from relation extractor JSON (before graph_id normalization)."""

    node_id: str = Field(..., min_length=1, description="Extractor-local id; mapped on ingest")
    node_type: NodeType
    label: str = Field(..., min_length=1)
    properties: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class ExtractedEdge(BaseModel):
    """Single edge from relation extractor JSON."""

    edge_type: EdgeType
    source_id: str = Field(..., min_length=1)
    target_id: str = Field(..., min_length=1)
    evidence_text: Optional[str] = None
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class RelationExtractionResult(BaseModel):
    """Validated LLM output for one chunk (Phase 2.3)."""

    chunk_id: str
    document_id: str
    project_name: str
    nodes: List[ExtractedNode] = Field(default_factory=list)
    edges: List[ExtractedEdge] = Field(default_factory=list)

    @field_validator("project_name")
    @classmethod
    def strip_project(cls, v: str) -> str:
        return v.strip()


def filter_extraction_by_confidence(
    result: RelationExtractionResult,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> RelationExtractionResult:
    """Drop nodes/edges below threshold (assumptions should use same gate in MVP)."""
    nodes = [n for n in result.nodes if n.confidence >= min_confidence]
    kept_ids = {n.node_id for n in nodes}
    edges = [
        e
        for e in result.edges
        if e.confidence >= min_confidence
        and e.source_id in kept_ids
        and e.target_id in kept_ids
    ]
    return result.model_copy(update={"nodes": nodes, "edges": edges})


class GraphOntologyVersion(BaseModel):
    """Version stamp written with graph data for migrations."""

    version: str = "0.3.0-cross-artifact"
    notes: str = "Phase 5.2 SystemComponent hubs and SOW↔solution cross-doc edges"
