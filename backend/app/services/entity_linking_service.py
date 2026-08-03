"""Phase 5.2: project-level entity linking (SOW ↔ solution / SystemComponent)."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from app.config import settings
from app.models.document import DocumentType, ProjectMetadata
from app.models.graph import (
    CROSS_ARTIFACT_ONTOLOGY_VERSION,
    DELIVERABLE_ID_RE,
    MILESTONE_ID_RE,
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
    graph_id_system_component,
    normalize_slug,
    validate_edge_endpoints,
)
from app.services.graph_store import get_graph_store
from app.services.project_metadata import ProjectMetadataService

logger = logging.getLogger(__name__)

SOW_DOC = DocumentType.STATEMENT_OF_WORK.value
SOLUTION_DOC = DocumentType.SOLUTION_DESCRIPTION.value

# Resolution-pass edges (rewritten on each link run; extractor edges keep no link_method)
RESOLUTION_EDGE_TYPES: Tuple[EdgeType, ...] = (
    EdgeType.IMPLEMENTS,
    EdgeType.ADDRESSES,
    EdgeType.TRACES_TO,
    EdgeType.SAME_AS,
)

SOW_CANDIDATE_TYPES = (
    NodeType.DELIVERABLE.value,
    NodeType.MILESTONE.value,
    NodeType.REQUIREMENT.value,
    NodeType.ROLE.value,
)
SOLUTION_CANDIDATE_TYPES = (
    NodeType.ACTIVITY.value,
    NodeType.REQUIREMENT.value,
    NodeType.RISK.value,
    NodeType.CONTROL.value,
    NodeType.DELIVERABLE.value,
)
CANONICAL_TYPES = (
    NodeType.SYSTEM_COMPONENT.value,
    NodeType.PRODUCT.value,
)

ID_MATCH_CONFIDENCE = 0.95
SLUG_MATCH_CONFIDENCE = 0.88
METADATA_SAME_AS_CONFIDENCE = 0.90


@dataclass(frozen=True)
class LinkCandidate:
    node_id: str
    node_type: str
    label: str
    document_id: Optional[str] = None
    document_type: Optional[str] = None


@dataclass(frozen=True)
class ProposedLink:
    edge_type: EdgeType
    source_id: str
    target_id: str
    source_type: NodeType
    target_type: NodeType
    confidence: float
    link_method: str
    evidence_text: Optional[str] = None


def extract_contract_ids(text: str) -> Set[str]:
    """Return normalized contract IDs (D-12, M-3) found in text."""
    found: Set[str] = set()
    for m in DELIVERABLE_ID_RE.finditer(text or ""):
        found.add(m.group(1).upper())
    for m in MILESTONE_ID_RE.finditer(text or ""):
        found.add(m.group(1).upper())
    return found


def _node_type_enum(value: str) -> Optional[NodeType]:
    try:
        return NodeType(value)
    except ValueError:
        return None


def _is_sow(c: LinkCandidate) -> bool:
    return (c.document_type or "") == SOW_DOC


def _is_solution(c: LinkCandidate) -> bool:
    return (c.document_type or "") == SOLUTION_DOC


def build_system_component_nodes(
    meta: ProjectMetadata,
) -> List[GraphNode]:
    """Canonical SystemComponent hubs from project metadata products."""
    project_name = meta.project_name.strip()
    nodes: List[GraphNode] = []
    for product in meta.csg_products or []:
        name = (product or "").strip()
        if not name:
            continue
        nodes.append(
            GraphNode(
                node_id=graph_id_system_component(project_name, name),
                node_type=NodeType.SYSTEM_COMPONENT,
                label=name,
                project_name=project_name,
                properties={"source": "project_metadata"},
                confidence=1.0,
            )
        )
    return nodes


def propose_id_matches(
    sow_nodes: List[LinkCandidate],
    solution_nodes: List[LinkCandidate],
) -> List[ProposedLink]:
    """Match when the same D-/M- ID appears in both SOW and solution labels."""
    sow_by_id: Dict[str, List[LinkCandidate]] = defaultdict(list)
    for n in sow_nodes:
        for cid in extract_contract_ids(n.label):
            sow_by_id[cid].append(n)

    sol_by_id: Dict[str, List[LinkCandidate]] = defaultdict(list)
    for n in solution_nodes:
        for cid in extract_contract_ids(n.label):
            sol_by_id[cid].append(n)

    proposals: List[ProposedLink] = []
    for cid, sols in sol_by_id.items():
        sows = sow_by_id.get(cid) or []
        if not sows:
            continue
        if len(sows) > 1:
            logger.info(
                "ambiguous_candidates id=%s sow_count=%d — skipping id_match",
                cid,
                len(sows),
            )
            continue
        sow = sows[0]
        sow_type = _node_type_enum(sow.node_type)
        if sow_type is None:
            continue
        for sol in sols:
            sol_type = _node_type_enum(sol.node_type)
            if sol_type is None:
                continue
            edge = _edge_for_pair(sol_type, sow_type)
            if edge is None:
                continue
            proposals.append(
                ProposedLink(
                    edge_type=edge,
                    source_id=sol.node_id,
                    target_id=sow.node_id,
                    source_type=sol_type,
                    target_type=sow_type,
                    confidence=ID_MATCH_CONFIDENCE,
                    link_method="id_match",
                    evidence_text=f"Shared contract id {cid}",
                )
            )
    return proposals


def propose_slug_matches(
    sow_nodes: List[LinkCandidate],
    solution_nodes: List[LinkCandidate],
    components: List[LinkCandidate],
) -> List[ProposedLink]:
    """Slug-equal labels across solution↔SOW and SOW/solution↔SystemComponent."""
    proposals: List[ProposedLink] = []

    sow_by_slug: Dict[str, List[LinkCandidate]] = defaultdict(list)
    for n in sow_nodes:
        sow_by_slug[normalize_slug(n.label)].append(n)

    sol_by_slug: Dict[str, List[LinkCandidate]] = defaultdict(list)
    for n in solution_nodes:
        sol_by_slug[normalize_slug(n.label)].append(n)

    for slug, sols in sol_by_slug.items():
        if slug == "unknown":
            continue
        sows = sow_by_slug.get(slug) or []
        if len(sows) > 1:
            logger.info(
                "ambiguous_candidates slug=%s sow_count=%d — skipping slug match",
                slug,
                len(sows),
            )
            continue
        if len(sows) == 1:
            sow = sows[0]
            sow_type = _node_type_enum(sow.node_type)
            for sol in sols:
                sol_type = _node_type_enum(sol.node_type)
                if sow_type is None or sol_type is None:
                    continue
                edge = _edge_for_pair(sol_type, sow_type)
                if edge is None:
                    continue
                proposals.append(
                    ProposedLink(
                        edge_type=edge,
                        source_id=sol.node_id,
                        target_id=sow.node_id,
                        source_type=sol_type,
                        target_type=sow_type,
                        confidence=SLUG_MATCH_CONFIDENCE,
                        link_method="slug",
                        evidence_text=f"Slug match {slug}",
                    )
                )

    comp_by_slug = {normalize_slug(c.label): c for c in components}
    for side in (*sow_nodes, *solution_nodes):
        slug = normalize_slug(side.label)
        comp = comp_by_slug.get(slug)
        if not comp or slug == "unknown":
            continue
        src_type = _node_type_enum(side.node_type)
        if src_type is None:
            continue
        if not validate_edge_endpoints(
            EdgeType.SAME_AS, src_type, NodeType.SYSTEM_COMPONENT
        ):
            continue
        proposals.append(
            ProposedLink(
                edge_type=EdgeType.SAME_AS,
                source_id=side.node_id,
                target_id=comp.node_id,
                source_type=src_type,
                target_type=NodeType.SYSTEM_COMPONENT,
                confidence=SLUG_MATCH_CONFIDENCE,
                link_method="slug",
                evidence_text=f"Slug match to SystemComponent {slug}",
            )
        )

    return proposals


def propose_metadata_component_links(
    sow_nodes: List[LinkCandidate],
    components: List[LinkCandidate],
) -> List[ProposedLink]:
    """SAME_AS when SOW deliverable label contains a SystemComponent product name."""
    proposals: List[ProposedLink] = []
    for comp in components:
        comp_slug = normalize_slug(comp.label)
        if comp_slug == "unknown":
            continue
        token = comp.label.strip().lower()
        matches = [
            n
            for n in sow_nodes
            if n.node_type == NodeType.DELIVERABLE.value
            and (
                normalize_slug(n.label) == comp_slug
                or token in (n.label or "").lower()
            )
        ]
        if len(matches) != 1:
            if len(matches) > 1:
                logger.info(
                    "ambiguous_candidates component=%s deliverable_count=%d",
                    comp.label,
                    len(matches),
                )
            continue
        deliv = matches[0]
        proposals.append(
            ProposedLink(
                edge_type=EdgeType.SAME_AS,
                source_id=deliv.node_id,
                target_id=comp.node_id,
                source_type=NodeType.DELIVERABLE,
                target_type=NodeType.SYSTEM_COMPONENT,
                confidence=METADATA_SAME_AS_CONFIDENCE,
                link_method="slug",
                evidence_text=f"Metadata product '{comp.label}' in deliverable label",
            )
        )
    return proposals


def _edge_for_pair(sol_type: NodeType, sow_type: NodeType) -> Optional[EdgeType]:
    """Choose resolution edge type for solution → SOW pair."""
    candidates = (
        EdgeType.IMPLEMENTS,
        EdgeType.ADDRESSES,
        EdgeType.TRACES_TO,
        EdgeType.GAPS,
        EdgeType.SAME_AS,
    )
    for et in candidates:
        if validate_edge_endpoints(et, sol_type, sow_type):
            return et
    return None


def filter_and_cap_proposals(
    proposals: List[ProposedLink],
    *,
    min_confidence: float,
    max_per_source: int,
) -> List[ProposedLink]:
    """Keep high-confidence links; prefer id_match over slug; cap per source."""
    method_rank = {"id_match": 0, "slug": 1, "embedding": 2, "llm": 3, "manual": 4}
    # Deduplicate by (source, target, edge_type), keep best confidence / method
    best: Dict[Tuple[str, str, str], ProposedLink] = {}
    for p in proposals:
        if p.confidence < min_confidence:
            continue
        if not validate_edge_endpoints(p.edge_type, p.source_type, p.target_type):
            continue
        key = (p.source_id, p.target_id, p.edge_type.value)
        prev = best.get(key)
        if prev is None:
            best[key] = p
            continue
        if p.confidence > prev.confidence:
            best[key] = p
        elif p.confidence == prev.confidence and method_rank.get(
            p.link_method, 9
        ) < method_rank.get(prev.link_method, 9):
            best[key] = p

    by_source: Dict[str, List[ProposedLink]] = defaultdict(list)
    for p in best.values():
        by_source[p.source_id].append(p)

    capped: List[ProposedLink] = []
    for src, items in by_source.items():
        items.sort(key=lambda x: (-x.confidence, method_rank.get(x.link_method, 9)))
        capped.extend(items[: max(1, max_per_source)])
    return capped


def proposals_to_edges(
    proposals: List[ProposedLink],
    *,
    project_name: str,
) -> List[GraphEdge]:
    edges: List[GraphEdge] = []
    for p in proposals:
        edges.append(
            GraphEdge(
                edge_type=p.edge_type,
                source_id=p.source_id,
                target_id=p.target_id,
                project_name=project_name,
                evidence_text=p.evidence_text,
                confidence=p.confidence,
                properties={
                    "link_method": p.link_method,
                    "ontology_version": CROSS_ARTIFACT_ONTOLOGY_VERSION,
                },
            )
        )
    return edges


def link_candidates_from_rows(rows: List[Dict[str, Any]]) -> List[LinkCandidate]:
    return [
        LinkCandidate(
            node_id=r["node_id"],
            node_type=r.get("node_type") or "",
            label=r.get("label") or "",
            document_id=r.get("document_id"),
            document_type=r.get("document_type"),
        )
        for r in rows
        if r.get("node_id")
    ]


async def resolve_project_cross_links(project_name: str) -> Dict[str, int]:
    """
    Upsert SystemComponent hubs and write high-confidence SOW↔solution edges.

    Returns counts: components, edges_written, edges_cleared.
    """
    if not settings.GRAPH_ENABLED:
        return {"components": 0, "edges_written": 0, "edges_cleared": 0}

    project_name = project_name.strip()
    store = await get_graph_store()
    meta_service = ProjectMetadataService()
    meta = meta_service.get_project_metadata(project_name)

    component_nodes: List[GraphNode] = []
    if meta:
        component_nodes = build_system_component_nodes(meta)
        if component_nodes:
            await store.upsert_nodes(component_nodes)

    all_types = list(
        dict.fromkeys(
            (*SOW_CANDIDATE_TYPES, *SOLUTION_CANDIDATE_TYPES, *CANONICAL_TYPES)
        )
    )
    rows = await store.list_nodes_by_types(project_name, all_types)
    candidates = link_candidates_from_rows(rows)

    sow_nodes = [c for c in candidates if _is_sow(c)]
    solution_nodes = [c for c in candidates if _is_solution(c)]
    components = [
        c for c in candidates if c.node_type == NodeType.SYSTEM_COMPONENT.value
    ]
    # Include freshly built components not yet visible if list missed them
    if component_nodes:
        existing = {c.node_id for c in components}
        for gn in component_nodes:
            if gn.node_id not in existing:
                components.append(
                    LinkCandidate(
                        node_id=gn.node_id,
                        node_type=gn.node_type.value,
                        label=gn.label,
                    )
                )

    proposals: List[ProposedLink] = []
    proposals.extend(propose_id_matches(sow_nodes, solution_nodes))
    proposals.extend(propose_slug_matches(sow_nodes, solution_nodes, components))
    if components:
        proposals.extend(propose_metadata_component_links(sow_nodes, components))

    min_conf = float(settings.GRAPH_CROSS_LINK_MIN_CONFIDENCE)
    max_per = int(settings.GRAPH_CROSS_LINK_MAX_EDGES_PER_SOURCE)
    kept = filter_and_cap_proposals(
        proposals, min_confidence=min_conf, max_per_source=max_per
    )

    cleared = await store.delete_resolution_edges(
        project_name,
        [e.value for e in RESOLUTION_EDGE_TYPES],
    )
    edges = proposals_to_edges(kept, project_name=project_name)
    if edges:
        await store.upsert_edges(edges)

    logger.info(
        "Cross-link resolve project=%s components=%d edges_written=%d edges_cleared=%d",
        project_name,
        len(component_nodes),
        len(edges),
        cleared,
    )
    return {
        "components": len(component_nodes),
        "edges_written": len(edges),
        "edges_cleared": cleared,
    }
