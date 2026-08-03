"""Phase 5.2: entity linking (SOW ↔ solution) unit tests."""

from app.models.document import ProjectMetadata
from app.models.graph import (
    EdgeType,
    NodeType,
    graph_id_system_component,
    validate_edge_endpoints,
)
from app.services.entity_linking_service import (
    LinkCandidate,
    ProposedLink,
    build_system_component_nodes,
    extract_contract_ids,
    filter_and_cap_proposals,
    propose_id_matches,
    propose_metadata_component_links,
    propose_slug_matches,
    proposals_to_edges,
)


def test_extract_contract_ids():
    assert extract_contract_ids("Deliverable D-12 billing") == {"D-12"}
    assert extract_contract_ids("Cutover M-3 and d-12") == {"M-3", "D-12"}
    assert extract_contract_ids("no ids here") == set()


def test_graph_id_system_component():
    assert (
        graph_id_system_component("Freedom Encompass Modernization", "Encompass Billing")
        == "comp:freedom_encompass_modernization:encompass_billing"
    )


def test_build_system_component_nodes_from_metadata():
    meta = ProjectMetadata(
        project_name="Freedom Encompass Modernization",
        customer="Acme",
        csg_role="SI",
        csg_products=["Encompass", "Billing"],
        integration_complexity="high",
        client_type="bank",
        project_size="large",
        project_complexity="high",
    )
    nodes = build_system_component_nodes(meta)
    assert len(nodes) == 2
    assert all(n.node_type == NodeType.SYSTEM_COMPONENT for n in nodes)
    assert nodes[0].node_id.startswith("comp:freedom_encompass_modernization:")


def test_propose_id_match_implements_d12():
    sow = [
        LinkCandidate(
            node_id="del:doc-sow:ch3:d_12_billing_integration",
            node_type=NodeType.DELIVERABLE.value,
            label="D-12 Billing Integration",
            document_type="statement of work",
        )
    ]
    sol = [
        LinkCandidate(
            node_id="act:doc-sol:ch4:billing_migration_approach",
            node_type=NodeType.ACTIVITY.value,
            label="Billing migration for D-12",
            document_type="solution description document",
        )
    ]
    links = propose_id_matches(sow, sol)
    assert len(links) == 1
    assert links[0].edge_type == EdgeType.IMPLEMENTS
    assert links[0].link_method == "id_match"
    assert links[0].source_id == sol[0].node_id
    assert links[0].target_id == sow[0].node_id


def test_propose_id_match_skips_ambiguous_sow():
    sow = [
        LinkCandidate(
            node_id="del:a",
            node_type=NodeType.DELIVERABLE.value,
            label="D-12 A",
            document_type="statement of work",
        ),
        LinkCandidate(
            node_id="del:b",
            node_type=NodeType.DELIVERABLE.value,
            label="D-12 B",
            document_type="statement of work",
        ),
    ]
    sol = [
        LinkCandidate(
            node_id="act:x",
            node_type=NodeType.ACTIVITY.value,
            label="Implements D-12",
            document_type="solution description document",
        )
    ]
    assert propose_id_matches(sow, sol) == []


def test_propose_slug_match_same_as_component():
    sow = [
        LinkCandidate(
            node_id="del:doc:ch:billing_integration",
            node_type=NodeType.DELIVERABLE.value,
            label="Billing Integration",
            document_type="statement of work",
        )
    ]
    components = [
        LinkCandidate(
            node_id="comp:proj:billing_integration",
            node_type=NodeType.SYSTEM_COMPONENT.value,
            label="Billing Integration",
        )
    ]
    links = propose_slug_matches(sow, [], components)
    assert len(links) == 1
    assert links[0].edge_type == EdgeType.SAME_AS
    assert links[0].target_id == components[0].node_id


def test_propose_metadata_component_links():
    sow = [
        LinkCandidate(
            node_id="del:sow:ch:d12",
            node_type=NodeType.DELIVERABLE.value,
            label="D-12 Encompass billing upgrade",
            document_type="statement of work",
        )
    ]
    components = [
        LinkCandidate(
            node_id="comp:p:encompass",
            node_type=NodeType.SYSTEM_COMPONENT.value,
            label="Encompass",
        )
    ]
    links = propose_metadata_component_links(sow, components)
    assert len(links) == 1
    assert links[0].edge_type == EdgeType.SAME_AS


def test_filter_and_cap_and_edge_provenance():
    proposals = [
        ProposedLink(
            edge_type=EdgeType.IMPLEMENTS,
            source_id="act:1",
            target_id="del:1",
            source_type=NodeType.ACTIVITY,
            target_type=NodeType.DELIVERABLE,
            confidence=0.95,
            link_method="id_match",
        ),
        ProposedLink(
            edge_type=EdgeType.IMPLEMENTS,
            source_id="act:1",
            target_id="del:2",
            source_type=NodeType.ACTIVITY,
            target_type=NodeType.DELIVERABLE,
            confidence=0.88,
            link_method="slug",
        ),
        ProposedLink(
            edge_type=EdgeType.IMPLEMENTS,
            source_id="act:1",
            target_id="del:3",
            source_type=NodeType.ACTIVITY,
            target_type=NodeType.DELIVERABLE,
            confidence=0.88,
            link_method="slug",
        ),
    ]
    kept = filter_and_cap_proposals(proposals, min_confidence=0.80, max_per_source=2)
    assert len(kept) == 2
    edges = proposals_to_edges(kept, project_name="P")
    assert edges[0].properties["link_method"] == "id_match"
    assert edges[0].properties["ontology_version"] == "0.3.0-cross-artifact"


def test_cross_doc_edge_validation():
    assert validate_edge_endpoints(
        EdgeType.IMPLEMENTS, NodeType.ACTIVITY, NodeType.DELIVERABLE
    )
    assert validate_edge_endpoints(
        EdgeType.DESCRIBES_CURRENT_STATE_OF,
        NodeType.ACTIVITY,
        NodeType.SYSTEM_COMPONENT,
    )
    assert validate_edge_endpoints(
        EdgeType.ADDRESSES, NodeType.RISK, NodeType.MILESTONE
    )
    assert not validate_edge_endpoints(
        EdgeType.IMPLEMENTS, NodeType.DELIVERABLE, NodeType.ACTIVITY
    )
