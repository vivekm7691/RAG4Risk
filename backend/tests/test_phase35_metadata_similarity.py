"""Phase 3.5: metadata similarity weighting."""

from app.models.document import ProjectMetadata
from app.services.project_similarity import ProjectSimilarityService


def test_metadata_similarity_identical_projects_high_score():
    svc = ProjectSimilarityService()
    a = ProjectMetadata(
        project_name="p1",
        customer="Acme",
        csg_products=["Singleview"],
        csg_role="Prime Contractor",
        integration_complexity="High",
        client_type="Enterprise",
        project_size="Large",
        project_complexity="Complex",
        date_range={"start_date": "2023-01-01", "end_date": "2024-01-01"},
    )
    b = ProjectMetadata(
        project_name="p2",
        customer="Other",
        csg_products=["Singleview"],
        csg_role="Prime Contractor",
        integration_complexity="High",
        client_type="Enterprise",
        project_size="Large",
        project_complexity="Complex",
        date_range={"start_date": "2023-06-01", "end_date": "2024-01-01"},
    )
    score = svc._metadata_similarity(a, b)
    assert score >= 0.85


def test_metadata_similarity_unrelated_projects_lower_than_identical():
    svc = ProjectSimilarityService()
    a = ProjectMetadata(
        project_name="p1",
        customer="A",
        csg_products=["A"],
        csg_role="Prime Contractor",
        integration_complexity="Low",
        client_type="SMB",
        project_size="Small",
        project_complexity="Simple",
        date_range={"end_date": "2020-01-01"},
    )
    b = ProjectMetadata(
        project_name="p2",
        customer="B",
        csg_products=["Z"],
        csg_role="Consultant",
        integration_complexity="High",
        client_type="Government",
        project_size="Large",
        project_complexity="Complex",
        date_range={"end_date": "2024-01-01"},
    )
    same = svc._metadata_similarity(a, a)
    diff = svc._metadata_similarity(a, b)
    assert same > diff
