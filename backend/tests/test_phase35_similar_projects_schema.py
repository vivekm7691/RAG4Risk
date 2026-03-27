"""Phase 3.5: similar project response shaping."""

from app.models.document import ProjectMetadata
from app.api.routes.query import _similar_projects_to_preview


def test_similar_projects_to_preview_with_models():
    meta = ProjectMetadata(
        project_name="past1",
        customer="Beta Inc",
        csg_products=[],
    )
    raw = [
        {
            "project_name": "past1",
            "customer": "Beta Inc",
            "similarity_score": 0.8123,
            "metadata": meta,
        }
    ]
    out = _similar_projects_to_preview(raw)
    assert out is not None and len(out) == 1
    assert out[0].project_name == "past1"
    assert out[0].similarity_score == 0.8123
    assert out[0].metadata is not None
    assert out[0].metadata.get("customer") == "Beta Inc"


def test_similar_projects_to_preview_empty():
    assert _similar_projects_to_preview(None) is None
    assert _similar_projects_to_preview([]) is None
