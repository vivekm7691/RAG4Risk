"""Sanitization of QueryFilters before Qdrant (Swagger placeholder values)."""

from app.models.query import QueryFilters, metadata_filters_for_vector_search


def test_sanitize_drops_swagger_string_placeholders():
    f = QueryFilters(
        severity="string",
        status="STRING",
        category="  string  ",
        owner="",
    )
    assert metadata_filters_for_vector_search(f) is None


def test_sanitize_keeps_real_excel_filters():
    f = QueryFilters(severity="High", status="string", owner="alice")
    assert metadata_filters_for_vector_search(f) == {"severity": "High", "owner": "alice"}


def test_sanitize_date_range_only_when_real_dates():
    f = QueryFilters(
        date_range={"additionalProp1": "string", "start_date": "2024-01-01"},
    )
    assert metadata_filters_for_vector_search(f) == {
        "date_range": {"start_date": "2024-01-01"},
    }
