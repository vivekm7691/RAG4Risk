"""Tests for relation_extractor_service helpers."""

import httpx

from app.services.relation_extractor_service import _format_extraction_error


def test_format_extraction_error_read_timeout_empty_message():
    exc = httpx.ReadTimeout("")
    assert (
        _format_extraction_error(exc, timeout_seconds=1800, model="qwen2.5:14b")
        == "ReadTimeout after 1800s (model=qwen2.5:14b)"
    )


def test_format_extraction_error_with_message():
    exc = RuntimeError("Relation extractor: empty Ollama response")
    assert (
        _format_extraction_error(exc, timeout_seconds=120, model="llama3.2:3b")
        == "RuntimeError: Relation extractor: empty Ollama response"
    )


def test_format_extraction_error_empty_non_timeout():
    exc = RuntimeError("")
    assert (
        _format_extraction_error(exc, timeout_seconds=120, model="llama3.2:3b")
        == "RuntimeError (RuntimeError(''))"
    )
