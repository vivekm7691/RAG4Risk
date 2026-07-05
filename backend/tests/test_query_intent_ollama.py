"""Phase 3.75 task 2: Ollama /api/chat intent call (mocked httpx)."""

import httpx
import pytest

from app.services import query_intent_service as qis


_VALID_INTENT_JSON = (
    '{"intent_summary":"test","document_weights":'
    '{"statement of work":0.2,"solution description document":0.2,"proposal document":0.2,'
    '"risk register":0.2,"issue log":0.2}}'
)


def _patch_async_client(monkeypatch, handler):
    real_client = httpx.AsyncClient

    def factory(*args, **kwargs):
        transport = httpx.MockTransport(handler)
        return real_client(transport=transport, timeout=kwargs.get("timeout", 30.0))

    monkeypatch.setattr(qis.httpx, "AsyncClient", factory)


@pytest.mark.asyncio
async def test_ollama_intent_chat_completion_parses_message(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        body = request.content.decode()
        assert "messages" in body
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": _VALID_INTENT_JSON}},
        )

    _patch_async_client(monkeypatch, handler)
    text = await qis.ollama_intent_chat_completion("What are the main risks?")
    assert text.strip() == _VALID_INTENT_JSON


@pytest.mark.asyncio
async def test_classify_from_ollama_end_to_end_mocked(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": _VALID_INTENT_JSON}},
        )

    _patch_async_client(monkeypatch, handler)
    svc = qis.QueryIntentService()
    r = await svc.classify_from_ollama("hello")
    assert not r.used_fallback
    assert abs(sum(r.document_weights.values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_classify_from_ollama_thinking_field_used_when_content_empty(monkeypatch):
    """DeepSeek R1 may put JSON in `thinking` and leave `content` empty."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {
                    "role": "assistant",
                    "content": "",
                    "thinking": "reasoning here\n" + _VALID_INTENT_JSON,
                }
            },
        )

    _patch_async_client(monkeypatch, handler)
    svc = qis.QueryIntentService()
    r = await svc.classify_from_ollama("hello")
    assert not r.used_fallback
    assert abs(sum(r.document_weights.values()) - 1.0) < 1e-6


@pytest.mark.asyncio
async def test_classify_from_ollama_http_error_falls_back(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    _patch_async_client(monkeypatch, handler)
    svc = qis.QueryIntentService()
    r = await svc.classify_from_ollama("hello")
    assert r.used_fallback
