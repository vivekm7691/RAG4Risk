"""Phase 4: query route observability and LLM timeout handling."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from app.api.routes import query as query_routes
from app.models.query import QueryRequest


def test_is_ollama_timeout():
    assert query_routes._is_ollama_timeout(RuntimeError("Request to Ollama timed out after 3000 seconds"))
    assert not query_routes._is_ollama_timeout(RuntimeError("other error"))


@pytest.mark.asyncio
async def test_query_documents_returns_504_on_llm_timeout():
    request = QueryRequest(query="What is UAT?", project_name="P1", top_k=5)

    with patch.object(
        query_routes.embedding_service,
        "generate_embedding",
        return_value=MagicMock(tolist=lambda: [0.1, 0.2]),
    ), patch(
        "app.api.routes.query.get_vector_store",
        AsyncMock(
            return_value=MagicMock(
                search=AsyncMock(return_value=[{
                    "text": "UAT context",
                    "metadata": {"chunk_id": "c1", "project_name": "P1", "file_name": "sow.docx", "document_type": "statement of work"},
                }])
            )
        ),
    ), patch.object(
        query_routes.rag_service,
        "generate_response",
        AsyncMock(side_effect=RuntimeError("Request to Ollama timed out after 3000 seconds")),
    ):
        with pytest.raises(HTTPException) as exc_info:
            await query_routes.query_documents(request)

    assert exc_info.value.status_code == 504
