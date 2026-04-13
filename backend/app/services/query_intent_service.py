"""Phase 3.75: query intent — document-type weights from a lightweight LLM (JSON).

Task 1: prompts, parsing, normalization, fallbacks. Task 2: Ollama /api/chat. Task 3: allocate_top_k_by_weights for per-type retrieval.
"""

from __future__ import annotations

import json
import logging
import math
import re
from typing import Any, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from app.config import settings
from app.models.document import DocumentType

logger = logging.getLogger(__name__)

# All payload values used in Qdrant / upload (must match DocumentType enum)
DOCUMENT_TYPE_VALUES: List[str] = [e.value for e in DocumentType]


class QueryIntentResult(BaseModel):
    """Validated intent output consumed by retrieval (Phase 3.75+)."""

    intent_summary: str = Field(..., description="Short description of what the user is asking")
    document_weights: Dict[str, float] = Field(
        ...,
        description="Non-negative weights summing to 1.0 over relevant document types",
    )
    priority_order: Optional[List[str]] = Field(
        None,
        description="Optional tie-break order (higher priority first)",
    )
    used_fallback: bool = Field(False, description="True if JSON parse failed and weights were inferred")


class _RawIntentJson(BaseModel):
    """Shape expected inside the LLM JSON object."""

    intent_summary: str = ""
    document_weights: Dict[str, Any] = Field(default_factory=dict)
    priority_order: Optional[List[str]] = None

    @field_validator("document_weights", mode="before")
    @classmethod
    def coerce_weights(cls, v: Any) -> Dict[str, Any]:
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("document_weights must be an object")
        return v


_INTENT_SYSTEM_PROMPT = """You are a retrieval planner for a RAG system over project documents.

Document types (use these exact strings as keys in document_weights):
1. "statement of work" — roles, responsibilities, timelines, fees, scope.
2. "solution description document" — technical solution, integrations, technical assumptions.
3. "proposal document" — pre-sales proposal to the customer.
4. "risk register" — tabular / structured project risks.
5. "issue log" — tabular / structured issues.

Your job: infer the user's intent and assign non-negative weights to EVERY type above so they sum to 1.0.
Use near-zero weight for risk register or issue log unless the question clearly concerns risks/issues in tabular form.

Respond with ONLY a single JSON object (no markdown fences, no commentary) with this exact shape:
{"intent_summary": "<1-2 sentences>", "document_weights": {"statement of work": <float>, "solution description document": <float>, "proposal document": <float>, "risk register": <float>, "issue log": <float>}, "priority_order": ["<type>", "..."] }

priority_order is optional; if present, list types from most to least important for this query."""


def build_intent_user_message(user_query: str) -> str:
    return f'User question (analyze intent and output JSON only):\n"""{user_query.strip()}"""'


def build_ollama_chat_messages(user_query: str) -> List[Dict[str, str]]:
    """Messages for Ollama /api/chat."""
    return [
        {"role": "system", "content": _INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": build_intent_user_message(user_query)},
    ]


def resolve_intent_ollama_base_url(explicit: Optional[str] = None) -> str:
    """Intent server URL; explicit wins, else INTENT_LLM_BASE_URL if set, else OLLAMA_BASE_URL."""
    if explicit and explicit.strip():
        return explicit.rstrip("/")
    configured = (settings.INTENT_LLM_BASE_URL or "").strip()
    if configured:
        return configured.rstrip("/")
    return settings.OLLAMA_BASE_URL.rstrip("/")


async def ollama_intent_chat_completion(
    user_query: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    timeout: Optional[float] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Non-streaming Ollama /api/chat for JSON intent output.

    Raises:
        RuntimeError: empty response, unexpected JSON shape, or HTTP/network errors (mirrors RAGService).
    """
    bu = resolve_intent_ollama_base_url(base_url)
    m = model or settings.INTENT_LLM_MODEL
    to = settings.INTENT_LLM_TIMEOUT_SECONDS if timeout is None else timeout
    mt = settings.INTENT_LLM_MAX_TOKENS if max_tokens is None else max_tokens
    temp = settings.INTENT_LLM_TEMPERATURE if temperature is None else temperature

    url = f"{bu}/api/chat"
    payload: Dict[str, Any] = {
        "model": m,
        "messages": build_ollama_chat_messages(user_query),
        "stream": False,
        "options": {
            "num_predict": mt,
            "temperature": temp,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=to) as client:
            response = await client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Intent LLM request to Ollama timed out after {to} seconds (base_url={bu}, model={m})"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(
            f"Intent LLM Ollama error: {e.response.status_code} - {e.response.text}"
        )
    except httpx.RequestError as e:
        raise RuntimeError(f"Failed to connect to intent Ollama at {bu}: {e!s}")

    msg = data.get("message")
    if not isinstance(msg, dict):
        raise RuntimeError("Intent LLM: unexpected Ollama response (missing message object)")

    content = (msg.get("content") or "").strip()
    thinking = (msg.get("thinking") or "").strip()

    logger.debug("Intent LLM raw — content (%d chars): %s", len(content), content[:500] if content else "<empty>")
    logger.debug("Intent LLM raw — thinking (%d chars): %s", len(thinking), thinking[:500] if thinking else "<empty>")

    # Reasoning models (DeepSeek R1) may put everything in `thinking` and leave
    # `content` empty. Try content first, then thinking, then combine both.
    if content:
        return content
    if thinking:
        logger.info("Intent LLM: content was empty; using thinking field for JSON extraction")
        return thinking
    raise RuntimeError("Intent LLM: Ollama returned empty assistant content (both content and thinking)")


def strip_reasoning_traces(text: str) -> str:
    """Remove common reasoning/thinking wrappers so JSON can be found."""
    if not text:
        return text
    # Strip model reasoning blocks (e.g. DeepSeek-R1) before JSON extraction
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def extract_json_object(text: str) -> Optional[str]:
    """Extract a single JSON object from model output (fenced or raw)."""
    text = strip_reasoning_traces(text)
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
            continue
        if ch in "\"'":
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def normalize_weights_all_types(weights: Dict[str, float]) -> Dict[str, float]:
    """Clamp to known types, non-negative, sum to 1.0; uniform if empty."""
    out: Dict[str, float] = {}
    for k in DOCUMENT_TYPE_VALUES:
        v = weights.get(k, 0.0)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = 0.0
        out[k] = max(0.0, fv)
    total = sum(out.values())
    if total <= 0:
        u = 1.0 / len(DOCUMENT_TYPE_VALUES)
        return {k: u for k in DOCUMENT_TYPE_VALUES}
    return {k: out[k] / total for k in DOCUMENT_TYPE_VALUES}


def mask_weights_to_project_types(
    weights: Dict[str, float],
    document_types_in_project: Optional[List[str]],
) -> Dict[str, float]:
    """
    Zero out types not present in the project, then renormalize to sum 1.0.
    If nothing left or list is empty, return uniform over all types (caller may treat as global).
    """
    if not document_types_in_project:
        return dict(weights)
    allowed = set(document_types_in_project)
    filtered = {k: weights.get(k, 0.0) for k in DOCUMENT_TYPE_VALUES if k in allowed}
    total = sum(filtered.values())
    if total <= 0:
        u = 1.0 / len(document_types_in_project)
        return {k: u for k in document_types_in_project if k in DOCUMENT_TYPE_VALUES}
    return {k: filtered[k] / total for k in filtered}


def sanitize_priority_order(
    order: Optional[List[str]],
    document_types_in_project: Optional[List[str]],
) -> Optional[List[str]]:
    if not order:
        return None
    seen = set()
    out: List[str] = []
    allowed = set(document_types_in_project) if document_types_in_project else set(DOCUMENT_TYPE_VALUES)
    for t in order:
        if t in DOCUMENT_TYPE_VALUES and t in allowed and t not in seen:
            seen.add(t)
            out.append(t)
    return out or None


def allocate_top_k_by_weights(weights: Dict[str, float], total_k: int) -> Dict[str, int]:
    """
    Split total_k retrieval slots across document types (largest remainder).

    Uses types with positive weight after coercion; renormalizes if needed.
    Returns counts per type (including zeros for known keys in ``weights`` when applicable).
    """
    if total_k <= 0:
        return {}

    def _f(v: Any) -> float:
        try:
            return max(0.0, float(v))
        except (TypeError, ValueError):
            return 0.0

    positive = [t for t, v in weights.items() if _f(v) > 0]
    if not positive:
        positive = list(DOCUMENT_TYPE_VALUES)
        norm: Dict[str, float] = {t: 1.0 / len(positive) for t in positive}
    else:
        s = sum(_f(weights[t]) for t in positive)
        if s <= 0:
            norm = {t: 1.0 / len(positive) for t in positive}
        else:
            norm = {t: _f(weights[t]) / s for t in positive}

    labels = list(norm.keys())
    raw = {t: total_k * norm[t] for t in labels}
    floors = {t: int(math.floor(raw[t] + 1e-12)) for t in labels}
    rem = total_k - sum(floors.values())
    fracs = sorted(labels, key=lambda t: (raw[t] - floors[t], t), reverse=True)
    out: Dict[str, int] = {t: floors[t] for t in labels}
    for i in range(max(0, rem)):
        out[fracs[i]] += 1

    for t in weights:
        if t not in out:
            out[t] = 0
    return out


class QueryIntentService:
    """Parse and validate intent JSON; apply project-type masking and fallbacks."""

    async def classify_from_ollama(
        self,
        user_query: str,
        document_types_in_project: Optional[List[str]] = None,
        *,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> QueryIntentResult:
        """
        Call intent model via Ollama /api/chat, then parse into QueryIntentResult.

        On HTTP/network/timeout errors or empty model output, returns uniform fallback weights.
        """
        try:
            raw = await ollama_intent_chat_completion(
                user_query,
                base_url=base_url,
                model=model,
                timeout=timeout,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            logger.warning("Intent Ollama call failed: %s", e)
            return self._fallback_uniform(document_types_in_project)
        return self.process_llm_output(raw, document_types_in_project)

    def process_llm_output(
        self,
        raw_llm_text: str,
        document_types_in_project: Optional[List[str]] = None,
    ) -> QueryIntentResult:
        """
        Parse model output into QueryIntentResult.

        document_types_in_project: distinct document_type values that exist in Qdrant for this project
        (from a lightweight scan). If None/empty, weights apply to all five types without masking.
        """
        blob = extract_json_object(raw_llm_text or "")
        if not blob:
            logger.warning("Intent: no JSON object found in model output; using fallback weights")
            return self._fallback_uniform(document_types_in_project)

        try:
            data = json.loads(blob)
        except json.JSONDecodeError as e:
            logger.warning("Intent: JSON decode failed: %s; using fallback", e)
            return self._fallback_uniform(document_types_in_project)

        try:
            raw = _RawIntentJson.model_validate(data)
        except Exception as e:
            logger.warning("Intent: Pydantic validation failed: %s; using fallback", e)
            return self._fallback_uniform(document_types_in_project)

        # Coerce string numbers in weights
        coerced: Dict[str, float] = {}
        for k, v in raw.document_weights.items():
            if k not in DOCUMENT_TYPE_VALUES:
                continue
            try:
                coerced[k] = float(v)
            except (TypeError, ValueError):
                coerced[k] = 0.0

        normalized = normalize_weights_all_types(coerced)
        masked = mask_weights_to_project_types(normalized, document_types_in_project)
        priority = sanitize_priority_order(raw.priority_order, document_types_in_project)

        summary = (raw.intent_summary or "").strip() or "Intent parsed from model JSON."

        return QueryIntentResult(
            intent_summary=summary,
            document_weights=masked,
            priority_order=priority,
            used_fallback=False,
        )

    def _fallback_uniform(self, document_types_in_project: Optional[List[str]]) -> QueryIntentResult:
        types = [t for t in (document_types_in_project or []) if t in DOCUMENT_TYPE_VALUES]
        if not types:
            types = list(DOCUMENT_TYPE_VALUES)
        u = 1.0 / len(types)
        return QueryIntentResult(
            intent_summary="Uniform document-type weights (fallback: model output was missing or invalid JSON).",
            document_weights={t: u for t in types},
            priority_order=list(types),
            used_fallback=True,
        )
