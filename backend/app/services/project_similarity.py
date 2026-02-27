"""Project similarity service for Phase 3.5 Past Projects Enhancement.

Hybrid similarity: semantic (project-level embeddings) + metadata-based.
"""

import logging
from typing import List, Optional, Dict, Any

from app.config import settings
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore, get_vector_store
from app.services.project_metadata import ProjectMetadataService
from app.models.document import ProjectMetadata

logger = logging.getLogger(__name__)

# Max chars for project document aggregation (sentence-transformers often have 512 token limit)
MAX_PROJECT_DOCUMENT_CHARS = 8000

# Ordinal-like values for "adjacent" scoring (exact=1.0, adjacent=0.5)
COMPLEXITY_ORDER = ["Low", "Medium", "High"]
SIZE_ORDER = ["Small", "Medium", "Large"]
PROJECT_COMPLEXITY_ORDER = ["Simple", "Moderate", "Complex"]


def _jaccard(a: List[str], b: List[str]) -> float:
    """Jaccard similarity between two sets (from lists)."""
    sa, sb = set(a or []), set(b or [])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _ordinal_score(order: List[str], a: Optional[str], b: Optional[str]) -> float:
    """Exact match 1.0, adjacent in order 0.5, else 0."""
    if not a or not b:
        return 0.5 if (a or b) else 1.0
    a = a.strip()
    b = b.strip()
    if a == b:
        return 1.0
    try:
        i, j = order.index(a), order.index(b)
        return 0.5 if abs(i - j) == 1 else 0.0
    except ValueError:
        return 0.0


def _date_proximity_score(date_range_a: Optional[Dict[str, str]], date_range_b: Optional[Dict[str, str]]) -> float:
    """Simple date proximity: if both have end_date, closer years = higher score (0-1)."""
    if not date_range_a or not date_range_b:
        return 0.5
    end_a = (date_range_a.get("end_date") or date_range_a.get("start_date") or "")[:4]
    end_b = (date_range_b.get("end_date") or date_range_b.get("start_date") or "")[:4]
    if not end_a or not end_b or not end_a.isdigit() or not end_b.isdigit():
        return 0.5
    diff = abs(int(end_a) - int(end_b))
    if diff == 0:
        return 1.0
    if diff <= 1:
        return 0.7
    if diff <= 2:
        return 0.4
    return max(0.0, 0.3 - diff * 0.05)


class ProjectSimilarityService:
    """Hybrid project similarity: semantic + metadata."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        metadata_service: Optional[ProjectMetadataService] = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.metadata_service = metadata_service or ProjectMetadataService()
        self._project_embeddings: Dict[str, Any] = {}  # project_name -> vector (list)
        self._similarity_cache: Dict[str, List[Dict]] = {}  # cache_key -> top_k results
        self._top_k = getattr(settings, "SIMILAR_PROJECTS_COUNT", 3)
        self._semantic_weight = getattr(settings, "PROJECT_SIMILARITY_SEMANTIC_WEIGHT", 0.5)
        self._metadata_weight = getattr(settings, "PROJECT_SIMILARITY_METADATA_WEIGHT", 0.5)

    async def _get_project_embedding(self, project_name: str, vector_store: VectorStore) -> Optional[List[float]]:
        """Build aggregated project document and return its embedding (cached in memory)."""
        if project_name in self._project_embeddings:
            return self._project_embeddings[project_name]
        chunks = await vector_store.get_chunks_by_project(project_name)
        if not chunks:
            logger.debug(f"No chunks for project {project_name}, cannot compute semantic similarity")
            return None
        # Aggregate text (truncate to avoid token limit)
        parts = [c.get("text", "") for c in chunks]
        aggregated = "\n\n".join(parts)
        if len(aggregated) > MAX_PROJECT_DOCUMENT_CHARS:
            aggregated = aggregated[:MAX_PROJECT_DOCUMENT_CHARS] + "..."
        try:
            emb = self.embedding_service.generate_embedding(aggregated)
            vec = emb.tolist()
            self._project_embeddings[project_name] = vec
            return vec
        except Exception as e:
            logger.warning(f"Failed to embed project {project_name}: {e}")
            return None

    def _metadata_similarity(self, current: ProjectMetadata, other: ProjectMetadata) -> float:
        """Score 0-1 from metadata fields (weighted average)."""
        scores = []
        # CSG products: Jaccard
        scores.append(_jaccard(current.csg_products, other.csg_products))
        # Ordinal-like
        scores.append(_ordinal_score(COMPLEXITY_ORDER, current.integration_complexity, other.integration_complexity))
        scores.append(_ordinal_score(SIZE_ORDER, current.project_size, other.project_size))
        scores.append(_ordinal_score(PROJECT_COMPLEXITY_ORDER, current.project_complexity, other.project_complexity))
        # Exact match for role and client_type
        scores.append(1.0 if (current.csg_role and current.csg_role == other.csg_role) else 0.0)
        scores.append(1.0 if (current.client_type and current.client_type == other.client_type) else 0.0)
        # Date proximity
        scores.append(_date_proximity_score(current.date_range, other.date_range))
        return sum(scores) / len(scores) if scores else 0.5

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Cosine similarity (embeddings assumed normalized)."""
        if len(a) != len(b) or not a:
            return 0.0
        return sum(x * y for x, y in zip(a, b))

    async def find_similar_projects(
        self,
        project_name: str,
        top_k: Optional[int] = None,
        exclude_project_names: Optional[List[str]] = None,
        force_project: Optional[Dict[str, str]] = None,
        vector_store: Optional[VectorStore] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find projects most similar to the given project (hybrid semantic + metadata).
        Returns list of {"project_name", "customer", "similarity_score", "metadata": ProjectMetadata}.
        If force_project is provided (customer + project_name), that project is included as first result
        with score 1.0 (and optionally other similar projects after).
        """
        top_k = top_k or self._top_k
        exclude_project_names = exclude_project_names or []
        if force_project:
            force_name = force_project.get("project_name")
            force_customer = force_project.get("customer", "")
        else:
            force_name = force_customer = None

        vector_store = vector_store or await get_vector_store()
        current_meta = self.metadata_service.get_project_metadata(project_name)
        all_projects = self.metadata_service.get_all_projects()
        # Exclude current and any explicitly excluded
        candidates = [p for p in all_projects if p.project_name != project_name and p.project_name not in exclude_project_names]

        # If force_project specified, ensure that project is in results (and optionally still add similar)
        if force_name and force_name != project_name:
            forced = next((p for p in candidates if p.project_name == force_name), None)
            if forced:
                result = [{"project_name": forced.project_name, "customer": forced.customer, "similarity_score": 1.0, "metadata": forced}]
                candidates = [p for p in candidates if p.project_name != force_name]
            else:
                # Force project not in metadata DB: still add a stub so retrieval can target it
                result = [{"project_name": force_name, "customer": force_customer, "similarity_score": 1.0, "metadata": None}]
        else:
            result = []

        if not candidates:
            return result

        current_embedding = await self._get_project_embedding(project_name, vector_store)
        scores_list: List[tuple] = []

        for other in candidates:
            if current_meta:
                meta_score = self._metadata_similarity(current_meta, other)
            else:
                meta_score = 0.5
            if current_embedding:
                other_embedding = await self._get_project_embedding(other.project_name, vector_store)
                if other_embedding:
                    sem_score = self._cosine_similarity(current_embedding, other_embedding)
                    sem_score = max(0.0, min(1.0, sem_score))  # clamp
                else:
                    sem_score = 0.5
            else:
                sem_score = 0.5
            combined = self._semantic_weight * sem_score + self._metadata_weight * meta_score
            scores_list.append((combined, other))

        scores_list.sort(key=lambda x: -x[0])
        for score, proj in scores_list[: top_k - len(result)]:
            result.append({
                "project_name": proj.project_name,
                "customer": proj.customer,
                "similarity_score": round(score, 4),
                "metadata": proj,
            })

        return result
