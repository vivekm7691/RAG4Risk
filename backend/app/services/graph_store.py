"""Knowledge graph storage layer (Phase 1): GraphStore protocol and Neo4j implementation."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field
from neo4j import AsyncGraphDatabase, AsyncDriver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from app.config import settings
from app.models.graph import (
    GraphEdge,
    GraphNode,
    graph_id_chunk,
    graph_id_document,
    graph_id_statement_of_work,
)

logger = logging.getLogger(__name__)

# Prefixes for semantic / row nodes: {prefix}:{document_id}:...
_SEMANTIC_PREFIXES = (
    "req",
    "ctrl",
    "asm",
    "risk_sem",
    "prod",
    "del",
    "mls",
    "act",
    "svc",
    "chg",
    "cc",
    "role",
    "org",
    "risk",
    "issue",
)
_ROW_PREFIX_RE = re.compile(r"^(risk|issue):([^:]+):row:\d+$")

_graph_store_instance: Optional["GraphStore"] = None


class GraphNeighborhoodResult(BaseModel):
    """Chunk IDs discovered via graph expansion (Qdrant-ready, without ``chunk:`` prefix)."""

    chunk_ids: List[str] = Field(default_factory=list)
    node_ids: List[str] = Field(default_factory=list)
    paths_summary: List[str] = Field(default_factory=list)


class GraphTextSearchResult(BaseModel):
    """Seed node IDs from full-text search on graph node labels."""

    node_ids: List[str] = Field(default_factory=list)
    scores: List[float] = Field(default_factory=list)


@runtime_checkable
class GraphStore(Protocol):
    """Async graph persistence contract for ingest (Phase 2) and retrieval (Phase 3)."""

    async def ensure_schema(self) -> None: ...

    async def close(self) -> None: ...

    async def upsert_nodes(self, nodes: List[GraphNode]) -> None: ...

    async def upsert_edges(self, edges: List[GraphEdge]) -> None: ...

    async def delete_project_subgraph(self, project_name: str) -> int: ...

    async def delete_document_subgraph(self, document_id: str) -> int: ...

    async def neighborhood(
        self,
        chunk_ids: List[str],
        depth: int,
        limit: int,
        *,
        project_name: str,
    ) -> GraphNeighborhoodResult: ...

    async def text_search_seed(
        self,
        query: str,
        limit: int,
        *,
        project_name: str,
    ) -> GraphTextSearchResult: ...


def document_id_from_node_id(node_id: str) -> Optional[str]:
    """Derive document_id from stable graph node_id patterns (Phase 1 cascade helper)."""
    if node_id.startswith("document:"):
        return node_id.split(":", 1)[1]
    if node_id.startswith("sow:"):
        return node_id.split(":", 1)[1]
    m = _ROW_PREFIX_RE.match(node_id)
    if m:
        return m.group(2)
    parts = node_id.split(":")
    if len(parts) >= 3 and parts[0] in _SEMANTIC_PREFIXES:
        return parts[1]
    return None


def _chunk_graph_ids(chunk_ids: List[str]) -> List[str]:
    return [
        cid if cid.startswith("chunk:") else graph_id_chunk(cid)
        for cid in chunk_ids
        if cid
    ]


def _strip_chunk_prefix(graph_node_id: str) -> str:
    return graph_node_id.split(":", 1)[1] if graph_node_id.startswith("chunk:") else graph_node_id


def _node_props(node: GraphNode) -> Dict[str, Any]:
    props = dict(node.properties)
    props["label"] = node.label
    props["project_name"] = node.project_name
    props["node_type"] = node.node_type.value
    if node.confidence is not None:
        props["confidence"] = node.confidence
    doc_id = props.get("document_id") or document_id_from_node_id(node.node_id)
    if doc_id:
        props["document_id"] = doc_id
    return props


def _edge_props(edge: GraphEdge) -> Dict[str, Any]:
    props = dict(edge.properties)
    props["project_name"] = edge.project_name
    if edge.evidence_text:
        props["evidence_text"] = edge.evidence_text
    if edge.confidence is not None:
        props["confidence"] = edge.confidence
    return props


class DisabledGraphStore:
    """No-op store when ``GRAPH_ENABLED=false`` (vector-only mode)."""

    async def ensure_schema(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def upsert_nodes(self, nodes: List[GraphNode]) -> None:
        return None

    async def upsert_edges(self, edges: List[GraphEdge]) -> None:
        return None

    async def delete_project_subgraph(self, project_name: str) -> int:
        return 0

    async def delete_document_subgraph(self, document_id: str) -> int:
        return 0

    async def neighborhood(
        self,
        chunk_ids: List[str],
        depth: int,
        limit: int,
        *,
        project_name: str,
    ) -> GraphNeighborhoodResult:
        return GraphNeighborhoodResult()

    async def text_search_seed(
        self,
        query: str,
        limit: int,
        *,
        project_name: str,
    ) -> GraphTextSearchResult:
        return GraphTextSearchResult()


class Neo4jGraphStore:
    """Neo4j-backed GraphStore using ``:GraphNode`` base label plus type-specific labels."""

    FULLTEXT_INDEX = "graph_node_label_ft"

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver: Optional[AsyncDriver] = None
        self._schema_ready = False

    async def _get_driver(self) -> AsyncDriver:
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    async def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        driver = await self._get_driver()
        statements = [
            "CREATE CONSTRAINT graph_node_id IF NOT EXISTS "
            "FOR (n:GraphNode) REQUIRE n.node_id IS UNIQUE",
            "CREATE INDEX graph_node_project IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.project_name)",
            "CREATE INDEX graph_node_document IF NOT EXISTS "
            "FOR (n:GraphNode) ON (n.document_id)",
        ]
        async with driver.session(database=self._database) as session:
            for stmt in statements:
                await session.run(stmt)
            try:
                await session.run(
                    f"""
                    CREATE FULLTEXT INDEX {self.FULLTEXT_INDEX} IF NOT EXISTS
                    FOR (n:GraphNode) ON EACH [n.label]
                    """
                )
            except Neo4jError as exc:
                logger.warning("Could not create graph full-text index: %s", exc)
        self._schema_ready = True
        logger.info("Neo4j graph schema ensured")

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()
            self._driver = None
        self._schema_ready = False

    async def upsert_nodes(self, nodes: List[GraphNode]) -> None:
        if not nodes:
            return
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for node in nodes:
            label = node.node_type.value
            by_type[label].append({"node_id": node.node_id, "props": _node_props(node)})

        driver = await self._get_driver()
        async with driver.session(database=self._database) as session:
            for label, rows in by_type.items():
                # label comes only from NodeType enum values (safe for Cypher injection)
                query = f"""
                UNWIND $rows AS row
                MERGE (n:GraphNode:{label} {{node_id: row.node_id}})
                SET n += row.props
                """
                await session.run(query, rows=rows)

    async def upsert_edges(self, edges: List[GraphEdge]) -> None:
        if not edges:
            return
        by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for edge in edges:
            rel = edge.edge_type.value
            by_type[rel].append(
                {
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "props": _edge_props(edge),
                }
            )

        driver = await self._get_driver()
        async with driver.session(database=self._database) as session:
            for rel_type, rows in by_type.items():
                query = f"""
                UNWIND $rows AS row
                MATCH (s:GraphNode {{node_id: row.source_id}})
                MATCH (t:GraphNode {{node_id: row.target_id}})
                MERGE (s)-[r:{rel_type}]->(t)
                SET r += row.props
                """
                await session.run(query, rows=rows)

    async def delete_project_subgraph(self, project_name: str) -> int:
        driver = await self._get_driver()
        query = """
        MATCH (n:GraphNode {project_name: $project_name})
        WITH collect(n) AS nodes, count(n) AS deleted
        UNWIND nodes AS node
        DETACH DELETE node
        RETURN deleted
        """
        async with driver.session(database=self._database) as session:
            result = await session.run(query, project_name=project_name.strip())
            record = await result.single()
            return int(record["deleted"]) if record else 0

    async def delete_document_subgraph(self, document_id: str) -> int:
        doc_node = graph_id_document(document_id)
        sow_node = graph_id_statement_of_work(document_id)
        risk_prefix = f"risk:{document_id}:row:"
        issue_prefix = f"issue:{document_id}:row:"
        semantic_prefix = f":{document_id}:"

        driver = await self._get_driver()
        query = """
        MATCH (n:GraphNode)
        WHERE n.node_id = $doc_node
           OR n.node_id = $sow_node
           OR n.node_id STARTS WITH $risk_prefix
           OR n.node_id STARTS WITH $issue_prefix
           OR n.document_id = $document_id
           OR n.node_id CONTAINS $semantic_prefix
        WITH collect(DISTINCT n) AS nodes
        UNWIND nodes AS n
        DETACH DELETE n
        RETURN count(n) AS deleted
        """
        async with driver.session(database=self._database) as session:
            result = await session.run(
                query,
                doc_node=doc_node,
                sow_node=sow_node,
                risk_prefix=risk_prefix,
                issue_prefix=issue_prefix,
                document_id=document_id,
                semantic_prefix=semantic_prefix,
            )
            record = await result.single()
            return int(record["deleted"]) if record else 0

    async def neighborhood(
        self,
        chunk_ids: List[str],
        depth: int,
        limit: int,
        *,
        project_name: str,
    ) -> GraphNeighborhoodResult:
        seed_ids = _chunk_graph_ids(chunk_ids)
        if not seed_ids:
            return GraphNeighborhoodResult()

        depth = max(1, min(int(depth), settings.GRAPH_MAX_DEPTH))
        limit = max(1, int(limit))
        project_name = project_name.strip()

        driver = await self._get_driver()
        query = f"""
        MATCH (seed:GraphNode:Chunk)
        WHERE seed.node_id IN $seed_ids AND seed.project_name = $project_name
        MATCH path = (seed)-[*1..{depth}]-(neighbor:GraphNode)
        WHERE neighbor.project_name = $project_name
        WITH neighbor, path
        ORDER BY length(path)
        RETURN DISTINCT neighbor.node_id AS node_id, labels(neighbor) AS labels,
               [r IN relationships(path) | type(r)] AS rel_types
        LIMIT $limit
        """
        chunk_ids_out: List[str] = []
        node_ids: List[str] = []
        paths_summary: List[str] = []

        async with driver.session(database=self._database) as session:
            result = await session.run(
                query,
                seed_ids=seed_ids,
                project_name=project_name,
                limit=limit,
            )
            async for record in result:
                nid = record["node_id"]
                node_ids.append(nid)
                labels = record["labels"] or []
                if "Chunk" in labels:
                    chunk_ids_out.append(_strip_chunk_prefix(nid))
                rel_types = record["rel_types"] or []
                if rel_types:
                    paths_summary.append(f"…-{'-'.join(rel_types)}->{nid}")

        return GraphNeighborhoodResult(
            chunk_ids=chunk_ids_out,
            node_ids=node_ids,
            paths_summary=paths_summary[:20],
        )

    async def text_search_seed(
        self,
        query: str,
        limit: int,
        *,
        project_name: str,
    ) -> GraphTextSearchResult:
        q = (query or "").strip()
        if not q:
            return GraphTextSearchResult()

        limit = max(1, int(limit))
        project_name = project_name.strip()
        driver = await self._get_driver()
        cypher = f"""
        CALL db.index.fulltext.queryNodes('{self.FULLTEXT_INDEX}', $query)
        YIELD node, score
        WHERE node.project_name = $project_name
        RETURN node.node_id AS node_id, score
        ORDER BY score DESC
        LIMIT $limit
        """
        node_ids: List[str] = []
        scores: List[float] = []

        try:
            async with driver.session(database=self._database) as session:
                result = await session.run(
                    cypher,
                    query=q,
                    project_name=project_name,
                    limit=limit,
                )
                async for record in result:
                    node_ids.append(record["node_id"])
                    scores.append(float(record["score"]))
        except Neo4jError as exc:
            logger.warning("Graph full-text search unavailable, returning empty: %s", exc)

        return GraphTextSearchResult(node_ids=node_ids, scores=scores)


async def get_graph_store() -> GraphStore:
    """Singleton GraphStore: Neo4j when enabled, otherwise DisabledGraphStore."""
    global _graph_store_instance
    if _graph_store_instance is None:
        if settings.GRAPH_ENABLED:
            _graph_store_instance = Neo4jGraphStore(
                uri=settings.NEO4J_URI,
                user=settings.NEO4J_USER,
                password=settings.NEO4J_PASSWORD,
                database=settings.NEO4J_DATABASE,
            )
        else:
            _graph_store_instance = DisabledGraphStore()
    return _graph_store_instance


async def reset_graph_store() -> None:
    """Close and clear singleton (tests)."""
    global _graph_store_instance
    if _graph_store_instance is not None:
        await _graph_store_instance.close()
        _graph_store_instance = None


def is_graph_available() -> bool:
    """True when graph feature flag is on (Neo4j may still be unreachable)."""
    return settings.GRAPH_ENABLED


async def verify_graph_connection() -> bool:
    """Ping Neo4j; returns False when disabled or unreachable."""
    if not settings.GRAPH_ENABLED:
        return False
    store = await get_graph_store()
    if isinstance(store, DisabledGraphStore):
        return False
    try:
        driver = await store._get_driver()  # type: ignore[attr-defined]
        async with driver.session(database=settings.NEO4J_DATABASE) as session:
            await session.run("RETURN 1")
        return True
    except (ServiceUnavailable, Neo4jError, OSError) as exc:
        logger.warning("Neo4j connection check failed: %s", exc)
        return False
