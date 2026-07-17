# RAG4Risk knowledge graph ontology (Phase 0 + enterprise SOW)

Canonical contract for ingest, relation extraction, and Cypher queries. Implemented in [`app/models/graph.py`](app/models/graph.py).

**Version:** `0.2.0-enterprise-sow`

## Purpose

Complement Qdrant chunk retrieval with **explicit relationships** scoped by **`project_name`**, including delivery/SOW constructs (deliverables, milestones, roles) from your enterprise relationship table.

## StatementOfWork vs Document

| Concept | In RAG4Risk | Graph node |
|---------|-------------|------------|
| Uploaded `.docx` SOW file | `Document` (`document_type = statement of work`) | `document:{document_id}` |
| Logical SOW entity (table label) | Same file, semantic hub | `sow:{document_id}` (`StatementOfWork`) |

On SOW ingest (Phase 2): create **both** nodes and `REPRESENTS_SOW` (Document → StatementOfWork). Extractor edges that use **SOW** / **StatementOfWork** as source attach to the `StatementOfWork` node (or `Document` when `source_document_type` is `statement of work`).

## Node types

### Structural

| Type | ID pattern |
|------|------------|
| `Project` | `project:{project_name}` |
| `Document` | `document:{document_id}` |
| `Chunk` | `chunk:{chunk_id}` |
| `StatementOfWork` | `sow:{document_id}` |
| `Risk` / `Issue` (Excel) | `risk:{document_id}:row:{n}` / `issue:...` |

### Semantic (extracted)

| Type | ID prefix | Typical source doc |
|------|-----------|-------------------|
| `Requirement` | `req:` | SOW, solution, proposal |
| `Deliverable` | `del:` | SOW, proposal |
| `Milestone` | `mls:` | SOW |
| `Activity` | `act:` | SOW, solution |
| `Service` | `svc:` | SOW |
| `ChangeRequest` | `chg:` | SOW |
| `ChangeControl` | `cc:` | SOW |
| `Role` / `Organization` | `role:` / `org:` | SOW |
| `Assumption` / `Control` / `Risk` (text) | `asm:` / `ctrl:` / `risk_sem:` | solution, SOW |

## Enterprise relationships (from your table)

Mapped in `ALLOWED_EDGE_PAIRS`:

| Edge | Source → Target | Notes |
|------|-----------------|-------|
| `IS_PART_OF` | Deliverable, Milestone, Risk, ChangeRequest, Requirement, Assumption, StatementOfWork → Project | Inverse view of project scope |
| `HAS_RISK` | Project → Risk | Complements `IS_PART_OF` Risk→Project |
| `INCLUDES` | Project → Milestone, ChangeRequest | |
| `HAS_DEPENDENCY_ON` | Project → Project | Past-project linking (Phase 3.5+) |
| `DEFINES` | StatementOfWork → Deliverable, Milestone, Requirement, Assumption | Core SOW semantics |
| `HAS_DELIVERABLE` | StatementOfWork, Service → Deliverable | |
| `HAS_MILESTONE` | StatementOfWork → Milestone | |
| `REQUIRES_ACCEPTANCE_OF` | StatementOfWork → Deliverable | |
| `REFERENCES_REQUIREMENT` | Project, StatementOfWork → Requirement | |
| `IS_RESPONSIBLE_FOR` | Role, Organization → Deliverable | Table `RESPONSIBLE_FOR` merged here |
| `ASSOCIATED_WITH` | Milestone → Deliverable | |
| `PRODUCES` | Activity → Deliverable | |
| `COVERS` / `OUTLINES_OBJECTIVE_FOR` | StatementOfWork → Project | |
| `MANAGED_VIA` | StatementOfWork → ChangeControl | |
| `RECORDED_IN` | Risk → Document | Excel/tabular provenance (alias of `FROM_ROW` intent) |

Original MVP edges (`HAS_DOCUMENT`, `CONTAINS_CHUNK`, `FROM_ROW`, `MITIGATES`, etc.) unchanged.

## How to incorporate at runtime

1. **Ingest (deterministic):** Project, Document, Chunk, StatementOfWork (for SOW files), Excel Risk/Issue + `FROM_ROW` / `RECORDED_IN`.
2. **Extract (LLM, SOW chunks):** Emit Deliverable, Milestone, Role, … and edges in `SOW_EXTRACTABLE_EDGES`.
3. **Validate:** `validate_edge_endpoints(edge, src_type, tgt_type, source_document_type=...)`.
4. **Query:** Traverse e.g. `StatementOfWork -[:DEFINES]-> Deliverable -[:IS_PART_OF]-> Project`.

## Extraction policy

See `SEMANTIC_NODES_BY_DOCUMENT_TYPE` and `SOW_EXTRACTABLE_EDGES` in `graph.py`. SOW documents have the richest allowed node/edge set.

## Confidence

Persist LLM nodes/edges when `confidence >= 0.75`. Assumptions require `is_explicit: false` in properties.

## Next phases

- **Phase 1:** Neo4j `GraphStore` ✓
- **Phase 2:** Ingest hooks + relation extractor ✓ (`graph_sync_service`, `relation_extractor_service`)
- **Phase 3:** Graph-augmented retrieval (`graph_retrieval.py`, `use_graph_augmentation` on query API) ✓
- **Phase 4:** Observability + guardrails — enriched `graph_expansion` metadata, stage timings, Neo4j reachability degrade, per-seed degree cap, 504 on LLM timeout, frontend toggle ✓
- **Phase 5:** Offline recall@k eval harness (`eval/graph_eval_set.json`, `run_graph_eval.py`, `graph_eval_service.py`) ✓
