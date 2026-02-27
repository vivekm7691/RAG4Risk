# Phase 3.5: Past Projects Enhancement – Manual Test Guide

Use this guide to manually verify Phase 3.5 backend changes.  
**Base URL:** `http://localhost:8000` (ensure backend and Qdrant are running, e.g. `docker-compose up -d`).

---

## Prerequisites

- Backend running at `http://localhost:8000`
- Qdrant running (e.g. port 6333)
- At least one document already uploaded and indexed (for similarity to have data)
- Optional: Ollama running for full query/stream tests

---

## Step 1: Create or update project metadata

**Purpose:** Verify project metadata CRUD and storage.

**Request:**

```bash
curl -s -X POST "http://localhost:8000/api/projects/metadata" \
  -H "Content-Type: application/json" \
  -d "{
    \"project_name\": \"Test Project\",
    \"customer\": \"Acme Corp\",
    \"csg_products\": [\"CSG Singleview\", \"CSG Ascendon\"],
    \"csg_role\": \"Prime Contractor\",
    \"integration_complexity\": \"Medium\",
    \"client_type\": \"Enterprise\",
    \"project_size\": \"Large\",
    \"project_complexity\": \"Moderate\",
    \"date_range\": {\"start_date\": \"2024-01-01\", \"end_date\": \"2024-12-31\"}
  }"
```

**Expected:** HTTP 200, JSON with `project_name`, `customer`, and the fields you sent.  
**Check:** `project_name` and `customer` are present; optional fields match.

---

## Step 2: Get project metadata

**Purpose:** Verify GET by project name.

**Request:**

```bash
curl -s "http://localhost:8000/api/projects/Test%20Project/metadata"
```

**Expected:** HTTP 200, same metadata as in Step 1.  
**If 404:** Create metadata first (Step 1).

---

## Step 3: List all projects

**Purpose:** Verify list endpoint.

**Request:**

```bash
curl -s "http://localhost:8000/api/projects"
```

**Expected:** HTTP 200, JSON array of project metadata objects. At least the project from Step 1.

---

## Step 4: Get similar projects

**Purpose:** Verify hybrid similarity (semantic + metadata). Requires at least one project with chunks in Qdrant.

**Request:**

```bash
curl -s "http://localhost:8000/api/projects/Test%20Project/similar?top_k=3"
```

**Expected:** HTTP 200, JSON like:

```json
{
  "project_name": "Test Project",
  "similar_projects": [
    {
      "project_name": "...",
      "customer": "...",
      "similarity_score": 0.xxxx,
      "metadata": { ... }
    }
  ]
}
```

**Notes:**

- If only one project exists, `similar_projects` may be empty.
- To test properly, upload documents for **two or more projects** (different `project_name`), then run this again.

---

## Step 5: Retrieve preview (no LLM)

**Purpose:** Verify retrieval preview: similar projects + candidate chunks with `chunk_id`, no LLM call.

**Request:**

```bash
curl -s -X POST "http://localhost:8000/api/query/retrieve-preview" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"include_past_projects\": true,
    \"top_k\": 10
  }"
```

**Expected:** HTTP 200, JSON with:

- `similar_projects`: array of `{ project_name, customer, similarity_score, metadata }`
- `chunks`: array of `{ chunk_id, text, project_name, customer?, metadata, is_past_project }`
- `query`, `project_name`

**Checks:**

- No LLM is called (fast response).
- Each chunk has a `chunk_id`.
- Chunks from current project have `is_past_project: false`; from other projects `true`.

**Without past projects:**

```bash
curl -s -X POST "http://localhost:8000/api/query/retrieve-preview" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"What is the project about?\", \"project_name\": \"Test Project\", \"include_past_projects\": false}"
```

**Expected:** `similar_projects: []`, `chunks: []` (or only current-project chunks if your implementation returns them for this case).

---

## Step 6: Query with past projects (non-streaming)

**Purpose:** Verify full query flow with `include_past_projects`, and optional `exclude_chunk_ids` / `force_project`.

**6a. With past projects enabled (no exclusions)**

```bash
curl -s -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"top_k\": 10,
    \"include_past_projects\": true
  }"
```

**Expected:** HTTP 200, JSON with `answer`, `sources`, `query`, `project_name`, and optionally `similar_projects`.  
**Check:** `sources` entries include `chunk_id` and `is_past_project` (true/false).

**6b. With exclude_chunk_ids (drop specific chunks)**

1. Call **retrieve-preview** (Step 5) and note one `chunk_id` you want to exclude.
2. Then:

```bash
curl -s -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"top_k\": 10,
    \"include_past_projects\": true,
    \"exclude_chunk_ids\": [\"<paste-one-chunk_id-here>\"]
  }"
```

**Expected:** HTTP 200. The excluded chunk should **not** appear in `sources`.

**6c. With force_project (force a specific past project)**

If you have another project (e.g. "Other Project", customer "Beta Inc"):

```bash
curl -s -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"top_k\": 10,
    \"include_past_projects\": true,
    \"force_project\": { \"customer\": \"Beta Inc\", \"project_name\": \"Other Project\" }
  }"
```

**Expected:** HTTP 200. Past-project context should favor or include "Other Project" (and `similar_projects` may reflect that).

---

## Step 7: Query with past projects (streaming)

**Purpose:** Verify streaming endpoint with Phase 3.5 parameters.

**Request:**

```bash
curl -s -X POST "http://localhost:8000/api/query/stream" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"top_k\": 10,
    \"include_past_projects\": true
  }"
```

**Expected:**

- First SSE message: `type: "sources"` with `sources` (each with `chunk_id`, `is_past_project`) and optionally `similar_projects`.
- Then `type: "chunk"` messages with LLM text.
- Finally `type: "done"` with `total_time`.

**Check:** Sources in the first event include `is_past_project` and `chunk_id`.

---

## Step 8: Document upload with optional project_metadata

**Purpose:** Verify that upload can create/update project metadata via form field.

**Request (PowerShell – multipart):**

```powershell
$uri = "http://localhost:8000/api/documents/upload"
$projectMetadata = '{"customer":"Acme Corp","csg_products":["CSG Singleview"],"csg_role":"Prime Contractor"}'
# Use your test document path
$filePath = "C:\path\to\your\test_document.docx"
$form = @{
    file = Get-Item -Path $filePath
    project_name = "Test Project"
    document_type = "statement of work"
    project_metadata = $projectMetadata
}
Invoke-RestMethod -Uri $uri -Method Post -Form $form
```

**Using curl (multipart):**

```bash
curl -s -X POST "http://localhost:8000/api/documents/upload" \
  -F "file=@/path/to/test_document.docx" \
  -F "project_name=Test Project" \
  -F "document_type=statement of work" \
  -F "project_metadata={\"customer\":\"Acme Corp\",\"csg_products\":[\"CSG Singleview\"]}"
```

**Expected:** HTTP 201, document processed and chunks added.  
**Check:** Then call **Step 2** (GET project metadata) for "Test Project" – metadata should exist or be updated with the provided customer and optional fields.

---

## Step 9: Context weighting (optional)

**Purpose:** Verify optional 70/30 (or custom) context weighting.

**Request:**

```bash
curl -s -X POST "http://localhost:8000/api/query/retrieve-preview" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"What is the project about?\",
    \"project_name\": \"Test Project\",
    \"include_past_projects\": true,
    \"top_k\": 10,
    \"context_weighting\": {
      \"current_project_weight\": 0.8,
      \"past_projects_weight\": 0.2
    }
  }"
```

**Expected:** HTTP 200. Chunk counts should reflect roughly 80% from current project and 20% from past projects (when multiple projects exist).

---

## Quick checklist

| Step | What to verify |
|------|----------------|
| 1 | POST project metadata returns 200 and stored fields |
| 2 | GET project metadata returns same data |
| 3 | GET list projects returns array including created project |
| 4 | GET similar projects returns list with scores (if ≥2 projects) |
| 5 | Retrieve-preview returns similar_projects + chunks with chunk_id, no LLM |
| 6a | Query with include_past_projects returns answer + sources with is_past_project |
| 6b | Query with exclude_chunk_ids omits those chunks from sources |
| 6c | Query with force_project uses that past project in context |
| 7 | Stream response includes sources (with chunk_id, is_past_project) then chunks then done |
| 8 | Upload with project_metadata creates/updates project metadata |
| 9 | Context weighting changes proportion of current vs past chunks in preview |

---

## Troubleshooting

- **404 on /api/projects:** Ensure the projects router is registered in `main.py` and the backend was restarted.
- **Empty similar_projects:** Add at least two projects with documents (different `project_name`), then add metadata for both (Step 1) and retry Step 4 / 5.
- **SQLite project_metadata.db:** Stored under `backend/data/` by default (or path in `PROJECT_METADATA_DB_PATH`). Ensure the process has write permission.
- **Retrieve-preview returns empty chunks:** Ensure the vector store has chunks for the given `project_name` (upload a document first).
