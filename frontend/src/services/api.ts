/**
 * API client service for RAG4Risk frontend
 */

import axios, { isAxiosError, type AxiosProgressEvent } from 'axios';

/**
 * In Vite dev, default to same-origin + `/api` proxy (see vite.config.ts) so the browser
 * does not depend on reaching localhost:8000 directly (CORS/firewall quirks).
 * Set VITE_API_BASE_URL to override (e.g. http://192.168.1.10:8000).
 */
function resolveApiBaseUrl(): string {
  const fromEnv = import.meta.env?.VITE_API_BASE_URL as string | undefined;
  if (fromEnv != null && String(fromEnv).trim() !== '') {
    return String(fromEnv).replace(/\/$/, '');
  }
  if (import.meta.env.DEV) {
    return '';
  }
  return 'http://localhost:8000';
}

const API_BASE_URL = resolveApiBaseUrl();

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

apiClient.interceptors.request.use(
  (config) => config,
  (error) => Promise.reject(error)
);

apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      console.error('Unauthorized access');
    }
    return Promise.reject(error);
  }
);

// --- Types (Phase 3.5) ---

export interface ContextWeighting {
  current_project_weight: number;
  past_projects_weight: number;
}

export interface ForceProjectSelection {
  customer: string;
  project_name: string;
}

export interface PastProjectIntentSlots {
  project_name: string;
  budget: number;
  slots: Record<string, number>;
}

/** Phase 3.75: returned when the server ran intent classification (see QUERY_INTENT_ENABLED). */
export interface QueryIntentInfo {
  intent_summary: string;
  document_weights: Record<string, number>;
  priority_order?: string[] | null;
  used_fallback: boolean;
  n_current_slots: number;
  n_past_slots: number;
  slots_current_project: Record<string, number>;
  slots_past_by_project: PastProjectIntentSlots[];
}

export interface RetrievePreviewRequest {
  query: string;
  project_name?: string | null;
  top_k?: number;
  include_past_projects: true;
  context_weighting?: ContextWeighting;
  exclude_chunk_ids?: string[];
  force_project?: ForceProjectSelection;
  /** When true and server enables intent, weighted per-type retrieval (requires project_name). */
  use_query_intent?: boolean;
}

export interface SimilarProjectPreview {
  project_name: string;
  customer?: string | null;
  similarity_score: number;
  metadata?: Record<string, unknown> | null;
}

export interface PreviewChunk {
  chunk_id: string;
  text: string;
  project_name: string;
  customer?: string | null;
  metadata?: Record<string, unknown> | null;
  is_past_project: boolean;
}

export interface RetrievalPreviewResponse {
  similar_projects: SimilarProjectPreview[];
  chunks: PreviewChunk[];
  query: string;
  project_name?: string | null;
  query_intent?: QueryIntentInfo | null;
}

export interface SourceCitation {
  document_id: string;
  document_name: string;
  chunk_id: string;
  project_name: string;
  document_type: string;
  relevance_score?: number | null;
  is_past_project: boolean;
  row_number?: number | null;
  sheet_name?: string | null;
}

export interface QueryStreamRequest {
  query: string;
  project_name?: string | null;
  top_k?: number;
  model?: string | null;
  include_past_projects?: boolean;
  context_weighting?: ContextWeighting;
  exclude_chunk_ids?: string[];
  force_project?: ForceProjectSelection;
  /** When true and server enables intent, weighted per-type retrieval (requires project_name). */
  use_query_intent?: boolean;
}

export interface ProjectMetadata {
  project_name: string;
  customer: string;
  csg_products?: string[];
  csg_role?: string | null;
  integration_complexity?: string | null;
  client_type?: string | null;
  project_size?: string | null;
  project_complexity?: string | null;
  date_range?: Record<string, string> | null;
  created_at?: string | null;
  updated_at?: string | null;
}

/** Document row from GET /api/documents (subset used by UI). */
export interface DocumentListItem {
  project_name: string;
  document_id: string;
  file_name?: string;
}

/** Full document metadata from GET /api/documents (unique docs from vector store). */
export interface DocumentMetadataRecord {
  document_id: string;
  project_name: string;
  document_type: string;
  title?: string | null;
  author?: string | null;
  upload_date: string;
  file_name: string;
  file_size: number;
}

/** Backend `DocumentType` values for upload forms. */
export const DOCUMENT_TYPE_OPTIONS: { value: string; label: string }[] = [
  { value: 'statement of work', label: 'Statement of work' },
  { value: 'solution description document', label: 'Solution description' },
  { value: 'proposal document', label: 'Proposal' },
  { value: 'risk register', label: 'Risk register (.xlsx)' },
  { value: 'issue log', label: 'Issue log (.xlsx)' },
];

function formatAxiosError(scope: string, e: unknown): string {
  if (isAxiosError(e)) {
    const st = e.response?.status;
    const data = e.response?.data;
    const detail =
      data && typeof data === 'object' && 'detail' in data
        ? JSON.stringify((data as { detail: unknown }).detail)
        : data
          ? JSON.stringify(data)
          : '';
    if (st) {
      return `${scope}: HTTP ${st}${detail ? ` — ${detail}` : ''}`;
    }
    return `${scope}: ${e.message || 'network error'} (is the API running?)`;
  }
  return `${scope}: ${String(e)}`;
}

export type ProjectNamesLoadResult = {
  names: string[];
  /** Set when one or both API calls failed (names may still be partial). */
  error?: string;
};

/**
 * Distinct project names from uploaded documents (Qdrant via GET /api/documents) and
 * project metadata (SQLite via GET /api/projects). Empty list usually means no uploads
 * yet and no metadata rows — not necessarily an error.
 */
export async function listDistinctProjectNames(): Promise<ProjectNamesLoadResult> {
  const names = new Set<string>();
  const errors: string[] = [];
  try {
    const res = await apiClient.get<DocumentListItem[]>('/api/documents');
    for (const d of res.data ?? []) {
      const n = d.project_name?.trim();
      if (n) names.add(n);
    }
  } catch (e) {
    errors.push(formatAxiosError('GET /api/documents', e));
    console.warn('[listDistinctProjectNames] documents list failed', e);
  }
  try {
    const res = await apiClient.get<ProjectMetadata[]>('/api/projects');
    for (const p of res.data ?? []) {
      const n = p.project_name?.trim();
      if (n) names.add(n);
    }
  } catch (e) {
    errors.push(formatAxiosError('GET /api/projects', e));
    console.warn('[listDistinctProjectNames] projects list failed', e);
  }
  return {
    names: Array.from(names).sort((a, b) => a.localeCompare(b, undefined, { sensitivity: 'base' })),
    error: errors.length ? errors.join(' · ') : undefined,
  };
}

export const documentAPI = {
  upload: async (
    file: File,
    projectName: string,
    documentType: string,
    options?: { onUploadProgress?: (e: AxiosProgressEvent) => void }
  ) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('project_name', projectName);
    formData.append('document_type', documentType);

    return apiClient.post('/api/documents/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: options?.onUploadProgress,
    });
  },

  list: async (projectName?: string): Promise<DocumentMetadataRecord[]> => {
    const res = await apiClient.get<DocumentMetadataRecord[]>('/api/documents', {
      params: projectName?.trim() ? { project_name: projectName.trim() } : {},
    });
    return res.data ?? [];
  },

  get: async (documentId: string) => {
    return apiClient.get(`/api/documents/${documentId}`);
  },

  delete: async (documentId: string) => {
    return apiClient.delete(`/api/documents/${documentId}`);
  },
};

export const queryAPI = {
  query: async (
    query: string,
    projectName?: string,
    topK?: number,
    extra?: Partial<QueryStreamRequest>
  ) => {
    return apiClient.post('/api/query', {
      query,
      project_name: projectName,
      top_k: topK,
      ...extra,
    });
  },

  retrievePreview: async (body: RetrievePreviewRequest) => {
    const res = await apiClient.post<RetrievalPreviewResponse>(
      '/api/query/retrieve-preview',
      body
    );
    return res.data;
  },
};

export const projectsAPI = {
  list: async () => {
    const res = await apiClient.get<ProjectMetadata[]>('/api/projects');
    return res.data;
  },
};

export interface OllamaModelInfo {
  name: string;
  size?: number;
  modified_at?: string | null;
}

/** List models from Ollama via backend diagnostics (for UI suggestions). */
export async function listOllamaModels(): Promise<string[]> {
  try {
    const res = await apiClient.get<{
      success?: boolean;
      models?: OllamaModelInfo[];
    }>('/api/diagnostics/ollama-models');
    const models = res.data?.models ?? [];
    return models.map((m) => m.name).filter(Boolean);
  } catch {
    return [];
  }
}

/** Base URL for fetch (SSE) — same as axios base */
export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

export interface StreamQueryCallbacks {
  onSources?: (sources: SourceCitation[], similarProjects?: SimilarProjectPreview[]) => void;
  onChunk: (text: string) => void;
  onDone?: (totalTime?: number, timings?: Record<string, number | null | undefined>) => void;
  onError?: (message: string) => void;
}

/**
 * POST /api/query/stream with SSE parsing (no LLM on server until this runs).
 */
export async function streamQuery(
  body: QueryStreamRequest,
  callbacks: StreamQueryCallbacks
): Promise<void> {
  const url = `${API_BASE_URL}/api/query/stream`;
  const res = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'text/event-stream',
    },
    body: JSON.stringify({
      query: body.query,
      project_name: body.project_name ?? undefined,
      top_k: body.top_k ?? 5,
      model: body.model ?? undefined,
      include_past_projects: body.include_past_projects ?? false,
      context_weighting: body.context_weighting,
      exclude_chunk_ids:
        body.exclude_chunk_ids && body.exclude_chunk_ids.length > 0
          ? body.exclude_chunk_ids
          : undefined,
      force_project: body.force_project,
      use_query_intent: body.use_query_intent ?? false,
    }),
  });

  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const j = await res.json();
      detail = (j as { detail?: string }).detail || JSON.stringify(j);
    } catch {
      detail = await res.text();
    }
    callbacks.onError?.(detail);
    return;
  }

  const reader = res.body?.getReader();
  if (!reader) {
    callbacks.onError?.('No response body');
    return;
  }

  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (!raw) continue;
        let data: Record<string, unknown>;
        try {
          data = JSON.parse(raw) as Record<string, unknown>;
        } catch {
          continue;
        }
        const type = data.type as string;
        if (type === 'sources') {
          const sources = (data.sources as SourceCitation[]) ?? [];
          const raw = data.similar_projects;
          let similarProjects: SimilarProjectPreview[] | undefined;
          if (Array.isArray(raw) && raw.length > 0) {
            similarProjects = raw.map((item: unknown) => {
              if (item && typeof item === 'object' && 'project_name' in item) {
                const o = item as Record<string, unknown>;
                return {
                  project_name: String(o.project_name ?? ''),
                  customer: o.customer != null ? String(o.customer) : null,
                  similarity_score: Number(o.similarity_score ?? 0),
                  metadata: (o.metadata as Record<string, unknown> | null) ?? null,
                };
              }
              if (typeof item === 'string') {
                return {
                  project_name: item,
                  customer: null,
                  similarity_score: 0,
                  metadata: null,
                };
              }
              return {
                project_name: '',
                customer: null,
                similarity_score: 0,
                metadata: null,
              };
            });
          }
          callbacks.onSources?.(sources, similarProjects);
        } else if (type === 'chunk') {
          callbacks.onChunk(String(data.text ?? ''));
        } else if (type === 'done') {
          callbacks.onDone?.(
            typeof data.total_time === 'number' ? data.total_time : undefined,
            data.timings as Record<string, number | null | undefined> | undefined
          );
        } else if (type === 'error') {
          callbacks.onError?.(String(data.message ?? 'Stream error'));
          return;
        }
      }
    }
  } catch (e) {
    callbacks.onError?.(e instanceof Error ? e.message : String(e));
  }
}

export default apiClient;
