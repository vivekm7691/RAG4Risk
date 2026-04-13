import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  listDistinctProjectNames,
  listOllamaModels,
  projectsAPI,
  queryAPI,
  streamQuery,
  type ContextWeighting,
  type ForceProjectSelection,
  type PreviewChunk,
  type ProjectMetadata,
  type RetrievalPreviewResponse,
  type SimilarProjectPreview,
  type SourceCitation,
} from '../services/api';
import './ChatInterface.css';

const FORCE_SEP = '\u001f'; // unit separator — unlikely in project/customer names

function parseForceValue(v: string): ForceProjectSelection | undefined {
  if (!v) return undefined;
  const i = v.indexOf(FORCE_SEP);
  if (i < 0) return undefined;
  return {
    customer: v.slice(0, i),
    project_name: v.slice(i + 1),
  };
}

function forceOptionValue(p: ProjectMetadata): string {
  return `${p.customer}${FORCE_SEP}${p.project_name}`;
}

export function ChatInterface() {
  const [projectName, setProjectName] = useState('');
  const [queryText, setQueryText] = useState('');
  const [topK, setTopK] = useState(10);
  const [includePastProjects, setIncludePastProjects] = useState(false);
  /** Phase 3.75: send use_query_intent when a project is selected (server needs QUERY_INTENT_ENABLED). */
  const [useQueryIntent, setUseQueryIntent] = useState(false);
  const [currentWeight, setCurrentWeight] = useState(0.7);
  const [pastWeight, setPastWeight] = useState(0.3);
  const [forceSelect, setForceSelect] = useState('');
  /** Ollama model for the final LLM call (empty = backend default from env). */
  const [ollamaModel, setOllamaModel] = useState('');
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [projectOptions, setProjectOptions] = useState<string[]>([]);
  /** Shown when /api/documents or /api/projects fails (dropdown may be empty or partial). */
  const [projectListError, setProjectListError] = useState<string | null>(null);
  const [projects, setProjects] = useState<ProjectMetadata[]>([]);
  const [preview, setPreview] = useState<RetrievalPreviewResponse | null>(null);
  /** Chunk IDs the user has excluded from context */
  const [excludedChunkIds, setExcludedChunkIds] = useState<Set<string>>(new Set());
  const [messages, setMessages] = useState<
    Array<{
      role: 'user' | 'assistant';
      content: string;
      sources?: SourceCitation[];
      similarProjectsFromStream?: SimilarProjectPreview[];
      timings?: Record<string, number | null | undefined>;
      totalTime?: number;
    }>
  >([]);
  const [loadingPreview, setLoadingPreview] = useState(false);
  const [loadingStream, setLoadingStream] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Index of assistant message being streamed (set in same tick as messages append). */
  const streamingAssistantIndexRef = useRef<number | null>(null);

  const refreshOllamaModels = useCallback(() => {
    void listOllamaModels().then(setOllamaModels);
  }, []);

  const refreshProjectNames = useCallback(() => {
    void listDistinctProjectNames().then(({ names, error }) => {
      setProjectOptions(names);
      setProjectListError(error ?? null);
    });
  }, []);

  useEffect(() => {
    projectsAPI
      .list()
      .then(setProjects)
      .catch(() => setProjects([]));
    refreshOllamaModels();
    refreshProjectNames();
  }, [refreshOllamaModels, refreshProjectNames]);

  /** Dropdown options: known projects plus current selection if not in list */
  const projectSelectOptions = useMemo(() => {
    if (!projectName.trim()) return projectOptions;
    if (projectOptions.includes(projectName)) return projectOptions;
    return [...projectOptions, projectName].sort((a, b) =>
      a.localeCompare(b, undefined, { sensitivity: 'base' })
    );
  }, [projectOptions, projectName]);

  /** Ollama dropdown: include current model if user had a value not in refreshed list */
  const ollamaSelectOptions = useMemo(() => {
    const m = ollamaModel.trim();
    if (!m) return ollamaModels;
    if (ollamaModels.includes(m)) return ollamaModels;
    return [...ollamaModels, m].sort((a, b) => a.localeCompare(b));
  }, [ollamaModels, ollamaModel]);

  const contextWeighting: ContextWeighting | undefined =
    includePastProjects
      ? {
          current_project_weight: currentWeight,
          past_projects_weight: pastWeight,
        }
      : undefined;

  const forceProject = parseForceValue(forceSelect);

  const intentActive = useQueryIntent && Boolean(projectName.trim());

  const runRetrievalPreview = useCallback(async () => {
    setError(null);
    if (!projectName.trim()) {
      setError('Project name is required when including similar past projects.');
      return;
    }
    setLoadingPreview(true);
    try {
      const data = await queryAPI.retrievePreview({
        query: queryText.trim(),
        project_name: projectName.trim(),
        top_k: topK,
        include_past_projects: true,
        context_weighting: contextWeighting,
        force_project: forceProject,
        use_query_intent: intentActive,
      });
      setPreview(data);
      setExcludedChunkIds(new Set());
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: string } } }).response?.data?.detail)
          : e instanceof Error
            ? e.message
            : String(e);
      setError(msg || 'Retrieve preview failed');
      setPreview(null);
    } finally {
      setLoadingPreview(false);
    }
  }, [projectName, queryText, topK, contextWeighting, forceProject, intentActive]);

  const executeStream = useCallback(
    async (excludeIds: string[]) => {
      setError(null);
      setLoadingStream(true);
      const userContent = queryText.trim();
      setMessages((m) => {
        const next = [
          ...m,
          { role: 'user' as const, content: userContent },
          { role: 'assistant' as const, content: '' },
        ];
        streamingAssistantIndexRef.current = next.length - 1;
        return next;
      });

      const idx = () => streamingAssistantIndexRef.current;

      await streamQuery(
        {
          query: userContent,
          project_name: projectName.trim() || undefined,
          top_k: topK,
          model: ollamaModel.trim() || undefined,
          include_past_projects: includePastProjects,
          context_weighting: includePastProjects ? contextWeighting : undefined,
          exclude_chunk_ids: excludeIds.length > 0 ? excludeIds : undefined,
          force_project: includePastProjects ? forceProject : undefined,
          use_query_intent: intentActive,
        },
        {
          onSources: (sources, similarProjects) => {
            const i = idx();
            if (i == null) return;
            setMessages((m) => {
              const copy = [...m];
              const a = copy[i];
              if (a && a.role === 'assistant') {
                copy[i] = {
                  ...a,
                  sources,
                  similarProjectsFromStream: similarProjects,
                };
              }
              return copy;
            });
          },
          onChunk: (text) => {
            const i = idx();
            if (i == null) return;
            setMessages((m) => {
              const copy = [...m];
              const a = copy[i];
              if (a && a.role === 'assistant') {
                copy[i] = { ...a, content: a.content + text };
              }
              return copy;
            });
          },
          onDone: (totalTime, timings) => {
            const i = idx();
            if (i == null) return;
            setMessages((m) => {
              const copy = [...m];
              const a = copy[i];
              if (a && a.role === 'assistant') {
                copy[i] = { ...a, totalTime, timings };
              }
              return copy;
            });
          },
          onError: (message) => {
            setError(message);
            const i = idx();
            if (i == null) return;
            setMessages((m) => {
              const copy = [...m];
              const a = copy[i];
              if (a && a.role === 'assistant' && !a.content) {
                copy[i] = { ...a, content: `(Error: ${message})` };
              }
              return copy;
            });
          },
        }
      );

      streamingAssistantIndexRef.current = null;

      setLoadingStream(false);
      setPreview(null);
      setExcludedChunkIds(new Set());
    },
    [
      queryText,
      projectName,
      topK,
      includePastProjects,
      contextWeighting,
      forceProject,
      ollamaModel,
      intentActive,
    ]
  );

  const handleSend = async () => {
    if (!queryText.trim()) {
      setError('Enter a question.');
      return;
    }
    if (useQueryIntent && !projectName.trim()) {
      setError('Select a project to use query intent (weighted retrieval by document type).');
      return;
    }
    if (includePastProjects && !projectName.trim()) {
      setError('Project name is required when including similar past projects.');
      return;
    }
    if (includePastProjects) {
      await runRetrievalPreview();
      return;
    }
    await executeStream([]);
  };

  const handleConfirmPreview = () => {
    const excludeIds = Array.from(excludedChunkIds);
    void executeStream(excludeIds);
  };

  const handleCancelPreview = () => {
    setPreview(null);
    setExcludedChunkIds(new Set());
  };

  const toggleChunkIncluded = (chunkId: string, included: boolean) => {
    setExcludedChunkIds((prev) => {
      const next = new Set(prev);
      if (included) {
        next.delete(chunkId);
      } else {
        next.add(chunkId);
      }
      return next;
    });
  };

  const forceOptions = projects.filter(
    (p) => !projectName.trim() || p.project_name !== projectName.trim()
  );

  return (
    <div className="chat-interface">
      <h1>RAG4Risk</h1>
      <p className="chat-subtitle">
        Query documents with optional similar past projects (Phase 3.5). When past projects are on,
        you review retrieval first, then confirm to call the LLM.
      </p>

      <div className="chat-controls">
        <div className="chat-row">
          <div className="chat-field">
            <span>Project (filter / current project)</span>
            <div className="chat-row" style={{ alignItems: 'stretch' }}>
              <select
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                aria-label="Select project"
                style={{ flex: 1, minWidth: 0 }}
              >
                <option value="">— Select project —</option>
                {projectSelectOptions.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
              <button
                type="button"
                className="secondary"
                style={{ alignSelf: 'center', whiteSpace: 'nowrap' }}
                onClick={refreshProjectNames}
                title="Reload projects from documents and metadata"
              >
                Refresh
              </button>
            </div>
            <span style={{ fontSize: '0.75rem', fontWeight: 400 }}>
              From uploaded documents and <code>/api/projects</code>. Upload documents or add metadata if empty.
            </span>
            {projectListError ? (
              <p className="chat-field-error" role="alert">
                Could not load some project data: {projectListError}. With <code>npm run dev</code>, ensure the
                Vite proxy can reach the API (backend on port 8000) or set <code>VITE_API_BASE_URL</code>.
              </p>
            ) : null}
            {!projectListError && projectOptions.length === 0 ? (
              <p className="chat-field-hint" style={{ fontSize: '0.75rem', marginTop: 4 }}>
                No project names yet — ingest at least one document with a project name, or POST project metadata
                to <code>/api/projects/metadata</code>.
              </p>
            ) : null}
          </div>
          <div className="chat-field" style={{ maxWidth: 100 }}>
            <span>Top K</span>
            <input
              type="number"
              min={1}
              max={20}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value) || 5)}
            />
          </div>
        </div>

        <div className="chat-row">
          <label>
            <input
              type="checkbox"
              checked={includePastProjects}
              onChange={(e) => setIncludePastProjects(e.target.checked)}
            />
            Include similar past projects
          </label>
        </div>

        <div className="chat-row">
          <label>
            <input
              type="checkbox"
              checked={useQueryIntent}
              onChange={(e) => setUseQueryIntent(e.target.checked)}
            />
            Use query intent (per-document-type weights)
          </label>
        </div>
        <p className="chat-field-hint" style={{ fontSize: '0.75rem', marginTop: -4, marginBottom: 8 }}>
          Runs the intent model when <code>QUERY_INTENT_ENABLED</code> is set on the API. Requires a project
          above; splits <code>top_k</code> across document types (SOW, solution description, etc.).
        </p>

        <div className="chat-field">
          <span>Ollama model (LLM)</span>
          <div className="chat-row" style={{ alignItems: 'stretch' }}>
            <select
              value={ollamaModel}
              onChange={(e) => setOllamaModel(e.target.value)}
              aria-label="Select Ollama model"
              style={{ flex: 1, minWidth: 0 }}
            >
              <option value="">Default (server configuration)</option>
              {ollamaSelectOptions.map((name) => (
                <option key={name} value={name}>
                  {name}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="secondary"
              style={{ alignSelf: 'center', whiteSpace: 'nowrap' }}
              onClick={refreshOllamaModels}
              title="Refresh list from Ollama (via API)"
            >
              Refresh models
            </button>
          </div>
          <span style={{ fontSize: '0.75rem', fontWeight: 400 }}>
            Used only for the streaming answer. List from <code>/api/diagnostics/ollama-models</code>.
          </span>
        </div>

        <details className="advanced">
          <summary>Advanced — context weighting &amp; force project</summary>
          <div className="weight-sliders">
            <label>
              Current project weight ({currentWeight.toFixed(2)})
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={currentWeight}
                onChange={(e) => setCurrentWeight(Number(e.target.value))}
                disabled={!includePastProjects}
              />
            </label>
            <label>
              Past projects weight ({pastWeight.toFixed(2)})
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={pastWeight}
                onChange={(e) => setPastWeight(Number(e.target.value))}
                disabled={!includePastProjects}
              />
            </label>
          </div>
          <div className="chat-field" style={{ marginTop: '0.75rem' }}>
            <span>Force-select past project (optional)</span>
            <select
              value={forceSelect}
              onChange={(e) => setForceSelect(e.target.value)}
              disabled={!includePastProjects}
            >
              <option value="">— None (use similarity) —</option>
              {forceOptions.map((p) => (
                <option key={forceOptionValue(p)} value={forceOptionValue(p)}>
                  {p.customer} — {p.project_name}
                </option>
              ))}
            </select>
            <span style={{ fontSize: '0.75rem', fontWeight: 400 }}>
              Uses metadata from GET /api/projects. Add metadata via upload or POST /api/projects/metadata.
            </span>
          </div>
        </details>

        <div className="chat-field">
          <span>Your question</span>
          <textarea
            className="chat-query"
            value={queryText}
            onChange={(e) => setQueryText(e.target.value)}
            placeholder="Ask about risks, scope, issues…"
            rows={4}
          />
        </div>

        {error && <div className="chat-error">{error}</div>}

        <div className="chat-actions">
          <button
            type="button"
            className="primary"
            onClick={() => void handleSend()}
            disabled={loadingPreview || loadingStream}
          >
            {includePastProjects
              ? loadingPreview
                ? 'Loading preview…'
                : 'Get retrieval preview'
              : loadingStream
                ? 'Streaming…'
                : 'Send (stream)'}
          </button>
        </div>
      </div>

      {preview && (
        <div className="retrieval-preview">
          <h3>Retrieval preview (no LLM yet)</h3>
          <p style={{ fontSize: '0.85rem', marginTop: 0 }}>
            Review similar projects and chunks. Uncheck a chunk to exclude it from context. Then confirm
            to run the streaming query.
          </p>

          {preview.query_intent && (
            <div className="preview-section" style={{ marginBottom: '1rem' }}>
              <h4>Query intent</h4>
              <p style={{ fontSize: '0.9rem', marginTop: 0 }}>{preview.query_intent.intent_summary}</p>
              <p style={{ fontSize: '0.8rem', color: '#666', marginBottom: 4 }}>Slots (current project)</p>
              <pre className="metadata-json" style={{ fontSize: '0.75rem' }}>
                {JSON.stringify(preview.query_intent.slots_current_project, null, 2)}
              </pre>
            </div>
          )}

          <div className="preview-section">
            <h4>Similar projects</h4>
            {preview.similar_projects.length === 0 ? (
              <p style={{ fontSize: '0.9rem' }}>No similar projects returned (need metadata + other projects with chunks).</p>
            ) : (
              preview.similar_projects.map((sp) => (
                <div key={sp.project_name} className="similar-project-card">
                  <div>
                    <strong>{sp.project_name}</strong>
                    {sp.customer != null && sp.customer !== '' && (
                      <span> — {sp.customer}</span>
                    )}
                    <span className="score"> · similarity {(sp.similarity_score * 100).toFixed(1)}%</span>
                  </div>
                  {sp.metadata && Object.keys(sp.metadata).length > 0 && (
                    <pre className="metadata-json">{JSON.stringify(sp.metadata, null, 2)}</pre>
                  )}
                </div>
              ))
            )}
          </div>

          <div className="preview-section">
            <h4>Candidate chunks ({preview.chunks.length})</h4>
            {preview.chunks.map((c: PreviewChunk) => {
              const included = !excludedChunkIds.has(c.chunk_id);
              return (
                <div
                  key={c.chunk_id}
                  className={`preview-chunk ${included ? '' : 'excluded'}`}
                >
                  <input
                    type="checkbox"
                    checked={included}
                    onChange={(e) => toggleChunkIncluded(c.chunk_id, e.target.checked)}
                    aria-label="Include chunk in context"
                  />
                  <div className="preview-chunk-body">
                    <div className="preview-chunk-meta">
                      <code style={{ fontSize: '0.7rem' }}>{c.chunk_id.slice(0, 8)}…</code>
                      <span>{c.project_name}</span>
                      {c.customer && <span>{c.customer}</span>}
                      {c.is_past_project ? (
                        <span className="badge-past">Past project</span>
                      ) : (
                        <span className="badge-current">Current project</span>
                      )}
                    </div>
                    <div className="chunk-snippet">
                      {c.text.length > 400 ? `${c.text.slice(0, 400)}…` : c.text}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>

          <div className="preview-actions">
            <button
              type="button"
              className="primary"
              onClick={() => void handleConfirmPreview()}
              disabled={loadingStream}
            >
              Confirm &amp; run query (stream)
            </button>
            <button type="button" className="secondary" onClick={handleCancelPreview}>
              Cancel
            </button>
          </div>
        </div>
      )}

      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-role">{msg.role}</div>
            <div className="message-content">{msg.content}</div>
            {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
              <div className="sources-block">
                <h5>Sources</h5>
                {msg.sources.map((s, j) => (
                  <div
                    key={`${s.chunk_id}-${j}`}
                    className={`source-line ${s.is_past_project ? 'past' : ''}`}
                  >
                    <strong>{s.document_name}</strong>
                    {s.is_past_project && <span className="badge-past" style={{ marginLeft: 8 }}>Past</span>}
                    <br />
                    <span style={{ color: '#888' }}>
                      {s.document_type} · {s.project_name}
                      {s.chunk_id && ` · chunk ${s.chunk_id.slice(0, 8)}…`}
                    </span>
                    {s.relevance_score != null && (
                      <span style={{ marginLeft: 8 }}>
                        relevance {(s.relevance_score * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                ))}
                {msg.similarProjectsFromStream && msg.similarProjectsFromStream.length > 0 && (
                  <div className="similar-used">
                    <div style={{ fontWeight: 600, marginBottom: 4 }}>Similar projects (from stream)</div>
                    <ul style={{ margin: 0, paddingLeft: '1.25rem' }}>
                      {msg.similarProjectsFromStream.map((sp) => (
                        <li key={sp.project_name}>
                          <strong>{sp.project_name}</strong>
                          {sp.customer ? ` — ${sp.customer}` : ''}
                          {sp.similarity_score != null && (
                            <span className="score"> · {(sp.similarity_score * 100).toFixed(1)}%</span>
                          )}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {msg.totalTime != null && (
                  <div className="similar-used">Total time: {msg.totalTime.toFixed(2)}s</div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export default ChatInterface;
