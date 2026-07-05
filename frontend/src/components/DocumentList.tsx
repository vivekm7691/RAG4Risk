import { useCallback, useEffect, useMemo, useState } from 'react';
import { documentAPI, listDistinctProjectNames } from '../services/api';
import type { DocumentMetadataRecord } from '../services/api';
import { useAppContext } from '../context/AppContext';
import './Phase4Pages.css';

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function formatWhen(iso: string): string {
  try {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return iso;
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

export function DocumentList() {
  const { documentsVersion, bumpDocuments } = useAppContext();
  const [filterProject, setFilterProject] = useState('');
  const [projectOptions, setProjectOptions] = useState<string[]>([]);
  const [rows, setRows] = useState<DocumentMetadataRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const loadProjects = useCallback(() => {
    void listDistinctProjectNames().then(({ names }) => setProjectOptions(names));
  }, []);

  const loadDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await documentAPI.list(filterProject.trim() || undefined);
      setRows(data);
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail)
          : 'Failed to load documents';
      setError(msg);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [filterProject, documentsVersion]);

  useEffect(() => {
    loadProjects();
  }, [loadProjects, documentsVersion]);

  useEffect(() => {
    void loadDocs();
  }, [loadDocs]);

  const sorted = useMemo(
    () =>
      [...rows].sort((a, b) =>
        (a.project_name + a.file_name).localeCompare(b.project_name + b.file_name, undefined, {
          sensitivity: 'base',
        })
      ),
    [rows]
  );

  const onDelete = async (doc: DocumentMetadataRecord) => {
    if (!window.confirm(`Delete “${doc.file_name || doc.title || doc.document_id}” from the vector store?`)) {
      return;
    }
    setDeletingId(doc.document_id);
    setError(null);
    try {
      await documentAPI.delete(doc.document_id);
      bumpDocuments();
      await loadDocs();
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail)
          : 'Delete failed';
      setError(msg);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="phase4-page">
      <h1>Documents</h1>
      <p className="page-lead">Ingested files (from the vector index). Deleting removes all chunks for that document.</p>

      <div className="phase4-toolbar">
        <label style={{ minWidth: 220 }}>
          Filter by project
          <select
            value={filterProject}
            onChange={(e) => setFilterProject(e.target.value)}
            style={{ marginTop: 4, width: '100%', padding: '0.45rem' }}
          >
            <option value="">All projects</option>
            {projectOptions.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </label>
        <button type="button" className="secondary" disabled={loading} onClick={() => void loadDocs()}>
          Refresh
        </button>
      </div>

      {error && <div className="phase4-error">{error}</div>}

      {loading ? (
        <p className="muted">Loading…</p>
      ) : sorted.length === 0 ? (
        <p className="muted">No documents found. Upload from the Upload tab.</p>
      ) : (
        <div className="phase4-table-wrap">
          <table className="phase4-table">
            <thead>
              <tr>
                <th>File</th>
                <th>Type</th>
                <th>Project</th>
                <th>Uploaded</th>
                <th>Size</th>
                <th className="actions">Actions</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((d) => (
                <tr key={d.document_id}>
                  <td>{d.file_name || d.title || d.document_id.slice(0, 8)}</td>
                  <td>{d.document_type}</td>
                  <td>{d.project_name}</td>
                  <td>{formatWhen(d.upload_date)}</td>
                  <td>{formatBytes(d.file_size)}</td>
                  <td className="actions">
                    <button
                      type="button"
                      className="link-button"
                      disabled={deletingId === d.document_id}
                      onClick={() => void onDelete(d)}
                    >
                      {deletingId === d.document_id ? 'Deleting…' : 'Delete'}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
