import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { documentAPI, projectsAPI, type ProjectMetadata } from '../services/api';
import { useAppContext } from '../context/AppContext';
import './Phase4Pages.css';

type Row = {
  project_name: string;
  documentCount: number;
  customer?: string | null;
  fromMetadata: boolean;
};

export function ProjectSelector() {
  const { documentsVersion } = useAppContext();
  const [filter, setFilter] = useState('');
  const [rows, setRows] = useState<Row[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docs, meta] = await Promise.all([documentAPI.list(), projectsAPI.list().catch(() => [] as ProjectMetadata[])]);
      const counts = new Map<string, number>();
      for (const d of docs) {
        const n = d.project_name?.trim();
        if (!n) continue;
        counts.set(n, (counts.get(n) ?? 0) + 1);
      }
      const metaByName = new Map<string, ProjectMetadata>();
      for (const p of meta) {
        const n = p.project_name?.trim();
        if (n) metaByName.set(n, p);
      }
      const names = new Set<string>([...counts.keys(), ...metaByName.keys()]);
      const list: Row[] = Array.from(names).map((project_name) => {
        const m = metaByName.get(project_name);
        return {
          project_name,
          documentCount: counts.get(project_name) ?? 0,
          customer: m?.customer ?? null,
          fromMetadata: Boolean(m),
        };
      });
      list.sort((a, b) => a.project_name.localeCompare(b.project_name, undefined, { sensitivity: 'base' }));
      setRows(list);
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail)
          : 'Failed to load projects';
      setError(msg);
      setRows([]);
    } finally {
      setLoading(false);
    }
  }, [documentsVersion]);

  useEffect(() => {
    void load();
  }, [load]);

  const filtered = useMemo(() => {
    const q = filter.trim().toLowerCase();
    if (!q) return rows;
    return rows.filter(
      (r) =>
        r.project_name.toLowerCase().includes(q) ||
        (r.customer && r.customer.toLowerCase().includes(q))
    );
  }, [rows, filter]);

  return (
    <div className="phase4-page">
      <h1>Projects</h1>
      <p className="page-lead">
        Projects appear from uploaded documents and from{' '}
        <code>/api/projects</code> metadata. Document count is derived from unique files in the vector index.
      </p>

      <div className="phase4-toolbar">
        <label style={{ flex: 1, minWidth: 200, maxWidth: 360 }}>
          Search
          <input
            type="search"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Project or customer"
            style={{ marginTop: 4, width: '100%', padding: '0.45rem' }}
          />
        </label>
        <button type="button" className="secondary" disabled={loading} onClick={() => void load()}>
          Refresh
        </button>
      </div>

      {error && <div className="phase4-error">{error}</div>}

      {loading ? (
        <p className="muted">Loading…</p>
      ) : filtered.length === 0 ? (
        <p className="muted">No projects match. Upload documents or add project metadata.</p>
      ) : (
        <div className="phase4-table-wrap">
          <table className="phase4-table">
            <thead>
              <tr>
                <th>Project</th>
                <th>Customer</th>
                <th>Documents</th>
                <th>Metadata</th>
                <th>Chat</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((r) => (
                <tr key={r.project_name}>
                  <td>
                    <strong>{r.project_name}</strong>
                  </td>
                  <td>{r.customer || '—'}</td>
                  <td>
                    <span className="badge-count">{r.documentCount}</span>
                  </td>
                  <td>{r.fromMetadata ? 'Yes' : '—'}</td>
                  <td>
                    <Link to={`/?project=${encodeURIComponent(r.project_name)}`}>Open chat</Link>
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
