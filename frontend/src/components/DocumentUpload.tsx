import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { DOCUMENT_TYPE_OPTIONS, documentAPI } from '../services/api';
import { useAppContext } from '../context/AppContext';
import './Phase4Pages.css';

const MAX_BYTES = 50 * 1024 * 1024;

function extensionForType(documentType: string): '.docx' | '.xlsx' {
  return documentType === 'risk register' || documentType === 'issue log' ? '.xlsx' : '.docx';
}

export function DocumentUpload() {
  const navigate = useNavigate();
  const { bumpDocuments } = useAppContext();
  const [projectName, setProjectName] = useState('');
  const [documentType, setDocumentType] = useState(DOCUMENT_TYPE_OPTIONS[0].value);
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  useEffect(() => {
    if (!file) return;
    const ext = extensionForType(documentType);
    if (!file.name.toLowerCase().endsWith(ext)) {
      setFile(null);
      setError(`File type does not match document type (expected ${ext}).`);
    }
  }, [documentType, file]);

  const pickFile = useCallback(
    (f: File | null) => {
      setError(null);
      setSuccess(null);
      if (!f) {
        setFile(null);
        return;
      }
      const ext = extensionForType(documentType);
      if (!f.name.toLowerCase().endsWith(ext)) {
        setError(`This document type requires a ${ext} file.`);
        setFile(null);
        return;
      }
      if (f.size > MAX_BYTES) {
        setError(`File is too large (max ${MAX_BYTES / (1024 * 1024)} MB).`);
        setFile(null);
        return;
      }
      setFile(f);
    },
    [documentType]
  );

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragActive(false);
      const f = e.dataTransfer.files?.[0];
      if (f) pickFile(f);
    },
    [pickFile]
  );

  const submit = async () => {
    setError(null);
    setSuccess(null);
    const pn = projectName.trim();
    if (!pn) {
      setError('Enter a project name.');
      return;
    }
    if (!file) {
      setError('Choose a file.');
      return;
    }
    setUploading(true);
    setProgress(0);
    try {
      await documentAPI.upload(file, pn, documentType, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded * 100) / e.total));
        },
      });
      setSuccess(`Uploaded “${file.name}” into project “${pn}”.`);
      setFile(null);
      bumpDocuments();
    } catch (e: unknown) {
      const msg =
        e && typeof e === 'object' && 'response' in e
          ? String((e as { response?: { data?: { detail?: unknown } } }).response?.data?.detail)
          : e instanceof Error
            ? e.message
            : String(e);
      setError(msg || 'Upload failed');
    } finally {
      setUploading(false);
      setProgress(0);
    }
  };

  return (
    <div className="phase4-page">
      <h1>Upload documents</h1>
      <p className="page-lead">
        Word types use <code>.docx</code>; risk register and issue log use <code>.xlsx</code>. Max{' '}
        {MAX_BYTES / (1024 * 1024)} MB.
      </p>

      <div className="phase4-form">
        <label>
          Project name
          <input
            type="text"
            value={projectName}
            onChange={(e) => setProjectName(e.target.value)}
            placeholder="e.g. Freedom Encompass Modernization"
            maxLength={100}
            autoComplete="off"
          />
        </label>

        <label>
          Document type
          <select value={documentType} onChange={(e) => setDocumentType(e.target.value)}>
            {DOCUMENT_TYPE_OPTIONS.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>

        <div
          className={`drop-zone ${dragActive ? 'drag-active' : ''}`}
          onDragOver={(e) => {
            e.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={onDrop}
          onClick={() => document.getElementById('phase4-file-input')?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              document.getElementById('phase4-file-input')?.click();
            }
          }}
        >
          <p>
            <strong>Drag and drop</strong> a file here, or click to browse
          </p>
          <p className="muted">Expected: {extensionForType(documentType)}</p>
          {file && (
            <p>
              Selected: <strong>{file.name}</strong> ({(file.size / 1024).toFixed(1)} KB)
            </p>
          )}
        </div>
        <input
          id="phase4-file-input"
          type="file"
          accept={extensionForType(documentType)}
          style={{ display: 'none' }}
          onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
        />

        {uploading && (
          <div>
            <span className="muted">Uploading… {progress}%</span>
            <div className="progress-bar">
              <div style={{ width: `${progress}%` }} />
            </div>
          </div>
        )}

        {error && <div className="phase4-error">{error}</div>}
        {success && (
          <div style={{ color: '#2e7d32', fontSize: '0.9rem' }}>
            {success}{' '}
            <button type="button" className="link-button" onClick={() => navigate(`/?project=${encodeURIComponent(projectName.trim())}`)}>
              Open chat for this project
            </button>
          </div>
        )}

        <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button type="button" className="primary" disabled={uploading} onClick={() => void submit()}>
            Upload
          </button>
          <button type="button" className="secondary" disabled={uploading} onClick={() => pickFile(null)}>
            Clear file
          </button>
        </div>
      </div>
    </div>
  );
}
