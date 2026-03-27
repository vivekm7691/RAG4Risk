# RAG4Risk frontend

## Dev server

```bash
npm install
npm run dev
```

- URL: **http://localhost:5173** (see `vite.config.ts`; avoids conflicting with Docker on port 3000).
- API: requests to `/api/*` are proxied to `http://localhost:8000` during dev. The app uses **same-origin** `/api/...` in dev (empty axios base URL) so the dropdown and chat hit the proxy; set `VITE_API_BASE_URL` only if you need a different API host.

## If `npm run dev` fails

1. Run `npm install` from this folder (`frontend/`).
2. Stop the Docker frontend container if you only need local Vite (Compose often binds host port **3000**).
3. If you see **EADDRINUSE**, another process is using the port—close it or change `server.port` in `vite.config.ts`.
4. On corporate OneDrive paths, antivirus/sync can slow installs; run from a short path if needed.

## Production build

```bash
npm run build
```

Output: `dist/` (served by nginx in Docker).
