# RAG4Risk

RAG system for ingesting Word documents, generating vector embeddings with project-based indexing, and providing semantic search-powered Q&A through a web interface using local LLM.

## Overview

RAG4Risk is a Retrieval-Augmented Generation (RAG) system designed to:
- Ingest Word documents (Statements of Work, Technical Solution Descriptions, Proposals)
- Create semantic embeddings with project-based indexing
- Provide Q&A interface powered by local LLM (Ollama)
- Enable semantic search across project documents

## Architecture

```
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌──────────┐
│  User   │────▶│ Frontend │────▶│   Backend   │────▶│  Qdrant  │
│         │◀────│  (React) │◀────│  (FastAPI)  │◀────│  (Vector │
└─────────┘     └──────────┘     └─────────────┘     │    DB)   │
                                                     └──────────┘
                                                           │
                                                     ┌──────────┐
                                                     │  Ollama  │
                                                     │   (LLM)  │
                                                     └──────────┘
```

## Technology Stack

- **Backend**: FastAPI (Python 3.11)
- **Frontend**: React + TypeScript (Vite)
- **Vector Database**: Qdrant
- **LLM**: Ollama (Llama 3.1 or Mistral)
- **RAG Framework**: LangChain
- **Containerization**: Docker Compose

## Prerequisites

- Docker and Docker Compose
- Ollama installed on host machine (for development)
  - Download from: https://ollama.ai
  - Pull a model: `ollama pull llama3.1`

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/vivekm7691/RAG4Risk.git
cd RAG4Risk
```

### 2. Set Up Environment Variables

```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Start Services with Docker Compose

```bash
docker-compose up -d
```

This will start:
- Backend API on http://localhost:8000
- Frontend on http://localhost:3000
- Qdrant vector database on http://localhost:6333

### 4. Verify Services

- Backend health: http://localhost:8000/health
- Frontend: http://localhost:3000
- Qdrant: http://localhost:6333/health

## Development

### Backend Development

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173** (Vite dev server). The app proxies `/api` to `http://localhost:8000`, so run the backend locally or ensure it is reachable.

**Note:** Docker Compose serves the built frontend on **http://localhost:3000**. If `npm run dev` used to fail with **port already in use**, that was often a clash on port 3000—the dev server now defaults to **5173**. If 5173 is busy, Vite will try the next free port (`strictPort: false`).

### Running Tests

**Backend:**
```bash
cd backend
pytest --cov=app --cov-report=term-missing
```

**Frontend:**
```bash
cd frontend
npm test
```

## Project Structure

```
RAG4Risk/
├── backend/              # FastAPI backend
│   ├── app/
│   │   ├── main.py      # FastAPI application
│   │   ├── config.py    # Configuration
│   │   ├── models/      # Pydantic models
│   │   ├── services/    # Business logic
│   │   ├── api/         # API routes
│   │   └── utils/       # Utilities
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/             # React frontend
│   ├── src/
│   │   ├── components/  # React components
│   │   ├── services/   # API client
│   │   └── App.tsx
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml   # Docker orchestration
├── .env.example         # Environment template
└── README.md
```

## Environment Variables

See `.env.example` for all available environment variables.

Key variables:
- `OLLAMA_BASE_URL`: Ollama service URL (default: http://ollama:11434)
- `OLLAMA_MODEL`: LLM model name (default: llama3.2:3b)
- `QDRANT_HOST`: Qdrant host (default: qdrant)
- `QDRANT_PORT`: Qdrant port (default: 6333)

## Branching Strategy

This project follows Git Flow branching strategy. See [GitHub Branching Strategy for RAG4Risk.md](GitHub%20Branching%20Strategy%20for%20RAG4Risk.md) for details.

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: Feature branches
- `bugfix/*`: Bug fix branches
- `hotfix/*`: Critical production fixes
- `release/*`: Release preparation

## CI/CD

GitHub Actions workflows are configured for:
- Automated testing (linting, unit tests, integration tests)
- Docker image builds
- Security scanning

See `.github/workflows/` for workflow definitions.

## Contributing

1. Create a feature branch from `develop`
2. Make your changes
3. Run tests and linting
4. Create a Pull Request to `develop`
5. Ensure all CI checks pass

## License

[Add your license here]

## Support

For issues and questions, please open an issue on GitHub.

