# Moderation App

Moderation App is a full-stack reference project that combines OpenAI's multimodal Moderation API with an optional PyTorch pre-filter service. It demonstrates how to layer custom rule logic on top of vendor moderation signals, expose the results through a FastAPI backend, and surface interactive tooling in a React UI. Docker, Kubernetes manifests, and dedicated docs round out the repo to help you move from local prototyping to production-style deployments.

## Features
- FastAPI backend that fans out to OpenAI Moderation and a local TorchScript classifier, merges the signals, and applies YAML-driven policy rules (`backend/`).
- PyTorch microservice that scores text and images before they hit OpenAI, letting you block/allow requests early (`torch_filter/`).
- Vite + React frontend for quick manual testing of text/image moderation pipelines (`frontend/`).
- Container-first workflow: Dockerfiles for each service, a `docker-compose.yml` for local orchestration, and Kubernetes manifests under `infra/`.
- Lightweight test suites (pytest) and rich reference docs under `docs/` to explain design decisions, pipelines, and rollout guidance.

## Repository Layout
| Path | Description |
| --- | --- |
| `backend/` | FastAPI service, moderation cascade (`main.py`), OpenAI/PyTorch integration, config & tests. |
| `torch_filter/` | TorchScript-powered FastAPI microservice plus export scripts and test suite. |
| `frontend/` | React + Vite single-page app for interacting with the moderation API. |
| `infra/` | Kubernetes manifests for torch filter, backend, and frontend deployments/services. |
| `docs/` | Planning notes, architecture write-ups, and detailed setup guides. |
| `docker-compose.yml` | Spins up all services locally with sane defaults. |

## Prerequisites
- Python 3.11+ for backend and torch filter development environments.
- Node 18+/npm for the Vite frontend.
- Docker 24+ (optional but recommended for running the 3 services together).
- An OpenAI API key with access to the Moderation endpoint.
- TorchScript models for the prefilter service (place them in `torch_filter/filters/` or use the provided export scripts).

## Quick Start (Docker Compose)
1. Copy `.env` (or `.env.example` if you create one) and set `OPENAI_API_KEY`, `TORCH_FILTER_URL`, and any policy overrides.
2. Build and start everything:
   ```powershell
   docker compose up --build
   ```
3. Visit the React app at <http://localhost:5173>. It proxies API calls to the backend on port `8000`, which in turn speaks to the torch filter on `9000` and OpenAI.
4. Stop the stack with `docker compose down` when you're done.

## Manual Development Workflow
### Torch Filter
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r torch_filter\requirements.txt
uvicorn torch_filter.service:app --reload --port 9000
```
> Drop TorchScript artifacts in `torch_filter/filters/` (`model.ts`, `text_model.ts`) or export them via `export_*.py` scripts.

### Backend
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r backend\requirements.txt
set OPENAI_API_KEY=sk-...            # or use backend/.env
set TORCH_FILTER_URL=http://localhost:9000
uvicorn backend.main:app --reload --port 8000
```
Policy changes live in `backend/policy.yaml`; edit thresholds/actions to tune decisions (`allow`, `warn`, `support`, `block`, `ban`).

### Frontend
```powershell
cd frontend
npm install
npm run dev -- --host
```
Expose the backend URL to the UI through `VITE_API_BASE` (env var or `.env` file in `frontend/`).

## Testing
- Backend tests: `pytest backend/tests`
- Torch filter tests: `pytest torch_filter/tests`
- Frontend linting/tests: add your preferred tooling (e.g., `npm run test`)—Vite scaffolding is in place.

## Deployment Notes
- Docker images are multi-stage and ready for registries; adjust environment variables through your orchestrator.
- `infra/` contains Kubernetes manifests for each service (Deployments + Services). Update container image names and any secrets before applying.
- For production, replace the example `.env` values with secrets, provision persistent storage for TorchScript models, and consider enabling HTTPS/CDN for the frontend.

## Further Reading
Additional planning docs, architecture diagrams, and playbooks are available under `docs/`. Start with `Startup Guide - Moderation App (open Ai Moderation + Py Torch Prefilter).docx` for a narrative walkthrough.
