# Moderation Frontend (Vite + React)

Single-page UI to paste **text** or upload **images/URLs**, call the backend,
and display moderation results (decision, categories, scores, actions).

## Quick start (local)

```bash
cd frontend
npm install
# Point to your backend; default is http://localhost:8000
VITE_API_BASE=http://localhost:8000 npm run dev
