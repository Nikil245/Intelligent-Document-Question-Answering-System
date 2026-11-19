# DocuMind AI — Frontend

React + Vite + Tailwind CSS frontend for the **Intelligent Document Question Answering System** academic NLP mini-project. Connects directly to the FastAPI backend powered by PyMuPDF, Scikit-learn TF-IDF + cosine similarity retrieval, and Hugging Face `distilbert/distilbert-base-cased-distilled-squad`.

---

## Features

- **Live Backend & Model Health Monitoring:** Polls `GET /api/health` to show model readiness (`distilbert/distilbert-base-cased-distilled-squad`), compute device (`CUDA` / `CPU`), and active documents in memory.
- **Drag-and-Drop PDF Upload:** Validates `.pdf` selection and 20 MB limit, displays live upload progress (`POST /api/documents/upload`), and surfaces clear errors (e.g., scanned PDFs without selectable text, encrypted PDFs, corrupted files, or oversized uploads).
- **Document Management:** Inspects filename, UUID, total pages, pages with text, skipped empty pages, extracted character count, chunk count, and status (`GET /api/documents/{id}`), with one-click refresh and confirmed deletion (`DELETE /api/documents/{id}`).
- **Multi-Turn Extractive QA Chat:** Ask multiple questions against the same uploaded PDF (`POST /api/questions/ask`) with loading states, suggested starter questions, and clean handling of unanswerable (`no_relevant_content`) and low-confidence (`low_confidence`) responses.
- **Source Reference Panel:** Displays the original PDF page number, chunk index, TF-IDF cosine similarity, and highlights the exact extracted answer span (`answer_start` to `answer_end`) inside the supporting passage, alongside an explicit note explaining that the model score is uncalibrated rather than a probability of correctness.

---

## Configuration

The backend URL is configured through `.env` (see `.env.example`):

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

---

## Installation & Running

Make sure the FastAPI backend is running first (see `../backend/README.md`):

```bash
# Terminal 1 — Start Backend (from backend/)
cd backend
HF_HUB_OFFLINE=1 .venv/bin/python run.py
```

Then install and run the frontend:

```bash
# Terminal 2 — Install & Start Frontend (from frontend/)
cd frontend
npm install
npm run dev
```

Open `http://127.0.0.1:5173` in your browser.

### Production Build

```bash
npm run build
npm run preview
```

---

## Project Structure

```text
frontend/
├── public/
│   └── favicon.svg
├── src/
│   ├── assets/
│   │   └── logo.svg
│   ├── components/
│   │   ├── ChatInterface.jsx
│   │   ├── DocumentCard.jsx
│   │   ├── DocumentUpload.jsx
│   │   ├── Header.jsx
│   │   ├── Sidebar.jsx
│   │   ├── SourcePanel.jsx
│   │   ├── StatsOverview.jsx
│   │   └── SystemStatusView.jsx
│   ├── hooks/
│   │   ├── useBackendHealth.js
│   │   └── useDocumentQA.js
│   ├── pages/
│   │   └── Dashboard.jsx
│   ├── services/
│   │   └── api.js
│   ├── utils/
│   │   └── formatters.js
│   ├── App.jsx
│   ├── index.css
│   └── main.jsx
├── .env
├── .env.example
├── .gitignore
├── index.html
├── package.json
├── postcss.config.js
├── tailwind.config.js
├── vite.config.js
└── README.md
```

