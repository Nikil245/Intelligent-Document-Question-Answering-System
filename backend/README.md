# Intelligent Document Question Answering System — Backend

Backend REST API for an academic NLP mini-project that lets users upload PDF documents and ask natural-language questions about their contents. The system extracts text page by page, indexes chunks with **TF-IDF + cosine similarity**, and extracts verbatim answer spans using the pretrained **`distilbert/distilbert-base-cased-distilled-squad`** Transformer model.

---

## 1. Project Introduction

Working through long multi-page PDF documents (lecture notes, research papers, technical manuals, or reports) to locate specific facts is slow and error-prone. This project builds an **Intelligent Document Question Answering System** backend using FastAPI, PyMuPDF, Scikit-learn, PyTorch, and Hugging Face Transformers.

Instead of returning raw keyword search hits alone or generating unverified text, the backend pairs classical information retrieval (**TF-IDF**) with **extractive neural question answering (DistilBERT)** so every returned answer is directly quoted from the uploaded PDF along with its page number, supporting passage, and model confidence score.

---

## 2. Problem Statement

Users frequently need precise answers to targeted questions buried inside multi-page PDF documents. Standard `Ctrl+F` keyword search fails when a query is phrased as a question rather than an exact phrase, while passing an entire multi-page PDF directly into an encoder Transformer like DistilBERT is impossible due to its **512-token sequence length limit**.

We need a modular backend pipeline that:
1. Reliably extracts clean text from PDFs while preserving page numbers.
2. Narrows a large document down to the most relevant candidate passages.
3. Extracts accurate, grounded answers within the Transformer's token budget while refusing to fabricate answers when information is absent.

---

## 3. Objectives

- Validate uploaded PDF files (extension, MIME type, `%PDF-` file signature, size limit, encryption, and extractable text).
- Extract text page by page using PyMuPDF while accurately preserving 1-based original PDF page numbers.
- Split long pages into overlapping, word-boundary-safe chunks measured in **DistilBERT tokens** so every chunk fits safely within the 512-token limit alongside the question and special tokens.
- Build a per-document **TF-IDF** index and retrieve the top relevant chunks via **cosine similarity**.
- Load **`distilbert/distilbert-base-cased-distilled-squad`** once at server startup (automatically selecting CUDA GPU when available, or CPU otherwise).
- Extract candidate answers from each retrieved chunk individually, compare model confidence scores, and return the strongest grounded answer, its page number, and supporting passage.
- Support asking multiple questions against the same uploaded PDF without re-uploading it, using an in-memory UUID document registry.
- Detect missing content, empty answers, and low-confidence predictions cleanly without inventing answers.
- Expose clean, CORS-enabled REST endpoints ready for integration with a separate React frontend.

---

## 4. Technology Stack

| Component | Technology / Package | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ *(verified on Python 3.12)* | Backend runtime |
| **Web Framework** | FastAPI (`fastapi>=0.110`) | REST API, dependency wiring, OpenAPI docs (`/docs`) |
| **ASGI Server** | Uvicorn (`uvicorn[standard]>=0.29`) | Production/development ASGI server |
| **Validation & Config** | Pydantic v2 & `pydantic-settings` | Request/response schemas and `.env` configuration |
| **File Uploads** | `python-multipart` | Multipart PDF file upload support |
| **PDF Extraction** | PyMuPDF (`PyMuPDF>=1.24.3`, imported as `pymupdf`) | Fast, accurate page-by-page PDF text extraction |
| **Document Retrieval** | Scikit-learn (`scikit-learn>=1.4`) & NumPy | TF-IDF vectorization and cosine similarity ranking |
| **Deep Learning / NLP** | PyTorch (`torch>=2.2`) & Hugging Face `transformers>=4.40,<5` | Pretrained `distilbert/distilbert-base-cased-distilled-squad` tokenizer, model, and `"question-answering"` pipeline |
| **Testing** | Pytest (`pytest>=8.0`) & HTTPX (`httpx>=0.27`) | Unit tests, FastAPI `TestClient` API tests, and real-model integration tests |

> **Important Dependency Note:** `requirements.txt` pins `transformers>=4.40,<5` because `transformers 5.x` removed the `"question-answering"` pipeline task that this backend uses.

---

## 5. Pretrained Model Explanation

This project uses the exact pretrained model required by the assignment:

- **Model ID:** `distilbert/distilbert-base-cased-distilled-squad`
- **Architecture:** DistilBERT (a distilled, lighter version of BERT-base with 6 Transformer encoder layers instead of 12 — ~40% fewer parameters and ~60% faster inference while retaining ~97% of BERT's language understanding performance).
- **Casing:** `cased` — preserves uppercase/lowercase distinctions (helpful for proper nouns, acronyms, people, and organization names).
- **Fine-tuning Dataset:** Pretrained on **SQuAD v1.1** (Stanford Question Answering Dataset) by Hugging Face.
- **Usage Policy:**
  - Loaded via `AutoTokenizer`, `AutoModelForQuestionAnswering`, and Hugging Face's `"question-answering"` `pipeline`.
  - **No training or fine-tuning** is performed in this project.
  - **No secondary Transformer or embedding model** is used anywhere in the system.
  - Loaded **once** during FastAPI startup (`lifespan` hook) rather than per question.
  - Automatically detects `torch.cuda.is_available()` (`DEVICE=auto`) to run on NVIDIA GPU (`cuda:0`) when available and falls back to `cpu` otherwise.

---

## 6. System Architecture

The backend separates HTTP routing, validation/schemas, PDF extraction, token-aware chunking, TF-IDF retrieval, model lifecycle, QA orchestration, and in-memory storage into focused modules:

```text
backend/
├── app/
│   ├── main.py                     # FastAPI factory, lifespan model load, CORS, exception handlers
│   ├── config.py                   # Pydantic Settings (.env loading & token-budget validation)
│   ├── exceptions.py               # Domain exceptions mapped cleanly to HTTP error codes
│   ├── api/
│   │   └── routes.py               # REST endpoints under /api
│   ├── schemas/
│   │   └── request_response.py     # Pydantic request & response models
│   ├── services/
│   │   ├── model_service.py        # Loads DistilBERT once, selects CUDA/CPU, thread-safe predict()
│   │   ├── pdf_service.py          # Upload validation & PyMuPDF page-by-page extraction
│   │   ├── document_service.py     # Upload workflow orchestration (extract -> chunk -> index -> store)
│   │   ├── retrieval_service.py    # Per-document TF-IDF vectorizer + cosine similarity ranking
│   │   └── qa_service.py           # Retrieval + per-chunk DistilBERT QA + candidate selection
│   ├── storage/
│   │   └── document_store.py       # Thread-safe in-memory UUID document registry
│   └── utils/
│       └── text_chunking.py        # DistilBERT token-aware overlapping page chunker
├── samples/
│   └── sample_nlp_document.pdf     # Ready-to-use 4-page sample PDF for testing/demos
├── scripts/
│   └── create_sample_pdf.py        # Script that regenerates samples/sample_nlp_document.pdf
├── tests/
│   ├── conftest.py                 # FakeModelService, SimpleTokenizer & PDF helpers
│   ├── test_pdf_service.py         # PDF validation & extraction tests
│   ├── test_text_chunking.py       # Token-aware chunking & overlap tests
│   ├── test_retrieval_service.py   # TF-IDF retrieval & cosine similarity tests
│   ├── test_qa_service.py          # QA service & threshold tests (mocked model)
│   ├── test_api.py                 # FastAPI endpoint, error handling & CORS tests
│   └── test_integration_real_model.py # Optional end-to-end tests with real DistilBERT
├── uploads/                        # Temporary uploaded PDFs (named <uuid>.pdf)
├── .env.example                    # Example environment configuration
├── .gitignore
├── pytest.ini
├── requirements.txt
├── run.py                          # Uvicorn entrypoint
└── README.md
```

### Architecture Diagram

```text
  +-------------------+        POST /api/documents/upload
  |  React Frontend / | ----------------------------------------------+
  |    API Client     |                                               v
  +-------------------+                                   +------------------------+
            |                                             |     pdf_service.py     |
            | POST /api/questions/ask                     | (validate + PyMuPDF    |
            | { document_id, question }                   |  page-by-page extract) |
            v                                             +------------------------+
  +-------------------+                                               |
  |   api/routes.py   |                                               v
  +-------------------+                                   +------------------------+
            |                                             |    text_chunking.py    |
            v                                             | (DistilBERT tokenizer  |
  +-------------------+    1. Lookup UUID + TF-IDF index  |  384-tok chunks, 64 ol)|
  |   qa_service.py   | <-------------------------------- +------------------------+
  +-------------------+                                               |
       |         ^                                                    v
       |         | 2. Top-k chunks + pages                +------------------------+
       v         |                                        |  retrieval_service.py  |
  +-------------------+                                   | (TfidfVectorizer +     |
  | retrieval_service |                                   |  cosine_similarity)    |
  +-------------------+                                   +------------------------+
       |                                                              |
       | 3. Run DistilBERT on EACH retrieved chunk                    v
       v                                                  +------------------------+
  +-------------------+                                   |   document_store.py    |
  |  model_service.py |                                   | (In-memory UUID store) |
  |  (DistilBERT QA   |                                   +------------------------+
  |   loaded once)    |
  +-------------------+
       |
       | 4. Pick highest-scoring span >= CONFIDENCE_THRESHOLD
       v
  JSON Response: { answer, confidence, source: { page_number, passage, ... } }
```

---

## 7. Application Workflow

### Phase 1 — PDF Upload & Indexing (`POST /api/documents/upload`)
1. Client sends a multipart `POST` with a `.pdf` file.
2. Middleware checks `Content-Length` early; route reads at most `max_upload_size_bytes + 1` bytes (default limit: **20 MB**).
3. `pdf_service.validate_pdf_upload` verifies:
   - non-empty filename ending in `.pdf`
   - allowed `Content-Type` (`application/pdf`, `application/x-pdf`, `application/acrobat`, or `application/octet-stream`)
   - non-empty byte stream within the size limit
   - `%PDF-` signature within the first 1024 bytes.
4. `pdf_service.extract_text_from_pdf` opens the PDF with PyMuPDF (`pymupdf.open`), rejects encrypted or 0-page PDFs, extracts text separately for every page (numbered from **1**), cleans hyphenated line breaks and excess whitespace, tracks empty pages, and raises a clear OCR-unsupported error (`422 no_extractable_text`) if no page contains text.
5. `text_chunking.chunk_pages` splits each page independently using the DistilBERT fast tokenizer (`return_offsets_mapping=True`) into overlapping chunks of up to **384 tokens** with **64-token overlap**, snapping chunk boundaries to word edges and keeping exact page numbers.
6. `retrieval_service.TfidfRetriever` fits a TF-IDF matrix over all chunks of that document.
7. The original PDF is saved as `uploads/<uuid>.pdf`, and the metadata, pages, chunks, and TF-IDF retriever are registered in `DocumentStore` under the generated `document_id`.

### Phase 2 — Question Answering (`POST /api/questions/ask`)
1. Client submits `{"document_id": "<uuid>", "question": "..."}`.
2. `DocumentStore.get` retrieves the pre-indexed document (raising `404` if unknown) so users can ask unlimited questions without re-uploading.
3. `QAService` normalizes whitespace in the question and checks that its DistilBERT token count does not exceed `MAX_QUESTION_TOKENS` (default **64**).
4. **Stage 1 (Retrieval):** `TfidfRetriever.retrieve` transforms the question into the document's TF-IDF vector space, computes cosine similarity against all chunks, and returns up to `TOP_K_CHUNKS` (default **3**) chunks with similarity `> MIN_RETRIEVAL_SIMILARITY` (default `0.0`). If no chunk shares vocabulary (`question_vector.nnz == 0`), the API immediately returns `success: false`, `status: "no_relevant_content"`, and `answer: null`.
5. **Stage 2 (Extractive QA):** `ModelService.predict` runs the DistilBERT QA pipeline on **each retrieved chunk individually** (`handle_impossible_answer=False`, `max_answer_len=30`, `max_seq_len=512`), producing `(answer, score, start, end)`.
6. **Stage 3 (Candidate Selection & Guardrail):** Empty spans are discarded. Among remaining candidates, the highest model score wins (ties broken by better TF-IDF retrieval rank). If the best score is below `CONFIDENCE_THRESHOLD` (default `0.1`), `QAService` withholds the guess (`success: false`, `status: "low_confidence"`, `answer: null`) and returns the closest passage for reference. Otherwise it returns `success: true`, the extracted `answer`, `confidence`, original PDF `page_number`, and a word-trimmed supporting `passage` with `answer_start` and `answer_end` character offsets for UI highlighting.

---

## 8. Installation Instructions

### Prerequisites
- **Python 3.11 or 3.12**
- `pip` and `venv`
- Internet access on the first run to download `distilbert/distilbert-base-cased-distilled-squad` (~260 MB) from the Hugging Face Hub (cached automatically in `~/.cache/huggingface/hub` afterwards)
- Optional: NVIDIA GPU with CUDA drivers for GPU acceleration (CPU works automatically if CUDA is unavailable)

### Setup Commands

```bash
cd backend

# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Optional: copy the example environment file to customize settings
cp .env.example .env

# 4. Optional: regenerate the 4-page sample PDF (already included in samples/)
python scripts/create_sample_pdf.py
```

---

## 9. How to Run the Backend

From the `backend/` directory (with `.venv` activated, or using `.venv/bin/python`):

```bash
python run.py
```

Or invoke Uvicorn directly:

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

At startup the server logs model loading and device selection, for example:
- GPU available: `Model loaded on cuda (NVIDIA GeForce RTX 3050 ...) in 3.30s`
- CPU only: `Model loaded on cpu in ...s`

Once running:
- **API Base URL:** `http://127.0.0.1:8000`
- **Swagger Interactive Docs:** `http://127.0.0.1:8000/docs`
- **ReDoc Documentation:** `http://127.0.0.1:8000/redoc`
- **CORS Allowed Origins (default):** `http://localhost:3000`, `http://localhost:5173`, `http://127.0.0.1:5173` (configurable via `CORS_ORIGINS` in `.env`)

> Tip: Once the model has been cached locally after the first run, you can start offline with `HF_HUB_OFFLINE=1 python run.py`.

---

## 10. API Endpoint Documentation

All error responses follow a uniform JSON structure without leaking stack traces or server file paths:

```json
{
  "success": false,
  "error": "error_code",
  "message": "Human-readable explanation",
  "details": ["optional validation details"]
}
```

### Endpoint Summary

| Method | Endpoint | Purpose | Success Code |
| :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Backend health, model load status, model name, and CUDA/CPU device | `200 OK` |
| `POST` | `/api/documents/upload` | Upload a PDF, extract pages, chunk text, and build TF-IDF index | `201 Created` |
| `POST` | `/api/questions/ask` | Ask a question about a previously uploaded document | `200 OK` |
| `GET` | `/api/documents/{document_id}` | Get document metadata, page counts, chunk count, and status | `200 OK` |
| `DELETE` | `/api/documents/{document_id}` | Remove document from in-memory store and delete uploaded PDF | `200 OK` |

---

### A. Health Check — `GET /api/health`

**Sample Request:**
```bash
curl -s http://127.0.0.1:8000/api/health
```

**Sample Response (`200 OK`):**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_status": "loaded",
  "model_name": "distilbert/distilbert-base-cased-distilled-squad",
  "device": "cuda",
  "device_name": "NVIDIA GeForce RTX 3050 4GB Laptop GPU",
  "model_error": null,
  "documents_in_memory": 0
}
```
If model loading failed at startup, `status` is `"degraded"`, `model_loaded` is `false`, `model_status` is `"failed"`, and `model_error` contains troubleshooting advice.

---

### B. Upload PDF — `POST /api/documents/upload`

Accepts multipart form field `file` (`.pdf`, max 20 MB by default).

**Sample Request:**
```bash
curl -s -X POST http://127.0.0.1:8000/api/documents/upload \
  -F "file=@samples/sample_nlp_document.pdf"
```

**Sample Response (`201 Created`):**
```json
{
  "success": true,
  "document_id": "0e67f4ef-9a9b-4b4a-9b86-5a5290689655",
  "filename": "sample_nlp_document.pdf",
  "num_pages": 4,
  "num_chunks": 4,
  "pages_with_text": 4,
  "empty_pages": [],
  "status": "ready",
  "message": "PDF processed successfully: 4 page(s), 4 chunk(s)."
}
```

**Possible Error Status Codes:**
- `400 empty_file` — zero-byte upload
- `413 file_too_large` — exceeds `MAX_UPLOAD_SIZE_MB` (default 20 MB)
- `415 invalid_file_type` — non-`.pdf` extension, invalid MIME type, or missing `%PDF-` signature
- `422 corrupted_pdf` / `encrypted_pdf` / `empty_pdf` / `no_extractable_text` (scanned PDF with OCR notice)
- `503 model_not_ready` — model is not loaded

---

### C. Ask a Question — `POST /api/questions/ask`

**Sample Request (Replace `<DOCUMENT_ID>` with the UUID from upload):**
```bash
curl -s -X POST http://127.0.0.1:8000/api/questions/ask \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "<DOCUMENT_ID>",
    "question": "Who created DistilBERT?"
  }'
```

**Sample Response — Answer Found (`200 OK`):**
```json
{
  "success": true,
  "status": "answered",
  "document_id": "0e67f4ef-9a9b-4b4a-9b86-5a5290689655",
  "question": "Who created DistilBERT?",
  "answer": "Hugging Face",
  "confidence": 0.9695,
  "source": {
    "page_number": 4,
    "passage": "DistilBERT and Question Answering DistilBERT is a smaller version of BERT created by Hugging Face using knowledge distillation. It has 40 percent fewer parameters than BERT and runs 60 percent faster, while retaining about 97 percent of BERT's language understanding performance. The model distilbert-base-cased-distilled-squad was fine-tuned on the Stanford Question Answering Dataset (SQuAD),",
    "answer_start": 85,
    "answer_end": 97,
    "chunk_id": 3,
    "retrieval_score": 0.2005
  },
  "message": "Answer retrieved successfully"
}
```

**Sample Response — Unanswerable / Off-Topic Question (`200 OK`, `success: false`):**
```json
{
  "success": false,
  "status": "no_relevant_content",
  "document_id": "0e67f4ef-9a9b-4b4a-9b86-5a5290689655",
  "question": "Who won the football world cup?",
  "answer": null,
  "confidence": null,
  "source": null,
  "message": "No relevant passages were found in the document for this question. The document may not contain this information. Try rephrasing the question using keywords that appear in the document."
}
```

Possible `status` values on HTTP `200`:
- `answered` — an answer span was extracted with confidence `>= CONFIDENCE_THRESHOLD`
- `low_confidence` — relevant chunk(s) matched, but DistilBERT's best confidence was below `CONFIDENCE_THRESHOLD`; `answer` is `null` and the closest `source` passage is returned for user inspection
- `no_relevant_content` — no chunk shared meaningful TF-IDF vocabulary with the question
- `no_answer` — relevant chunks matched, but DistilBERT returned only empty spans

**Possible Error Status Codes:** `400 invalid_question` (over `MAX_QUESTION_TOKENS`), `404 document_not_found`, `422 validation_error` (empty/missing question or invalid JSON), `500 model_inference_error`, `503 model_not_ready`.

---

### D. Get Document Details — `GET /api/documents/{document_id}`

**Sample Request:**
```bash
curl -s http://127.0.0.1:8000/api/documents/<DOCUMENT_ID>
```

**Sample Response (`200 OK`):**
```json
{
  "success": true,
  "document_id": "0e67f4ef-9a9b-4b4a-9b86-5a5290689655",
  "filename": "sample_nlp_document.pdf",
  "num_pages": 4,
  "num_chunks": 4,
  "pages_with_text": 4,
  "empty_pages": [],
  "total_characters": 1713,
  "file_size_bytes": 4081,
  "status": "ready",
  "uploaded_at": "2026-11-19T14:20:15.476558Z"
}
```

---

### E. Delete Document — `DELETE /api/documents/{document_id}`

**Sample Request:**
```bash
curl -s -X DELETE http://127.0.0.1:8000/api/documents/<DOCUMENT_ID>
```

**Sample Response (`200 OK`):**
```json
{
  "success": true,
  "document_id": "0e67f4ef-9a9b-4b4a-9b86-5a5290689655",
  "message": "Document 'sample_nlp_document.pdf' was deleted."
}
```

---

## 11. How the TF-IDF Retrieval Process Works

Because DistilBERT can inspect at most 512 tokens per forward pass, a long document cannot be fed to the model all at once. We use Scikit-learn's `TfidfVectorizer` and `cosine_similarity` in `app/services/retrieval_service.py`:

1. **Tokenization & N-grams:** Text is lowercased, split into word tokens (including single-character tokens like `"C"` or `"5"` via `r"(?u)\b\w+\b"`), and indexed with **unigrams and bigrams (`ngram_range=(1, 2)`)** so two-word concepts like `"self attention"` or `"cosine similarity"` match accurately.
2. **Stop-word Filtering:** A lightweight custom stop-word list removes common question/function words (`"what"`, `"who"`, `"is"`, `"the"`, `"according"`, etc.) while intentionally keeping domain words such as `"system"`, `"first"`, `"two"`, or `"name"` that scikit-learn's default list strips. (If a tiny document contains only stop words, the retriever retries without stop-word removal.)
3. **Sublinear TF-IDF Weighting:** Each term in a chunk is weighted by `(1 + log(tf)) * idf`, giving higher importance to terms that appear in that chunk while down-weighting terms scattered across every chunk in the PDF.
4. **Cosine Similarity Ranking:** At query time, the question is transformed using the exact vocabulary and IDF weights fitted on the document. If the question shares zero indexed terms with the document (`nnz == 0`), retrieval returns `[]` immediately. Otherwise, cosine similarity in `[0, 1]` is computed between the normalized question vector and all chunk vectors, and the top `TOP_K_CHUNKS` (default **3**) chunks above `MIN_RETRIEVAL_SIMILARITY` are returned with their original 1-based PDF page numbers.

---

## 12. How DistilBERT Extracts Answers

1. **Token Budget & Chunking:** DistilBERT accepts at most `512` tokens formatted as:
   ```text
   [CLS] <question tokens> [SEP] <context chunk tokens> [SEP]
   ```
   With defaults `CHUNK_SIZE_TOKENS=384` and `MAX_QUESTION_TOKENS=64` plus `3` special tokens (`384 + 64 + 3 = 451 <= 512`), every retrieved chunk is guaranteed to fit without silent truncation. Each chunk overlaps the previous chunk on the same page by `CHUNK_OVERLAP_TOKENS=64` tokens so answers near chunk edges stay intact, and chunks never cross page boundaries.
2. **Span Prediction:** For each retrieved chunk individually, DistilBERT predicts two probability distributions over the chunk's tokens: one for the **start token** and one for the **end token** of the answer span.
3. **Candidate Scoring & Confidence Interpretation:** The Hugging Face QA pipeline outputs `answer`, `start`, `end`, and `score` (obtained from the normalized start and end token probabilities of the highest-scoring valid span). `QAService` compares candidates across all retrieved chunks and chooses the highest-scoring candidate.
   - **Important Caveat on `confidence`:** The returned `confidence` value is the **model's internal confidence score** (derived from start/end softmax logits on that specific chunk), **not a calibrated probability of factual correctness**. Because `distilbert-base-cased-distilled-squad` was trained on SQuAD v1.1 (where every training question had an answer inside the context), the model always attempts to pick the most plausible span in any chunk passed to it—even if an unrelated chunk happened to share a single common keyword with the question. `CONFIDENCE_THRESHOLD` (default `0.1`) helps filter weak predictions, though it is not a guarantee against off-topic high-score spans.

---

## 13. Retrieval vs. Extractive Question Answering (Not Generative RAG)

During faculty evaluation, it is important to distinguish the components clearly:

| Aspect | Document Retrieval (`TfidfRetriever`) | Extractive QA (`DistilBERT`) | Generative RAG *(Not Used Here)* |
| :--- | :--- | :--- | :--- |
| **Role** | Finds *which chunks* of the PDF are most relevant to the question | Finds the *exact start & end span* of the answer inside each candidate chunk | Retrieves passages and prompts an LLM to *write a new answer* |
| **Method** | Sparse TF-IDF lexical vectors + cosine similarity | Transformer encoder predicting token start/end logits | Dense embeddings + autoregressive decoder/LLM (GPT, FLAN-T5, etc.) |
| **Output** | Ranked list of `TextChunk` passages + page numbers | Verbatim substring copied directly from the PDF text | Newly synthesized free-form prose |

**Why this system is NOT Generative RAG:**
There is **no generative LLM** in this backend and **no dense neural embedding retriever**. The system **cannot** summarize whole documents, paraphrase in its own words, combine facts across separate pages into a new sentence, or answer general world-knowledge questions outside the uploaded PDF—every answer is strictly **extracted verbatim** from the retrieved PDF chunk.

---

## 14. Limitations

1. **No OCR Support:** Scanned PDFs or image-only pages contain no selectable text stream; PyMuPDF extracts nothing and the backend explicitly returns `422 no_extractable_text` noting that OCR is not supported.
2. **In-Memory Document Store:** Uploaded document metadata, chunks, and TF-IDF indices live in memory (`DocumentStore`) and are cleared when the server stops or restarts.
3. **Lexical Matching Gap (TF-IDF):** Retrieval depends on shared keywords or 2-word phrases. Synonyms or heavily paraphrased questions without overlapping vocabulary may miss the right chunk.
4. **Extractive Span Constraints:** DistilBERT extracts a single contiguous span up to `MAX_ANSWER_LENGTH_TOKENS` (30 tokens). It cannot answer `"Summarize this paper"`, `"Compare page 2 and page 5"`, or yes/no questions unless `"yes"`/`"no"` appears literally in the text.
5. **SQuAD v1.1 Scoring Behaviour & Uncalibrated Confidence:** Because SQuAD v1.1 assumes an answer exists in the context, if a question shares a minor keyword with an unrelated chunk (e.g., `"according to the text"` or `"language"`), DistilBERT may still return a high confidence score for an unrelated span in that chunk.
6. **Single-Document Scope:** Questions target one `document_id` at a time rather than searching across all uploaded PDFs simultaneously.
7. **Complex PDF Layouts:** Multi-column academic papers or complex tables may read out of visual order in plain `"text"` extraction mode.

---

## 15. Testing Instructions

All tests run from the `backend/` directory using the project's virtual environment.

### 1. Syntax Check
```bash
.venv/bin/python -m compileall -q app tests run.py scripts
```

### 2. Automated Unit & API Tests (Mocked Inference — Fast, Offline)
Ordinary tests mock DistilBERT inference with `FakeModelService` and `SimpleTokenizer` in `tests/conftest.py` so the model is not downloaded on every test run:

```bash
.venv/bin/python -m pytest -v
```

Tests cover:
- PDF file validation (extension, content type, `%PDF-` signature, empty files, oversized files)
- PDF extraction (multi-page text, 1-based page number preservation, empty pages, 0-page PDFs, corrupted/truncated PDFs, password-protected PDFs, OCR-less scanned PDFs)
- Token-aware chunking & overlap boundaries
- TF-IDF retrieval, cosine similarity ranking, stop-word edge cases, and zero-overlap queries
- QA service candidate selection, confidence thresholding, empty/too-long question validation, and passage offsets
- FastAPI REST endpoints (`/api/health`, `/api/documents/upload`, `/api/documents/{id}`, `/api/questions/ask`), CORS rules, OpenAPI `/docs`, and sanitized error handling

### 3. Optional Integration Tests (Real Pretrained DistilBERT Model)
Runs end-to-end tests through FastAPI using the actual `distilbert/distilbert-base-cased-distilled-squad` weights (~260 MB download on first run):

```bash
RUN_INTEGRATION_TESTS=1 .venv/bin/python -m pytest -m integration -v
```
*(Add `HF_HUB_OFFLINE=1` once the model is cached locally.)*

