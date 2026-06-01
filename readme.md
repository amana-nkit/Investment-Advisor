# 📊 AI Investment Advisor

A memory-aware, RAG-powered investment proposal generator built with Streamlit, LangChain, LangGraph, and GPT-4.1. Upload a client's financial profile PDF and receive a structured, SEBI-compliant investment proposal — with full cross-session memory so every interaction builds on the last.

---

## Features

- **Structured proposal generation** — GPT-4.1 with `with_structured_output()` produces validated `InvestmentProposal` objects (Pydantic v2), including asset allocation, risk assessment, recommended products, and SEBI disclaimer
- **4-layer memory system** — session cache, conversation buffer, persistent client profiles (SQLite/PostgreSQL), and semantic ChromaDB embeddings survive across sessions
- **Real LangGraph pipeline** — 6-node DAG: `load_memory → retrieve → enrich → generate → validate → persist`
- **Multi-query RAG retrieval** — 5 targeted queries instead of one generic search, deduplicated for higher recall
- **Secure PDF ingestion** — pure-Python magic-byte validation, 10 MB size cap, guaranteed temp-file cleanup
- **Professional PDF export** — ReportLab Paragraph/Table API with a proper A4 layout
- **Transactional email** — SendGrid with `email-validator` (no raw SMTP, no header injection)
- **DPDP Act compliance** — consent logging and one-call full memory erasure (`delete_all_memory()`)
- **Multi-layer guardrails** — input validation, output schema enforcement, hallucination containment, regulatory disclaimer injection, and rate-limit hooks at every pipeline stage
- **Docker-ready** — multi-stage build, non-root user, health check, PostgreSQL via Compose

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     Streamlit UI (ui/app.py)            │
└────────────────────────┬────────────────────────────────┘
                         │
              ┌──────────▼──────────┐
              │   LangGraph Graph   │  workflow/graph.py
              │                     │
              │  load_memory        │◄─── L3 (profile) + L4 (recall)
              │      ↓              │
              │  retrieve           │◄─── L4 (vector store)
              │      ↓              │
              │  enrich             │◄─── Web context (Zerodha Varsity)
              │      ↓              │
              │  generate           │◄─── L2 (chat history) + all context
              │      ↓              │
              │  validate_output    │◄─── Pydantic schema check
              │      ↓              │
              │  persist_memory     │───► L3 (save to DB) + L4 (embed)
              └─────────────────────┘
```

### Memory layers

| Layer | Backend | Scope | What is stored |
|-------|---------|-------|---------------|
| **L1** | `st.session_state` | Page session | Vector store reference, file hash, current proposal |
| **L2** | `ConversationBufferWindowMemory` | Page session (10 turns) | Human / AI turns for follow-up questions |
| **L3** | SQLite / PostgreSQL (SQLAlchemy) | Permanent | Customer profile, proposal history (JSON), consent log |
| **L4** | ChromaDB (persistent) | Permanent | PDF chunks + past proposal embeddings per customer |

---

## Guardrails

Guardrails are enforced at every stage of the pipeline — from the moment a file is uploaded through to the final PDF sent to the client. They are grouped into four categories: **input**, **output**, **memory**, and **compliance**.

### Input guardrails

| Guardrail | Where enforced | What it prevents |
|-----------|---------------|-----------------|
| PDF magic-byte check | `loader/loaders.py` → `_is_pdf()` | Non-PDF files (executables, Office docs) disguised with a `.pdf` extension |
| File size cap (10 MB) | `loader/loaders.py` → `validate_and_read_pdf()` | Oversized uploads causing memory exhaustion or slow embedding |
| Temp file cleanup | `loader/loaders.py` → `try/finally` + `os.unlink()` | Customer financial data persisting on disk after processing |
| Email address validation | `utilities/email_service.py` → `_validate_recipient()` | SMTP header injection via malformed recipient strings |
| Customer ID scoping | `memory/memory_manager.py` | Cross-customer data leakage — every vector store, DB query, and ChromaDB collection is keyed to `customer_id` |
| `extra="ignore"` in Settings | `core/config.py` | Unknown `.env` keys crashing startup; enforces a clean, declared config surface |

### Output guardrails

| Guardrail | Where enforced | What it prevents |
|-----------|---------------|-----------------|
| Structured output schema | `workflow/graph.py` → `llm.with_structured_output(InvestmentProposal)` | Free-form text responses that can't be rendered, validated, or stored reliably |
| Pydantic field validation | `models/schemas.py` → `InvestmentProposal` | Invalid values — e.g. allocation percentages outside 0–100, horizon < 1 year, missing required fields |
| Allocation sum check | `models/schemas.py` → `allocations_sum_to_100` validator | Asset allocations that don't add up to 100%, which would produce a misleading proposal |
| `validate_output` node | `workflow/graph.py` | A second Pydantic re-check after generation; catches any schema drift introduced by LLM re-formatting |
| LLM error handling | `workflow/graph.py` → `generate_node` try/except | Unhandled LLM exceptions propagating to the UI as raw tracebacks |
| Mandatory SEBI disclaimer | `models/schemas.py` → `InvestmentProposal.disclaimer` | Proposals reaching clients without the required regulatory caveat; the disclaimer is a non-nullable field with a default value |
| ReportLab Paragraph API | `utilities/pdf_export.py` | Word-split and Unicode-broken PDFs caused by the original `line[:100]` slicing |

### Memory guardrails

| Guardrail | Where enforced | What it prevents |
|-----------|---------------|-----------------|
| File hash deduplication | `memory/memory_manager.py` → `is_file_already_ingested()` | The same PDF being re-embedded on every button click, causing duplicate chunks and wasted API spend |
| `@st.cache_resource` on LLM + embeddings | `workflow/graph.py`, `memory/memory_manager.py` | A new HTTP client being constructed on every Streamlit rerun |
| Per-customer ChromaDB collection | `memory/memory_manager.py` → `_get_chroma_store()` | One customer's documents leaking into another customer's retrieval results |
| Consent record before persistence | `models/db_models.py` → `ConsentRecord` | Financial data being stored without an auditable record of user consent (DPDP Act) |
| Full erasure path | `memory/memory_manager.py` → `delete_all_memory()` | Inability to honour right-to-erasure requests — deletes L3 rows and L4 ChromaDB collection atomically |

### Compliance guardrails

| Guardrail | Detail |
|-----------|--------|
| SEBI disclaimer (mandatory) | Hard-coded as a non-nullable field in `InvestmentProposal`; cannot be removed by prompt engineering |
| Consent logging | Every data ingestion event records `customer_id`, timestamp, IP address, and purpose in `consent_records` |
| No raw credentials | `SendGridAPIClient` replaces `smtplib`; all keys are loaded via `pydantic-settings`, never via `os.getenv()` directly |
| Gitignored secrets | `.env` is in `.gitignore`; `.env.example` ships instead |
| Non-root Docker container | The `app` OS user has no write access outside `/app/logs` and `/app/chroma_db` |

### Adding custom guardrails

The `validate_output` node in `workflow/graph.py` is the designed extension point. Add any additional checks there before the `persist_memory` node runs:

```python
def validate_output_node(state: ProposalState) -> ProposalState:
    # existing Pydantic check ...

    # Example: block proposals with expected return > 30% (hallucination signal)
    proposal = state.get("proposal_final", {})
    if proposal.get("expected_annual_return_pct", 0) > 30:
        state["validation_errors"].append(
            "Expected return exceeds 30% — likely a hallucination. Please review."
        )

    # Example: require at least 3 asset classes
    if len(proposal.get("asset_allocation", [])) < 3:
        state["validation_errors"].append(
            "Proposal must include at least 3 distinct asset classes."
        )

    return state
```

If `validation_errors` is non-empty, `persist_memory` is skipped and the error is surfaced in the UI — the proposal is never saved to L3 or L4.

---

## Project structure

```
investment_advisor/
├── core/
│   └── config.py               Typed settings via pydantic-settings (extra="ignore")
├── models/
│   ├── schemas.py              Pydantic v2 — InvestmentProposal, AssetAllocation, etc.
│   └── db_models.py            SQLAlchemy ORM — CustomerProfile, ProposalRecord, ConsentRecord
├── memory/
│   └── memory_manager.py       MemoryManager — unified L1–L4 interface
├── loader/
│   └── loaders.py              PDF validation (magic-byte, size cap), chunking, web loader
├── workflow/
│   ├── state.py                LangGraph ProposalState TypedDict
│   └── graph.py                6-node compiled LangGraph pipeline
├── utilities/
│   ├── logger.py               Centralised structured logger
│   ├── pdf_export.py           ReportLab A4 PDF with tables and proper word-wrap
│   └── email_service.py        SendGrid + email-validator
├── ui/
│   └── app.py                  Streamlit thin shell — all logic delegated to workflow
├── .env.example                Environment variable template
├── requirements.txt
├── Dockerfile                  Multi-stage, non-root, health check
└── docker-compose.yml          App + PostgreSQL services
```

---

## Quick start

### 1. Clone and set up environment

```bash
git clone https://github.com/your-org/investment-advisor.git
cd investment-advisor
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
OPENAI_API_KEY=sk-proj-...        # required — must be spelled exactly OPENAI_API_KEY
SENDGRID_API_KEY=SG....           # optional — only needed for email sending
EMAIL_FROM=advisor@yourdomain.com
DATABASE_URL=sqlite:///./investment_advisor.db   # or postgresql+psycopg2://...
CHROMA_PATH=./chroma_db
```

> **Common mistake:** The key must be `OPENAI_API_KEY`, not `OPEN_API_KEY`. A typo here causes a Pydantic `ValidationError` on startup. The `extra="ignore"` setting in `config.py` prevents a crash but the correct spelling is still required for the OpenAI client to authenticate.

### 3. Run

```bash
streamlit run ui/app.py
```

App opens at `http://localhost:8501`.

---

## Docker

### Build and run (SQLite, single container)

```bash
docker build -t investment-advisor .
docker run -p 8501:8501 --env-file .env investment-advisor
```

### Run with PostgreSQL (recommended for production)

```bash
# set DB_PASSWORD in your shell or .env first
docker-compose up --build
```

This starts:
- `db` — PostgreSQL 16 with a health check
- `app` — Streamlit app connected to the database

---

## Usage

1. **Enter a Customer ID** in the sidebar (e.g. `cust_001`). All memory is scoped to this ID.
2. **Upload a PDF** — the client's financial profile. The file is validated and embedded into ChromaDB (L4).
3. **Click "Generate Investment Proposal"** — the LangGraph pipeline runs, loads prior memory, retrieves relevant chunks, calls GPT-4.1 with structured output, and saves the result back to L3 + L4.
4. **Download the PDF** or **send via email** from the output section.
5. **Ask follow-up questions** in the chat input — conversation history (L2) is injected into every subsequent generation so the model remembers context like "make it more conservative" or "what if I retire in 5 years?".

On the **next session for the same Customer ID**, the system automatically:
- Loads the customer profile from L3 (PostgreSQL)
- Loads prior proposals from L3
- Performs a semantic recall from L4 (ChromaDB) — the model can reference what was discussed in past sessions

---

## Environment variable reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ | — | OpenAI API key |
| `DATABASE_URL` | — | `sqlite:///./investment_advisor.db` | SQLAlchemy connection string |
| `CHROMA_PATH` | — | `./chroma_db` | ChromaDB persistence directory |
| `SENDGRID_API_KEY` | — | `""` | SendGrid key for email sending |
| `EMAIL_FROM` | — | `advisor@yourdomain.com` | Sender address |
| `LLM_MODEL` | — | `gpt-4.1` | OpenAI model name |
| `EMBEDDING_MODEL` | — | `text-embedding-3-small` | Embedding model |
| `MAX_PDF_MB` | — | `10` | Maximum upload size in MB |
| `CHUNK_SIZE` | — | `1000` | RAG chunk size (tokens) |
| `CHUNK_OVERLAP` | — | `200` | RAG chunk overlap (tokens) |
| `RETRIEVAL_TOP_K` | — | `5` | Chunks retrieved per query |
| `LOG_LEVEL` | — | `INFO` | Logging level |

---

## Security notes

| Risk | Mitigation |
|------|-----------|
| Credential exposure | All secrets in `.env`, gitignored. `extra="ignore"` in Pydantic settings silently drops unknown keys instead of crashing. |
| Malicious file upload | Pure-Python `%PDF-` magic-byte check + 10 MB size cap in `loaders.py` |
| Email header injection | `email-validator` normalises and validates recipient before any send call |
| Temp file data leak | `try/finally` with `os.unlink()` guarantees cleanup even on exception |
| Data retention | `MemoryManager.delete_all_memory()` erases all L3 rows and L4 collection for a customer |
| Container hardening | Non-root `app` user, multi-stage Docker build, health check endpoint |

---

## Compliance

This application generates AI-assisted investment proposals. Before deploying to end users in India:

- **SEBI (Investment Advisers) Regulations, 2013** — distributing investment advice commercially requires SEBI registration. All generated proposals include a mandatory disclaimer enforced at the schema level.
- **DPDP Act, 2023 (India)** — explicit consent is logged in the `consent_records` table before any financial data is persisted. Provide users a data export and deletion path via `delete_all_memory()`.

---

## Development

### Run tests

```bash
pytest tests/ -v
```

### Linting

```bash
pip install ruff
ruff check .
```

### Resetting memory for a customer

Use the **"Erase all memory"** button in the sidebar, or call directly:

```python
from memory.memory_manager import MemoryManager
MemoryManager("cust_001").delete_all_memory()
```

---

## Tech stack

| Component | Library |
|-----------|---------|
| UI | Streamlit 1.35+ |
| LLM | GPT-4.1 via `langchain-openai` |
| Orchestration | LangGraph 0.1+ |
| RAG framework | LangChain 0.2+ |
| Vector store | ChromaDB 0.5+ (persistent) |
| Embeddings | `text-embedding-3-small` |
| L3 memory | SQLAlchemy 2.0 + SQLite / PostgreSQL |
| PDF parsing | pypdf + PyPDFLoader |
| PDF export | ReportLab 4.2+ |
| Email | SendGrid 6.11+ |
| Validation | Pydantic v2 + pydantic-settings |
| Containerisation | Docker + Docker Compose |
