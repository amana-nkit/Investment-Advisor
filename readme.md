# 💰 AI Investment Advisor (LangGraph + RAG)

An AI-powered **Investment Proposal Generator** that combines:

- 📄 Customer financial data (PDF)
- 🌐 Market insights from trusted sources
- 🧠 RAG (Retrieval-Augmented Generation)
- 🔄 LangGraph workflow orchestration
- ✅ Pydantic data validation
- 📚 Vector DB (ChromaDB)

---

## 🚀 Features

- 📤 Upload customer financial profile (PDF)
- 🌍 Fetch real-world investment insights (Zerodha, etc.)
- 🔍 Hybrid RAG (PDF + Web Data)
- 🧠 AI-generated investment proposals
- ✅ Document validation using Pydantic
- 📥 Export proposal as PDF
- 📧 Email proposal to client
- 📁 Clean logging system (stored in `/logs`)

---

## 🏗️ Project Structure

src/
├── ui/
│    └── app.py                # Streamlit UI
│
├── loader/
│    └── loaders.py           # PDF + Web data ingestion
│
├── vector/
│    └── vector_store.py      # ChromaDB vector store
│
├── workflow/
│    └── graph.py             # LangGraph workflow (retrieve → validate → generate)
│
├── rag/
│    └── rag_pipeline.py      # LLM response generation
│
├── models/
│    └── document_schema.py   # Pydantic validation schema
│
├── utilities/
│    ├── utils.py             # PDF + Email helpers
│    └── logger.py            # Logging system

## ⚙️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **LLM:** OpenAI GPT
- **Frameworks:** LangChain, LangGraph
- **Vector DB:** ChromaDB
- **Validation:** Pydantic
- **Logging:** Python logging (file-based)

---

## 🧩 Architecture Overview

PDF Upload + Web Data
↓
Loaders (PDF + Web)
↓
Chunking & Embeddings
↓
ChromaDB Vector Store
↓
LangGraph Pipeline:
retrieve → validate → generate
↓
Pydantic Validation Layer
↓
LLM (OpenAI)
↓
Investment Proposal

## ✅ Data Validation (Pydantic)

All retrieved documents are validated before use:

```python
class InvestmentDocument(BaseModel):
    content: str
    source: str
    doc_type: str

## Logging

Logs are stored in: /logs/app.log

Includes:

Data loading steps
API calls
Errors
Retrieval logs
Proposal generation status

## Example Curl

```bash
# Create and Activate Virtual environment

python -m venv .venv
 
activate:
.venv\Scripts\activate.bat

# Install all required python packages

pip install -r requirements.txt

python -m streamlit run main.py

# Docker login

 docker login -u amanankit  

# Build & Run Locally

# Build
docker build -t ai-investment-app .

# Run
docker run -p 8501:8501 ai-investment-app

http://localhost:8501

# Download Azure CLI

https://aka.ms/installazurecliwindows

Run the .msi file
Follow default steps (Next → Finish)
