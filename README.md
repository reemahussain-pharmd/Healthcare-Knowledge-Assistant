# 🏥 Healthcare Knowledge Assistant using RAG

> An intelligent Retrieval-Augmented Generation (RAG) system for clinical research and medical knowledge retrieval — built entirely in Python, runs locally, no cloud required.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4%2B-orange)](https://trychroma.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Architecture](#-architecture)
4. [Project Structure](#-project-structure)
5. [Installation Guide](#-installation-guide)
6. [Configuration](#-configuration)
7. [Running the Application](#-running-the-application)
8. [Usage Guide](#-usage-guide)
9. [Sample Documents](#-sample-documents)
10. [RAG Pipeline](#-rag-pipeline)
11. [Evaluation Metrics](#-evaluation-metrics)
12. [Screenshots](#-screenshots)
13. [Testing](#-testing)
14. [Future Enhancements](#-future-enhancements)
15. [Technical Stack](#-technical-stack)

---

## 🎯 Project Overview

The **Healthcare Knowledge Assistant** is an end-to-end RAG application designed for:

- **Clinical Data Scientists** — Query clinical trial reports and study protocols
- **Medical Researchers** — Extract evidence from research papers and FDA guidance documents
- **Healthcare Professionals** — Access drug information, dosage guidelines, and treatment protocols
- **Regulatory Affairs Specialists** — Navigate ICH-GCP guidelines and pharmacoeconomic studies

The system ingests healthcare documents (PDF, DOCX, TXT), indexes them in a local ChromaDB vector database using Sentence-Transformer embeddings, and answers natural-language questions using an LLM (OpenAI GPT or Google Gemini) grounded strictly in the provided documents.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-format ingestion** | PDF, DOCX, TXT support with automated text extraction |
| **Intelligent cleaning** | Page number removal, unicode normalization, hyphenation repair |
| **Recursive chunking** | Configurable chunk size (default: 1000) and overlap (default: 200) |
| **Local embeddings** | `all-MiniLM-L6-v2` via Sentence-Transformers — no API calls for embedding |
| **Persistent vector store** | ChromaDB stored locally in `data/chroma_db/` |
| **Semantic search** | Cosine similarity vector retrieval with metadata filtering |
| **Metadata filtering** | Filter by disease area, drug name, document type, publication year |
| **Evidence-based answers** | LLM answers grounded in retrieved context with strict hallucination controls |
| **Source citations** | Every answer includes document citations with relevance scores |
| **Confidence scoring** | Aggregate similarity score indicates answer reliability |
| **Medical term highlighting** | Key medical vocabulary highlighted in retrieved chunks |
| **Question history** | Full session history with timestamps and confidence scores |
| **Streamlit UI** | Clean multi-page interface for all functionality |

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Embedding Model    │  all-MiniLM-L6-v2 (local)
│  (Query Encoding)   │
└────────┬────────────┘
         │ Query Vector
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│   ChromaDB          │◄────│  Metadata Filters     │
│   Vector Search     │     │  (disease, drug, type)│
└────────┬────────────┘     └──────────────────────┘
         │ Top-K Chunks
         ▼
┌─────────────────────┐
│  Context Builder    │  Formats chunks + source headers
└────────┬────────────┘
         │ Structured Context
         ▼
┌─────────────────────┐     ┌──────────────────────┐
│  LLM (GPT/Gemini)   │◄────│  Healthcare Prompt    │
│  Answer Generation  │     │  Template             │
└────────┬────────────┘     └──────────────────────┘
         │
         ▼
┌─────────────────────┐
│  Answer + Citations │
│  + Confidence Score │
└─────────────────────┘
```

---

## 📁 Project Structure

```
healthcare-rag/
│
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── .gitignore
├── README.md
│
├── data/
│   ├── documents/                  # Upload healthcare documents here
│   │   ├── sample_clinical_trial.txt
│   │   ├── sample_hypertension_guideline.txt
│   │   └── sample_drug_information.txt
│   ├── chroma_db/                  # ChromaDB persistent storage (auto-created)
│   ├── embeddings/                 # Sentence-Transformer model cache (auto-created)
│   └── document_registry.json      # Document metadata registry (auto-created)
│
├── src/
│   ├── loaders/
│   │   └── document_loader.py      # PDF, DOCX, TXT text extraction
│   ├── preprocessing/
│   │   ├── text_cleaner.py         # Unicode normalization, cleaning pipeline
│   │   └── chunker.py              # Recursive character chunking with overlap
│   ├── embeddings/
│   │   └── embedding_model.py      # Sentence-Transformer wrapper
│   ├── retrieval/
│   │   ├── vector_store.py         # ChromaDB CRUD operations
│   │   └── retriever.py            # Semantic search + context formatting
│   ├── rag/
│   │   └── pipeline.py             # End-to-end RAG orchestration
│   ├── llm/
│   │   └── llm_client.py           # OpenAI / Gemini unified client
│   ├── ui/
│   │   ├── home.py                 # Home page
│   │   ├── upload.py               # Document upload page
│   │   ├── library.py              # Document library page
│   │   ├── ask_questions.py        # Q&A page
│   │   ├── chunks_viewer.py        # Retrieval debugging page
│   │   └── statistics.py           # System statistics page
│   └── utils/
│       ├── helpers.py              # Medical term highlighting, formatting
│       └── document_manager.py     # JSON document registry
│
├── notebooks/
│   └── healthcare_rag_demo.py      # CLI demo script
│
└── tests/
    ├── test_text_cleaner.py
    ├── test_chunker.py
    ├── test_document_loader.py
    └── test_helpers.py
```

---

## 🚀 Installation Guide

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/healthcare-rag.git
cd healthcare-rag
```

### Step 2: Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The first run will download the `all-MiniLM-L6-v2` model (~90 MB) automatically.

### Step 4: Configure API Keys

```bash
# Copy the example file
cp .env.example .env   # Windows: copy .env.example .env
```

Edit `.env` and add your API key:

```env
# For OpenAI
OPENAI_API_KEY=sk-your-key-here

# OR for Google Gemini
GEMINI_API_KEY=your-key-here
```

### Step 5: Verify Installation

```bash
python notebooks/healthcare_rag_demo.py
```

---

## ⚙️ Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key (Option A) |
| `GEMINI_API_KEY` | — | Google Gemini API key (Option B) |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | Override OpenAI model |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Override Gemini model |

**Chunking parameters** (configurable in the Upload UI):
- Default chunk size: **1000 characters**
- Default chunk overlap: **200 characters**

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

Then open your browser at **http://localhost:8501**

---

## 📖 Usage Guide

### 1. Upload Documents

1. Navigate to **Upload Documents** in the sidebar
2. Select a PDF, DOCX, or TXT healthcare document
3. Fill in the metadata:
   - Document title
   - Document type (Clinical Trial Report, Medical Guideline, etc.)
   - Disease area
   - Drug/compound name
   - Publication year
4. Optionally adjust chunk size and overlap
5. Click **Upload & Process Document**

### 2. Ask Questions

1. Navigate to **Ask Questions**
2. Optionally apply metadata filters (document type, disease area, drug, year)
3. Type your question or select from example questions
4. Click **Ask**

**Example questions:**
- *"What are the inclusion criteria in this clinical trial?"*
- *"What adverse effects are reported for Metformin?"*
- *"Summarize the treatment guidelines for hypertension."*
- *"What endpoints were used in the diabetes study?"*
- *"What is the recommended dosage of Atorvastatin?"*

### 3. Explore Retrieved Chunks

Use the **Chunks Viewer** to:
- Inspect exactly which passages were retrieved for any query
- View relevance scores and metadata
- Debug retrieval quality
- See highlighted medical terminology

### 4. Manage Documents

The **Document Library** allows you to:
- Browse all indexed documents with full metadata
- Filter by type, disease area, or search by name
- Remove documents from the knowledge base

---

## 📄 Sample Documents

Three sample healthcare documents are included in `data/documents/`:

| File | Type | Disease Area | Drug |
|------|------|-------------|------|
| `sample_clinical_trial.txt` | Clinical Trial Report | Diabetes | Metformin XR |
| `sample_hypertension_guideline.txt` | Medical Guideline | Cardiovascular | Multiple |
| `sample_drug_information.txt` | Drug Information | Cardiovascular | Atorvastatin |

Use the **CLI demo** to auto-ingest them:

```bash
python notebooks/healthcare_rag_demo.py
```

---

## 🔄 RAG Pipeline

```
Document Ingestion:
─────────────────────────────────────────────────────
  File (PDF/DOCX/TXT)
      │
      ▼ DocumentLoader
  Raw Text
      │
      ▼ TextCleaner
  Cleaned Text (page numbers removed, unicode normalized)
      │
      ▼ RecursiveChunker (size=1000, overlap=200)
  List of Chunks
      │
      ▼ EmbeddingModel (all-MiniLM-L6-v2)
  Chunk Embeddings (384-dim vectors)
      │
      ▼ VectorStore (ChromaDB)
  Persisted in local chroma_db/

Question Answering:
─────────────────────────────────────────────────────
  User Question
      │
      ▼ EmbeddingModel
  Query Vector (384-dim)
      │
      ▼ VectorStore.query() + Metadata Filters
  Top-K Similar Chunks (with cosine scores)
      │
      ▼ Retriever.format_context()
  Structured Context String (with [Source N] headers)
      │
      ▼ LLMClient (GPT-3.5-turbo / Gemini 1.5 Flash)
  Generated Answer
      │
      ▼ Response Object
  Answer + Citations + Confidence Score + Timing
```

---

## 📊 Evaluation Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Confidence Score** | Mean cosine similarity of retrieved chunks | > 0.6 |
| **Retrieval Precision** | % relevant chunks in top-K | > 80% |
| **Response Time** | End-to-end latency | < 10 seconds |
| **Tokens Used** | LLM tokens per response | Varies by question |
| **Chunks Retrieved** | Number of passages used | 3–7 typical |
| **Answer Grounding** | Proportion of answer sourced from context | 100% (by design) |

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_chunker.py -v

# Run with coverage
pip install pytest-cov
python -m pytest tests/ --cov=src --cov-report=term-missing
```

---

## 🔮 Future Enhancements

### Short-term
- [ ] **Table extraction from PDFs** — structured data from clinical trial tables
- [ ] **Multi-document comparison** — compare findings across multiple trials
- [ ] **Advanced chunking** — semantic/sentence-boundary-aware chunking
- [ ] **Reranking** — cross-encoder reranking for improved precision

### Medium-term
- [ ] **Hybrid search** — combine BM25 keyword search with vector search
- [ ] **RAGAS evaluation** — automated faithfulness, answer relevance, context precision metrics
- [ ] **Summarization pipeline** — auto-summarize uploaded documents
- [ ] **Entity extraction** — auto-populate drug name and disease area from content
- [ ] **Citation linking** — link cited passages back to source PDF page numbers

### Long-term
- [ ] **Fine-tuned biomedical embeddings** — BioBERT, PubMedBERT
- [ ] **Multi-modal support** — extract text from figures and medical images
- [ ] **Conversational memory** — multi-turn conversations with context retention
- [ ] **Regulatory intelligence** — automated comparison against current regulatory guidance

---

## 🛠️ Technical Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | 1.32+ |
| **Embedding Model** | Sentence-Transformers (`all-MiniLM-L6-v2`) | 2.7+ |
| **Vector Database** | ChromaDB | 0.4+ |
| **LLM (Option A)** | OpenAI GPT-3.5-turbo / GPT-4 | 1.20+ |
| **LLM (Option B)** | Google Gemini 1.5 Flash | 0.5+ |
| **PDF Parsing** | pdfplumber / PyMuPDF | 0.10+ |
| **DOCX Parsing** | python-docx | 1.1+ |
| **Vector Math** | NumPy | 1.26+ |
| **ML Framework** | PyTorch | 2.2+ |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⚠️ Disclaimer

This system is intended for **research and educational purposes only**. It does not constitute
medical advice. Clinical decisions must always be made by qualified healthcare professionals
based on individual patient assessment. The AI-generated answers may contain errors — always
verify critical information against primary sources.

---

## 🤝 Contributing

Contributions welcome! Please open an issue first to discuss proposed changes.

```bash
# Fork → clone → create branch → commit → push → open PR
git checkout -b feature/your-feature-name
```

---

*Built with ❤️ for the Clinical Data Science and Healthcare AI community.*
