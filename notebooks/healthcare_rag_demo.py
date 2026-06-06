"""
Healthcare RAG System - Command-Line Demo Script

This script demonstrates the full pipeline without the Streamlit UI.
Run:  python notebooks/healthcare_rag_demo.py

Requirements: All packages from requirements.txt must be installed.
              Copy .env.example to .env and add your API key.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from src.utils.helpers import setup_logging
setup_logging("INFO")

from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.rag.pipeline import HealthcareRAGPipeline
from src.utils.document_manager import DocumentManager

print("=" * 70)
print("  Healthcare Knowledge Assistant - Pipeline Demo")
print("=" * 70)

# 1. Initialise components
print("\n[1] Initialising components...")
embedding_model = EmbeddingModel(model_name="all-MiniLM-L6-v2", cache_dir="data/embeddings")
vector_store = VectorStore(persist_dir="data/chroma_db")
doc_manager = DocumentManager()

# Optionally initialise LLM
llm = None
try:
    from src.llm.llm_client import LLMClient
    llm = LLMClient(provider="auto")
    print(f"   LLM: {llm.provider} / {llm.model}")
except EnvironmentError:
    print("   LLM: Not configured (set OPENAI_API_KEY or GEMINI_API_KEY in .env)")

pipeline = HealthcareRAGPipeline(
    embedding_model=embedding_model,
    vector_store=vector_store,
    llm_client=llm,
)

# 2. Ingest sample documents
SAMPLE_DOCS = [
    {
        "path": "data/documents/sample_clinical_trial.txt",
        "metadata": {
            "title": "Phase III Metformin XR Clinical Trial Report",
            "document_type": "Clinical Trial Report",
            "disease_area": "Diabetes / Metabolic",
            "drug_name": "Metformin",
            "publication_year": "2023",
        },
    },
    {
        "path": "data/documents/sample_hypertension_guideline.txt",
        "metadata": {
            "title": "IHS Management of Hypertension Guidelines 2023",
            "document_type": "Medical Guideline",
            "disease_area": "Cardiovascular / Hypertension",
            "drug_name": "N/A",
            "publication_year": "2023",
        },
    },
    {
        "path": "data/documents/sample_drug_information.txt",
        "metadata": {
            "title": "Atorvastatin (Lipitor) Drug Information Document",
            "document_type": "Drug Information Document",
            "disease_area": "Cardiovascular",
            "drug_name": "Atorvastatin",
            "publication_year": "2023",
        },
    },
]

print(f"\n[2] Ingesting {len(SAMPLE_DOCS)} sample documents...")
for doc in SAMPLE_DOCS:
    if not os.path.exists(doc["path"]):
        print(f"   SKIP (not found): {doc['path']}")
        continue
    if any(d.get("title") == doc["metadata"]["title"] for d in doc_manager.list_all()):
        print(f"   SKIP (already ingested): {doc['metadata']['title']}")
        continue

    summary = pipeline.ingest_document(doc["path"], doc["metadata"])
    doc_manager.register(summary["document_id"], doc["metadata"], summary)
    print(f"   Ingested: {doc['metadata']['title']} ({summary['total_chunks']} chunks)")

# 3. Show system stats
print("\n[3] System Statistics:")
stats = pipeline.get_system_stats()
print(f"   Total documents : {stats['total_documents']}")
print(f"   Total chunks    : {stats['total_chunks']}")
print(f"   Embedding model : {stats['embedding_model']}")
print(f"   LLM provider    : {stats['llm_provider']}")

# 4. Demo retrieval
DEMO_QUESTIONS = [
    "What are the inclusion criteria in the clinical trial?",
    "What adverse effects are reported for Metformin?",
    "What is the recommended dosage of Atorvastatin?",
    "Summarize the treatment guidelines for hypertension.",
]

print("\n[4] Demo Retrieval (semantic search without LLM):")
for q in DEMO_QUESTIONS[:2]:
    print(f"\n  Q: {q}")
    chunks = pipeline.retriever.retrieve(q, top_k=3)
    for i, c in enumerate(chunks, 1):
        meta = c.get("metadata", {})
        print(f"    [{i}] Score: {c['score']:.4f} | {meta.get('title','?')} | {c['text'][:120]}...")

# 5. Demo Q&A if LLM available
if llm:
    print("\n[5] Demo Q&A with LLM:")
    for q in DEMO_QUESTIONS[:2]:
        print(f"\n  Q: {q}")
        result = pipeline.ask(q, top_k=5)
        print(f"  A: {result['answer'][:400]}...")
        print(f"  Confidence: {result['confidence_score']:.1%} | Time: {result['response_time_s']}s")
else:
    print("\n[5] Skipping LLM Q&A (no API key configured).")

print("\n" + "=" * 70)
print("  Demo complete. Run 'streamlit run app.py' for the full UI.")
print("=" * 70)
