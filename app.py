"""
Healthcare Knowledge Assistant – Streamlit Application Entry Point.

Run with:  streamlit run app.py
"""

import os
import sys
import logging

# Ensure src/ is importable from the project root
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

from src.rag.pipeline import HealthcareRAGPipeline
from src.embeddings.embedding_model import EmbeddingModel
from src.retrieval.vector_store import VectorStore
from src.utils.helpers import setup_logging
from src.utils.document_manager import DocumentManager

from src.ui.home import render_home
from src.ui.upload import render_upload
from src.ui.library import render_library
from src.ui.ask_questions import render_ask_questions
from src.ui.chunks_viewer import render_chunks_viewer
from src.ui.statistics import render_statistics
from src.ui.chat import render_chat

# Load .env
load_dotenv()
setup_logging("INFO")
logger = logging.getLogger(__name__)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Healthcare Knowledge Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp { background-color: #f5f7fa; }
        [data-testid="stSidebar"] { background-color: #1a2942; }
        [data-testid="stSidebar"] * { color: #e8edf5 !important; }
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stSlider label { color: #c0cfe0 !important; }
        .stButton > button { border-radius: 8px; font-weight: 600; }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #1a6eb5, #1a8c50);
            border: none; color: white;
        }
        .metric-container { background: white; border-radius: 10px; padding: 1rem; }
        h1, h2, h3 { color: #1a2942; }
        .stExpander { border: 1px solid #dee2e6; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached Resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def get_embedding_model():
    return EmbeddingModel(
        model_name="all-MiniLM-L6-v2",
        cache_dir="data/embeddings",
    )


@st.cache_resource(show_spinner="Connecting to vector store…")
def get_vector_store():
    return VectorStore(persist_dir="data/chroma_db")


@st.cache_resource(show_spinner="Initialising pipeline…")
def get_pipeline(_embedding_model, _vector_store):
    from src.llm.llm_client import LLMClient

    llm = None
    try:
        llm = LLMClient(provider="auto")
        logger.info(f"LLM initialised: {llm.provider} / {llm.model}")
    except EnvironmentError as e:
        logger.warning(f"LLM not available: {e}")

    return HealthcareRAGPipeline(
        embedding_model=_embedding_model,
        vector_store=_vector_store,
        llm_client=llm,
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
    )


@st.cache_resource
def get_doc_manager():
    return DocumentManager(registry_path="data/document_registry.json")


# ── Initialise shared resources ────────────────────────────────────────────────
embedding_model = get_embedding_model()
vector_store = get_vector_store()
pipeline = get_pipeline(embedding_model, vector_store)
doc_manager = get_doc_manager()


# ── Sidebar Navigation ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0 0.5rem 0;">
            <div style="font-size:3rem;">🏥</div>
            <h2 style="margin:0; font-size:1.1rem; font-weight:700;">
                Healthcare Knowledge<br>Assistant
            </h2>
            <p style="font-size:0.75rem; color:#90a8c0; margin:0.3rem 0 0 0;">
                RAG-Powered Clinical Research
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    pages = {
        "🏠 Home": "Home",
        "💬 Chat": "Chat",
        "📄 Upload Documents": "Upload",
        "📚 Document Library": "Library",
        "❓ Ask Questions": "Ask",
        "🔬 Chunks Viewer": "Chunks",
        "📊 Statistics": "Statistics",
    }

    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    for label, key in pages.items():
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key

    # Quick stats in sidebar
    st.markdown("---")
    total_docs = doc_manager.count()
    total_chunks = vector_store.count()
    st.markdown(
        f"""
        <div style="text-align:center; font-size:0.8rem; color:#90a8c0;">
            <b>Docs:</b> {total_docs} &nbsp;|&nbsp; <b>Chunks:</b> {total_chunks}
        </div>
        """,
        unsafe_allow_html=True,
    )

    llm_status = "✅ Connected" if pipeline.llm_client else "❌ No API Key"
    st.markdown(
        f"""
        <div style="text-align:center; font-size:0.8rem; color:#90a8c0; margin-top:0.3rem;">
            <b>LLM:</b> {llm_status}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.7rem; color:#607080;'>"
        "Built with Streamlit, ChromaDB<br>& Sentence-Transformers"
        "</p>",
        unsafe_allow_html=True,
    )


# ── Page Routing ───────────────────────────────────────────────────────────────
page = st.session_state.get("page", "Home")

if page == "Home":
    render_home()
elif page == "Chat":
    render_chat(pipeline, doc_manager)
elif page == "Upload":
    render_upload(pipeline, doc_manager)
elif page == "Library":
    render_library(doc_manager, vector_store)
elif page == "Ask":
    render_ask_questions(pipeline, doc_manager)
elif page == "Chunks":
    render_chunks_viewer(pipeline, doc_manager)
elif page == "Statistics":
    render_statistics(pipeline, doc_manager)
