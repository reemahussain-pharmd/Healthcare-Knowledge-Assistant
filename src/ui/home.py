"""
Streamlit Home page.
"""

import streamlit as st


def render_home():
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size:2.8rem; font-weight:700; color:#1a6eb5;">
                Healthcare Knowledge Assistant
            </h1>
            <p style="font-size:1.2rem; color:#555; max-width:700px; margin:auto;">
                An intelligent Retrieval-Augmented Generation (RAG) system for
                clinical research and medical knowledge retrieval.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div style="background:#eaf4fb; border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:2.5rem;">📄</div>
                <h3 style="color:#1a6eb5;">Upload Documents</h3>
                <p style="color:#555; font-size:0.9rem;">
                    Upload PDFs, Word documents, or text files including clinical trial reports,
                    medical guidelines, drug information sheets, and research papers.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
            <div style="background:#eaf7ee; border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:2.5rem;">🔍</div>
                <h3 style="color:#1a8c50;">Semantic Search</h3>
                <p style="color:#555; font-size:0.9rem;">
                    Ask questions in plain language. The system retrieves the most relevant
                    passages using AI embeddings and vector similarity search.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            """
            <div style="background:#fef8ea; border-radius:12px; padding:1.2rem; text-align:center;">
                <div style="font-size:2.5rem;">💊</div>
                <h3 style="color:#b58c1a;">Evidence-Based Answers</h3>
                <p style="color:#555; font-size:0.9rem;">
                    Receive AI-generated answers grounded in your documents,
                    with source citations and confidence scores.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Supported Document Types")
    doc_types = [
        ("🧪", "Clinical Trial Reports"),
        ("📋", "Clinical Study Protocols"),
        ("📖", "Medical Guidelines"),
        ("💉", "Drug Information Documents"),
        ("🔬", "Research Papers"),
        ("🏛️", "FDA Guidance Documents"),
        ("📜", "ICH-GCP Guidelines"),
        ("🩺", "Patient Information Leaflets"),
        ("💰", "Pharmacoeconomic Research Papers"),
    ]
    cols = st.columns(3)
    for i, (icon, label) in enumerate(doc_types):
        with cols[i % 3]:
            st.markdown(f"{icon} {label}")

    st.markdown("---")
    st.markdown("### How It Works")

    steps = [
        ("1️⃣", "Upload Documents", "PDF, DOCX, or TXT healthcare documents"),
        ("2️⃣", "Text Extraction & Cleaning", "Automated cleaning, normalization, and chunking"),
        ("3️⃣", "Embedding Generation", "Sentence-Transformer model (all-MiniLM-L6-v2)"),
        ("4️⃣", "Vector Storage", "ChromaDB persistent local vector database"),
        ("5️⃣", "Semantic Retrieval", "Top-K similarity search with optional metadata filters"),
        ("6️⃣", "Answer Generation", "LLM (GPT / Gemini) with evidence-based citations"),
    ]

    cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style="background:#f8f9fa; border-radius:10px; padding:1rem; margin-bottom:0.8rem;">
                    <div style="font-size:1.8rem; text-align:center;">{num}</div>
                    <h4 style="text-align:center; color:#1a6eb5; margin:0.3rem 0;">{title}</h4>
                    <p style="text-align:center; font-size:0.85rem; color:#666;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.info(
        "**Getting Started:** Navigate to **Upload Documents** in the sidebar to add your "
        "healthcare documents, then use **Ask Questions** to query them."
    )

    st.markdown(
        "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
        "Healthcare Knowledge Assistant • Built with Streamlit, ChromaDB & Sentence-Transformers"
        "</p>",
        unsafe_allow_html=True,
    )
