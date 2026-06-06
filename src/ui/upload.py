"""
Streamlit Upload Documents page.
"""

import os
import tempfile
import streamlit as st
from datetime import datetime

from src.utils.helpers import DOCUMENT_TYPE_OPTIONS, DISEASE_AREA_OPTIONS


def render_upload(pipeline, doc_manager):
    st.title("📄 Upload Documents")
    st.markdown(
        "Upload healthcare documents to add them to the knowledge base. "
        "Supported formats: **PDF**, **DOCX**, **TXT**."
    )

    st.markdown("---")
    st.subheader("Step 1: Select File")
    uploaded_file = st.file_uploader(
        "Choose a document",
        type=["pdf", "docx", "txt"],
        help="Maximum recommended size: 20 MB",
    )

    st.markdown("---")
    st.subheader("Step 2: Document Metadata")
    st.markdown("Providing accurate metadata improves search filtering accuracy.")

    col1, col2 = st.columns(2)
    with col1:
        title = st.text_input(
            "Document Title *",
            placeholder="e.g., Phase III Trial of Drug X in Type 2 Diabetes",
            help="A descriptive title for this document.",
        )
        document_type = st.selectbox("Document Type *", DOCUMENT_TYPE_OPTIONS)
        disease_area = st.selectbox("Disease Area", DISEASE_AREA_OPTIONS)

    with col2:
        drug_name = st.text_input(
            "Drug / Compound Name",
            placeholder="e.g., Metformin, Atorvastatin (or N/A)",
        )
        publication_year = st.text_input(
            "Publication Year",
            placeholder="e.g., 2023",
        )
        sponsor = st.text_input(
            "Sponsor / Author",
            placeholder="e.g., WHO, FDA, Smith et al.",
        )

    st.markdown("---")
    st.subheader("Step 3: Processing Options")
    col3, col4 = st.columns(2)
    with col3:
        chunk_size = st.slider(
            "Chunk Size (characters)", min_value=500, max_value=2000,
            value=1000, step=100,
            help="Larger chunks preserve more context; smaller chunks improve precision.",
        )
    with col4:
        chunk_overlap = st.slider(
            "Chunk Overlap (characters)", min_value=0, max_value=500,
            value=200, step=50,
            help="Overlap prevents important context from being split across chunk boundaries.",
        )

    st.markdown("---")

    if st.button("🚀 Upload & Process Document", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please select a file first.")
            return
        if not title.strip():
            st.error("Please enter a document title.")
            return

        _process_upload(
            uploaded_file=uploaded_file,
            pipeline=pipeline,
            doc_manager=doc_manager,
            metadata={
                "title": title.strip(),
                "document_type": document_type,
                "disease_area": disease_area,
                "drug_name": drug_name.strip() or "N/A",
                "publication_year": publication_year.strip() or "",
                "sponsor": sponsor.strip(),
            },
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )


def _process_upload(uploaded_file, pipeline, doc_manager, metadata, chunk_size, chunk_overlap):
    progress = st.progress(0, text="Starting...")
    status = st.empty()

    try:
        # Save uploaded file to a temp location
        suffix = "." + uploaded_file.name.rsplit(".", 1)[-1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        progress.progress(10, text="File saved. Extracting text...")
        status.info("Extracting and cleaning document text...")

        # Update chunking parameters on the pipeline's chunker
        pipeline.chunker.chunk_size = chunk_size
        pipeline.chunker.chunk_overlap = chunk_overlap

        progress.progress(30, text="Generating embeddings...")
        status.info("Generating embeddings — this may take a moment for large documents...")

        summary = pipeline.ingest_document(file_path=tmp_path, metadata=metadata)

        progress.progress(80, text="Storing in vector database...")
        status.info("Persisting vectors in ChromaDB...")

        doc_manager.register(
            document_id=summary["document_id"],
            metadata=metadata,
            ingest_summary=summary,
        )

        progress.progress(100, text="Done!")
        status.empty()
        os.unlink(tmp_path)

        st.success(f"✅ Document '{metadata['title']}' successfully processed!")
        _show_ingest_summary(summary, metadata)

    except Exception as exc:
        progress.empty()
        status.empty()
        st.error(f"❌ Processing failed: {exc}")
        st.exception(exc)
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _show_ingest_summary(summary: dict, metadata: dict):
    st.markdown("#### Processing Summary")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks Created", summary["total_chunks"])
    col2.metric("File Size", f"{summary['file_size_kb']:.1f} KB")
    col3.metric("Text Reduction", f"{summary['reduction_pct']}%")
    col4.metric("Document ID", summary["document_id"][:8] + "...")

    with st.expander("Full Processing Details"):
        st.json({
            "document_id": summary["document_id"],
            "title": metadata["title"],
            "document_type": metadata["document_type"],
            "disease_area": metadata["disease_area"],
            "drug_name": metadata["drug_name"],
            "total_chunks": summary["total_chunks"],
            "raw_characters": summary["raw_chars"],
            "cleaned_characters": summary["cleaned_chars"],
            "upload_date": summary["upload_date"],
        })
