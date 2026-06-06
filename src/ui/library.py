"""
Streamlit Document Library page.
"""

import streamlit as st
from src.utils.helpers import truncate_text


def render_library(doc_manager, vector_store):
    st.title("📚 Document Library")
    st.markdown("Browse and manage all documents currently in the knowledge base.")

    documents = doc_manager.list_all()

    if not documents:
        st.info(
            "No documents in the library yet. "
            "Go to **Upload Documents** to add your first healthcare document."
        )
        return

    # Summary metrics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Documents", len(documents))
    col2.metric("Total Chunks", vector_store.count())
    types = {d.get("document_type", "Unknown") for d in documents}
    col3.metric("Document Types", len(types))

    st.markdown("---")

    # Filters
    col_a, col_b, col_c = st.columns(3)
    filter_opts = doc_manager.get_filter_options()

    with col_a:
        type_filter = st.selectbox(
            "Filter by Type",
            ["All"] + filter_opts.get("document_type", []),
        )
    with col_b:
        area_filter = st.selectbox(
            "Filter by Disease Area",
            ["All"] + filter_opts.get("disease_area", []),
        )
    with col_c:
        search_term = st.text_input("Search by title or drug name", "")

    # Apply filters
    filtered = documents
    if type_filter != "All":
        filtered = [d for d in filtered if d.get("document_type") == type_filter]
    if area_filter != "All":
        filtered = [d for d in filtered if d.get("disease_area") == area_filter]
    if search_term:
        term_lower = search_term.lower()
        filtered = [
            d for d in filtered
            if term_lower in d.get("title", "").lower()
            or term_lower in d.get("drug_name", "").lower()
        ]

    st.markdown(f"**Showing {len(filtered)} of {len(documents)} documents**")
    st.markdown("---")

    for doc in filtered:
        _render_document_card(doc, doc_manager, vector_store)


def _render_document_card(doc: dict, doc_manager, vector_store):
    summary = doc.get("ingest_summary", {})
    doc_id = doc.get("document_id", "N/A")
    title = doc.get("title", "Untitled")
    doc_type = doc.get("document_type", "Unknown")
    disease_area = doc.get("disease_area", "N/A")
    drug_name = doc.get("drug_name", "N/A")
    pub_year = doc.get("publication_year", "")
    chunks = summary.get("total_chunks", "?")
    upload_date = doc.get("registered_at", "")[:10]

    type_colors = {
        "Clinical Trial Report": "#1a6eb5",
        "Medical Guideline": "#1a8c50",
        "Drug Information Document": "#b58c1a",
        "Research Paper": "#8c1ab5",
        "FDA Guidance Document": "#b51a1a",
        "Clinical Study Protocol": "#1a9eb5",
    }
    color = type_colors.get(doc_type, "#555")

    with st.expander(f"📄 {title}  |  {doc_type}  |  {disease_area}", expanded=False):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(f"**Document ID:** `{doc_id[:16]}...`")
            st.markdown(
                f"**Type:** <span style='color:{color}; font-weight:600;'>{doc_type}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**Disease Area:** {disease_area}")
            st.markdown(f"**Drug / Compound:** {drug_name}")
            if pub_year:
                st.markdown(f"**Publication Year:** {pub_year}")
            st.markdown(f"**Source File:** {doc.get('source_file', 'N/A')}")
            st.markdown(f"**Uploaded:** {upload_date}")
            st.markdown(f"**Chunks in DB:** {chunks}")

        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("🗑️ Remove", key=f"del_{doc_id}"):
                doc_manager.remove(doc_id)
                vector_store.delete_by_document_id(doc_id)
                st.success(f"Removed '{title}' from the knowledge base.")
                st.rerun()
