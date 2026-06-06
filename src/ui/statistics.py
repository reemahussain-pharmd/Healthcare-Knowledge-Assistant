"""
Streamlit System Statistics page.
"""

import streamlit as st
from src.utils.helpers import truncate_text


def render_statistics(pipeline, doc_manager):
    st.title("📊 System Statistics")
    st.markdown("Overview of the knowledge base, retrieval performance, and question history.")

    st.markdown("---")
    st.subheader("Knowledge Base Overview")

    stats = pipeline.get_system_stats()
    documents = doc_manager.list_all()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", stats.get("total_documents", 0))
    col2.metric("Total Chunks", stats.get("total_chunks", 0))
    col3.metric("Questions Asked", stats.get("questions_asked", 0))
    col4.metric("Embedding Dim", stats.get("embedding_dimension", "N/A"))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Embedding Model", stats.get("embedding_model", "N/A"))
    col6.metric("LLM Provider", stats.get("llm_provider", "N/A"))
    col7.metric("LLM Model", stats.get("llm_model", "N/A"))
    col8.metric("Chunk Size", stats.get("chunk_size", "N/A"))

    st.markdown("---")
    st.subheader("Document Type Distribution")

    doc_types = stats.get("document_types", {})
    if doc_types:
        st.bar_chart(doc_types)
    else:
        st.info("No documents indexed yet.")

    disease_areas = stats.get("disease_areas", {})
    if disease_areas:
        st.markdown("#### Disease Area Distribution")
        st.bar_chart(disease_areas)

    # Ingestion stats per document
    if documents:
        st.markdown("---")
        st.subheader("Document Details")
        rows = []
        for doc in documents:
            summary = doc.get("ingest_summary", {})
            rows.append({
                "Title": truncate_text(doc.get("title", "N/A"), 40),
                "Type": doc.get("document_type", "N/A"),
                "Disease Area": doc.get("disease_area", "N/A"),
                "Drug": doc.get("drug_name", "N/A"),
                "Year": doc.get("publication_year", ""),
                "Chunks": summary.get("total_chunks", "?"),
                "Size (KB)": summary.get("file_size_kb", "?"),
                "Uploaded": doc.get("registered_at", "")[:10],
            })
        st.dataframe(rows, use_container_width=True)

    # Question history
    if pipeline.question_history:
        st.markdown("---")
        st.subheader("Question History")

        # Average confidence
        confs = [q.get("confidence_score", 0) for q in pipeline.question_history]
        avg_conf = sum(confs) / len(confs) if confs else 0

        col_a, col_b = st.columns(2)
        col_a.metric("Total Questions", len(pipeline.question_history))
        col_b.metric("Avg Confidence", f"{avg_conf:.1%}")

        for i, qh in enumerate(reversed(pipeline.question_history), 1):
            with st.expander(
                f"Q{len(pipeline.question_history) - i + 1}: {truncate_text(qh['question'], 80)}",
                expanded=(i == 1),
            ):
                st.markdown(f"**Question:** {qh['question']}")
                st.markdown(f"**Answer (preview):** {qh['answer']}")
                col_i, col_j, col_k = st.columns(3)
                col_i.metric("Confidence", f"{qh.get('confidence_score', 0):.1%}")
                col_j.metric("Chunks Used", qh.get("num_chunks", 0))
                col_k.metric("Timestamp", qh.get("timestamp", "")[:16])
    else:
        st.info("No questions asked yet. Go to **Ask Questions** to get started.")

    # Evaluation metrics section
    st.markdown("---")
    st.subheader("Evaluation Metrics")
    st.markdown(
        """
        | Metric | Description | How to Interpret |
        |--------|-------------|------------------|
        | **Confidence Score** | Avg cosine similarity of retrieved chunks | >0.7 = High, 0.4–0.7 = Medium, <0.4 = Low |
        | **Chunks Retrieved** | Number of passages sent to LLM | Higher = more context (may add noise) |
        | **Response Time** | End-to-end latency | Includes embedding + retrieval + LLM call |
        | **Tokens Used** | LLM tokens consumed | Relevant for cost estimation |
        | **Chunk Size** | Characters per chunk | Larger = more context; smaller = more precise |
        | **Chunk Overlap** | Overlap between consecutive chunks | Reduces boundary-splitting artifacts |
        """
    )

    # Reset knowledge base
    st.markdown("---")
    st.subheader("Danger Zone")
    with st.expander("Reset Knowledge Base"):
        st.warning(
            "⚠️ This will permanently delete ALL documents and chunks from the vector database. "
            "This action cannot be undone."
        )
        confirm = st.text_input("Type RESET to confirm:", key="reset_confirm")
        if st.button("🗑️ Reset Knowledge Base", type="primary"):
            if confirm == "RESET":
                pipeline.vector_store.reset()
                # Clear registry
                for doc in doc_manager.list_all():
                    doc_manager.remove(doc["document_id"])
                pipeline.question_history.clear()
                st.success("Knowledge base reset. All documents and chunks deleted.")
                st.rerun()
            else:
                st.error("Please type RESET exactly to confirm.")
