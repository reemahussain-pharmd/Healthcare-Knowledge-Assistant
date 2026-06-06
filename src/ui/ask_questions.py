"""
Streamlit Ask Questions page.
"""

import streamlit as st
from src.utils.helpers import EXAMPLE_QUESTIONS, highlight_medical_terms, DOCUMENT_TYPE_OPTIONS, DISEASE_AREA_OPTIONS


def render_ask_questions(pipeline, doc_manager):
    st.title("❓ Ask Questions")
    st.markdown(
        "Ask questions about your uploaded healthcare documents. "
        "The system retrieves relevant passages and generates evidence-based answers."
    )

    if pipeline.vector_store.count() == 0:
        st.warning(
            "⚠️ No documents in the knowledge base yet. "
            "Please upload documents first via **Upload Documents**."
        )
        return

    if pipeline.llm_client is None:
        st.error(
            "❌ No LLM configured. Please set **OPENAI_API_KEY** or **GEMINI_API_KEY** in your `.env` file "
            "and restart the application."
        )
        return

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Search Filters")
        st.caption("Optionally narrow retrieval to specific document subsets.")
        filter_opts = doc_manager.get_filter_options()

        doc_type_filter = st.selectbox(
            "Document Type",
            ["All"] + filter_opts.get("document_type", []),
            key="q_doc_type",
        )
        disease_area_filter = st.selectbox(
            "Disease Area",
            ["All"] + filter_opts.get("disease_area", []),
            key="q_disease",
        )
        drug_filter = st.selectbox(
            "Drug Name",
            ["All"] + filter_opts.get("drug_name", []),
            key="q_drug",
        )
        year_filter = st.selectbox(
            "Publication Year",
            ["All"] + filter_opts.get("publication_year", []),
            key="q_year",
        )
        top_k = st.slider("Top K Chunks", min_value=1, max_value=15, value=5)
        min_score = st.slider("Min Relevance Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

    # Build filter dict
    filters = {}
    if doc_type_filter != "All":
        filters["document_type"] = doc_type_filter
    if disease_area_filter != "All":
        filters["disease_area"] = disease_area_filter
    if drug_filter != "All":
        filters["drug_name"] = drug_filter
    if year_filter != "All":
        filters["publication_year"] = year_filter

    # Question input
    st.markdown("---")
    st.markdown("### Your Question")

    # Example questions selector
    with st.expander("📝 Example Questions (click to load)"):
        for example in EXAMPLE_QUESTIONS:
            if st.button(example, key=f"ex_{example[:30]}"):
                st.session_state["current_question"] = example

    question = st.text_area(
        "Enter your question:",
        value=st.session_state.get("current_question", ""),
        height=100,
        placeholder="e.g., What are the inclusion criteria for this clinical trial?",
        key="question_input",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        ask_clicked = st.button("🔍 Ask", type="primary", use_container_width=True)
    with col2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.pop("current_question", None)
            st.session_state.pop("last_answer", None)
            st.rerun()

    if ask_clicked and question.strip():
        st.session_state["current_question"] = question
        with st.spinner("Retrieving relevant passages and generating answer..."):
            try:
                result = pipeline.ask(
                    question=question,
                    top_k=top_k,
                    filters=filters if filters else None,
                    min_score=min_score,
                )
                st.session_state["last_answer"] = result
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
                return

    # Display answer
    if "last_answer" in st.session_state:
        _render_answer(st.session_state["last_answer"])


def _render_answer(result: dict):
    st.markdown("---")
    st.markdown("### Answer")

    # Confidence badge
    conf = result.get("confidence_score", 0)
    if conf >= 0.7:
        conf_color, conf_label = "#1a8c50", "High"
    elif conf >= 0.4:
        conf_color, conf_label = "#b58c1a", "Medium"
    else:
        conf_color, conf_label = "#b51a1a", "Low"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Confidence", f"{conf:.1%}")
    col2.metric("Chunks Retrieved", result.get("num_chunks_retrieved", 0))
    col3.metric("Response Time", f"{result.get('response_time_s', 0):.2f}s")
    col4.metric("LLM", result.get("model", "N/A"))

    st.markdown(
        f"""
        <div style="background:#f8f9fa; border-left:4px solid {conf_color};
                    padding:1.5rem; border-radius:0 8px 8px 0; margin:1rem 0;">
            <p style="margin:0; font-size:1rem; line-height:1.7; color:#222;">
                {result['answer'].replace(chr(10), '<br>')}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Citations
    if result.get("citations"):
        st.markdown("#### Source Citations")
        for i, cit in enumerate(result["citations"], 1):
            st.markdown(
                f"""
                <div style="background:#eaf4fb; border-radius:8px; padding:0.8rem; margin:0.4rem 0;">
                    <strong>[{i}] {cit['title']}</strong><br>
                    <span style="color:#555; font-size:0.85rem;">
                        Type: {cit.get('document_type','N/A')} &nbsp;|&nbsp;
                        Disease: {cit.get('disease_area','N/A')} &nbsp;|&nbsp;
                        Drug: {cit.get('drug_name','N/A')} &nbsp;|&nbsp;
                        Year: {cit.get('publication_year','N/A')} &nbsp;|&nbsp;
                        File: {cit.get('source_file','N/A')} &nbsp;|&nbsp;
                        Relevance: {cit.get('score',0):.1%}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Retrieved chunks detail
    with st.expander("🔍 View Retrieved Chunks", expanded=False):
        for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
            meta = chunk.get("metadata", {})
            st.markdown(
                f"**Chunk {i}** | Score: `{chunk.get('score',0):.4f}` | "
                f"Source: {meta.get('title','N/A')} | "
                f"Chunk Index: {meta.get('chunk_index','N/A')}/{meta.get('total_chunks','N/A')}"
            )
            highlighted = highlight_medical_terms(chunk["text"])
            st.markdown(
                f"""
                <div style="background:#f0f0f0; padding:0.8rem; border-radius:6px;
                            font-size:0.85rem; line-height:1.5; margin-bottom:0.8rem;">
                    {highlighted}
                </div>
                """,
                unsafe_allow_html=True,
            )
