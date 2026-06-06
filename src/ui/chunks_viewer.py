"""
Streamlit Retrieved Chunks Viewer page.
"""

import streamlit as st
from src.utils.helpers import highlight_medical_terms, DOCUMENT_TYPE_OPTIONS, DISEASE_AREA_OPTIONS


def render_chunks_viewer(pipeline, doc_manager):
    st.title("🔬 Retrieved Chunks Viewer")
    st.markdown(
        "Explore what the vector store actually retrieves for any query. "
        "Useful for debugging retrieval quality and tuning parameters."
    )

    if pipeline.vector_store.count() == 0:
        st.warning("No documents indexed yet. Upload documents first.")
        return

    st.markdown("---")

    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Enter a search query:",
            placeholder="e.g., adverse events Metformin",
        )
    with col2:
        top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    # Filters
    with st.expander("Metadata Filters (optional)"):
        filter_opts = doc_manager.get_filter_options()
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            f_type = st.selectbox("Doc Type", ["All"] + filter_opts.get("document_type", []), key="cv_type")
        with col_b:
            f_area = st.selectbox("Disease Area", ["All"] + filter_opts.get("disease_area", []), key="cv_area")
        with col_c:
            f_drug = st.selectbox("Drug Name", ["All"] + filter_opts.get("drug_name", []), key="cv_drug")
        with col_d:
            f_year = st.selectbox("Year", ["All"] + filter_opts.get("publication_year", []), key="cv_year")

    if st.button("🔍 Retrieve Chunks", type="primary"):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        filters = {}
        if f_type != "All":
            filters["document_type"] = f_type
        if f_area != "All":
            filters["disease_area"] = f_area
        if f_drug != "All":
            filters["drug_name"] = f_drug
        if f_year != "All":
            filters["publication_year"] = f_year

        with st.spinner("Retrieving..."):
            chunks = pipeline.retriever.retrieve(
                query=query,
                top_k=top_k,
                filters=filters if filters else None,
            )

        if not chunks:
            st.info("No chunks retrieved for this query. Try different filters or a broader query.")
            return

        st.success(f"Retrieved {len(chunks)} chunks.")
        st.markdown("---")

        # Score distribution chart
        scores = [c.get("score", 0) for c in chunks]
        st.markdown("#### Relevance Score Distribution")
        score_data = {f"Chunk {i+1}": s for i, s in enumerate(scores)}
        st.bar_chart(score_data)

        st.markdown("---")
        st.markdown("#### Retrieved Chunks")

        for i, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            score = chunk.get("score", 0)
            color = "#1a8c50" if score >= 0.7 else ("#b58c1a" if score >= 0.4 else "#b51a1a")

            with st.expander(
                f"Chunk {i} | Score: {score:.4f} | {meta.get('title', 'N/A')} "
                f"[{meta.get('chunk_index', '?')}/{meta.get('total_chunks', '?')}]",
                expanded=(i == 1),
            ):
                col_meta, col_text = st.columns([1, 2])

                with col_meta:
                    st.markdown(
                        f"""
                        <div style="background:#f8f9fa; padding:0.8rem; border-radius:8px; font-size:0.85rem;">
                            <b>Score:</b> <span style="color:{color};">{score:.4f}</span><br>
                            <b>Distance:</b> {chunk.get('distance', 0):.4f}<br>
                            <b>Document:</b> {meta.get('title', 'N/A')}<br>
                            <b>Type:</b> {meta.get('document_type', 'N/A')}<br>
                            <b>Disease:</b> {meta.get('disease_area', 'N/A')}<br>
                            <b>Drug:</b> {meta.get('drug_name', 'N/A')}<br>
                            <b>Year:</b> {meta.get('publication_year', 'N/A')}<br>
                            <b>File:</b> {meta.get('source_file', 'N/A')}<br>
                            <b>Chunk #:</b> {meta.get('chunk_index', 'N/A')}<br>
                            <b>Chunk Size:</b> {meta.get('chunk_size', len(chunk['text']))} chars
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with col_text:
                    highlighted = highlight_medical_terms(chunk["text"])
                    st.markdown(
                        f"""
                        <div style="background:#fff; border:1px solid #ddd; padding:1rem;
                                    border-radius:8px; font-size:0.88rem; line-height:1.6;">
                            {highlighted}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
