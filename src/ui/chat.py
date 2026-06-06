"""
Streamlit Chat Interface — conversational RAG bot with message history.
"""

import streamlit as st
from datetime import datetime
from src.utils.helpers import EXAMPLE_QUESTIONS, DOCUMENT_TYPE_OPTIONS, DISEASE_AREA_OPTIONS


def render_chat(pipeline, doc_manager):
    st.title("💬 Chat with Your Documents")
    st.markdown(
        "Ask questions in a natural conversation. "
        "The assistant remembers the flow of your session and always cites sources."
    )

    # ── Init session state ────────────────────────────────────────────────────
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "👋 Hello! I'm your **Healthcare Knowledge Assistant**.\n\n"
                    "Upload your clinical documents and ask me anything — I'll answer "
                    "strictly from the evidence in your documents and always tell you "
                    "which source I used.\n\n"
                    "**Try asking:**\n"
                    "- *What are the inclusion criteria in the clinical trial?*\n"
                    "- *What adverse effects are reported for Metformin?*\n"
                    "- *What is the recommended dosage of Atorvastatin?*"
                ),
                "timestamp": datetime.now().strftime("%H:%M"),
                "citations": [],
                "confidence": None,
            }
        ]

    # ── Sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Chat Settings")

        top_k = st.slider("Sources per answer (Top K)", 1, 10, 5, key="chat_top_k")
        show_sources = st.toggle("Show source citations", value=True, key="chat_show_sources")
        show_chunks = st.toggle("Show retrieved chunks", value=False, key="chat_show_chunks")

        st.markdown("---")
        st.markdown("### Filters")
        filter_opts = doc_manager.get_filter_options()
        doc_type_f = st.selectbox("Document Type", ["All"] + filter_opts.get("document_type", []), key="cf_type")
        disease_f  = st.selectbox("Disease Area",   ["All"] + filter_opts.get("disease_area", []),  key="cf_disease")
        drug_f     = st.selectbox("Drug Name",      ["All"] + filter_opts.get("drug_name", []),      key="cf_drug")

        st.markdown("---")
        st.markdown("### Quick Questions")
        for q in EXAMPLE_QUESTIONS[:5]:
            if st.button(q[:50] + ("…" if len(q) > 50 else ""), key=f"cq_{q[:20]}", use_container_width=True):
                st.session_state["chat_prefill"] = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state["chat_messages"] = [st.session_state["chat_messages"][0]]
            st.session_state.pop("chat_prefill", None)
            st.rerun()

        total = len([m for m in st.session_state["chat_messages"] if m["role"] == "user"])
        st.caption(f"Messages this session: {total}")

    # ── Warnings ──────────────────────────────────────────────────────────────
    no_docs = pipeline.vector_store.count() == 0
    no_llm  = pipeline.llm_client is None

    if no_docs:
        st.warning("⚠️ No documents uploaded yet. Go to **Upload Documents** first.")
    if no_llm:
        st.error("❌ No LLM API key set. Add **OPENAI_API_KEY** or **GEMINI_API_KEY** to your `.env` file.")

    # ── Render message history ────────────────────────────────────────────────
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["chat_messages"]:
            _render_message(msg, show_sources, show_chunks)

    # ── Input bar ─────────────────────────────────────────────────────────────
    st.markdown("---")
    prefill = st.session_state.pop("chat_prefill", "")

    with st.form(key="chat_form", clear_on_submit=True):
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            user_input = st.text_input(
                "Your message",
                value=prefill,
                placeholder="Ask a question about your healthcare documents…",
                label_visibility="collapsed",
            )
        with col_btn:
            submitted = st.form_submit_button("Send ➤", use_container_width=True, type="primary")

    if submitted and user_input.strip():
        _handle_message(
            question=user_input.strip(),
            pipeline=pipeline,
            top_k=top_k,
            filters={k: v for k, v in {
                "document_type": doc_type_f if doc_type_f != "All" else None,
                "disease_area":  disease_f  if disease_f  != "All" else None,
                "drug_name":     drug_f     if drug_f     != "All" else None,
            }.items() if v},
            no_docs=no_docs,
            no_llm=no_llm,
        )
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _handle_message(question, pipeline, top_k, filters, no_docs, no_llm):
    # Append user message
    st.session_state["chat_messages"].append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%H:%M"),
        "citations": [],
        "confidence": None,
    })

    if no_docs or no_llm:
        reason = (
            "Please upload healthcare documents before asking questions."
            if no_docs else
            "No LLM API key configured. Please add your key to the `.env` file and restart."
        )
        st.session_state["chat_messages"].append({
            "role": "assistant",
            "content": reason,
            "timestamp": datetime.now().strftime("%H:%M"),
            "citations": [],
            "confidence": None,
        })
        return

    with st.spinner("Searching documents and generating answer…"):
        try:
            result = pipeline.ask(
                question=question,
                top_k=top_k,
                filters=filters if filters else None,
            )
            answer    = result["answer"]
            citations = result.get("citations", [])
            confidence = result.get("confidence_score", 0)
            chunks    = result.get("retrieved_chunks", [])
            time_s    = result.get("response_time_s", 0)
            model     = result.get("model", "")

        except Exception as e:
            answer    = f"❌ An error occurred: {e}"
            citations = []
            confidence = 0
            chunks    = []
            time_s    = 0
            model     = ""

    st.session_state["chat_messages"].append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().strftime("%H:%M"),
        "citations": citations,
        "confidence": confidence,
        "chunks": chunks,
        "response_time_s": time_s,
        "model": model,
    })


def _render_message(msg: dict, show_sources: bool, show_chunks: bool):
    role      = msg["role"]
    content   = msg["content"]
    ts        = msg.get("timestamp", "")
    citations = msg.get("citations", [])
    confidence = msg.get("confidence")
    chunks    = msg.get("chunks", [])
    time_s    = msg.get("response_time_s", 0)
    model     = msg.get("model", "")

    if role == "user":
        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:0.6rem 0;">
                <div style="
                    background: linear-gradient(135deg,#1a6eb5,#1a8c50);
                    color:white; padding:0.8rem 1.1rem; border-radius:18px 18px 4px 18px;
                    max-width:75%; font-size:0.95rem; line-height:1.5;
                    box-shadow:0 2px 6px rgba(0,0,0,0.15);">
                    {content}
                    <div style="font-size:0.7rem; opacity:0.7; text-align:right; margin-top:4px;">{ts}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Confidence colour
        if confidence is None:
            badge = ""
        elif confidence >= 0.7:
            badge = f"<span style='background:#1a8c50;color:white;border-radius:8px;padding:1px 8px;font-size:0.7rem;'>● High confidence {confidence:.0%}</span>"
        elif confidence >= 0.4:
            badge = f"<span style='background:#b58c1a;color:white;border-radius:8px;padding:1px 8px;font-size:0.7rem;'>● Medium confidence {confidence:.0%}</span>"
        else:
            badge = f"<span style='background:#b51a1a;color:white;border-radius:8px;padding:1px 8px;font-size:0.7rem;'>● Low confidence {confidence:.0%}</span>"

        meta_line = ""
        if model:
            meta_line = f"<div style='font-size:0.7rem;color:#999;margin-top:4px;'>{ts} · {model} · {time_s:.1f}s {badge}</div>"
        else:
            meta_line = f"<div style='font-size:0.7rem;color:#999;margin-top:4px;'>{ts}</div>"

        formatted_content = content.replace("\n", "<br>")

        st.markdown(
            f"""
            <div style="display:flex; justify-content:flex-start; margin:0.6rem 0;">
                <div style="margin-right:0.5rem; font-size:1.5rem; align-self:flex-end;">🏥</div>
                <div style="
                    background:white; color:#222; padding:0.8rem 1.1rem;
                    border-radius:18px 18px 18px 4px; max-width:80%;
                    font-size:0.95rem; line-height:1.6;
                    box-shadow:0 2px 6px rgba(0,0,0,0.08);
                    border:1px solid #e8edf5;">
                    {formatted_content}
                    {meta_line}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Citations
        if show_sources and citations:
            with st.expander(f"📎 {len(citations)} source(s) cited", expanded=False):
                for i, cit in enumerate(citations, 1):
                    score = cit.get("score", 0)
                    bar_width = int(score * 100)
                    st.markdown(
                        f"""
                        <div style="background:#f0f6ff;border-radius:8px;padding:0.6rem 0.8rem;margin:0.3rem 0;">
                            <strong>[{i}] {cit.get('title','N/A')}</strong><br>
                            <span style="font-size:0.8rem;color:#555;">
                                📂 {cit.get('document_type','N/A')} &nbsp;|&nbsp;
                                🩺 {cit.get('disease_area','N/A')} &nbsp;|&nbsp;
                                💊 {cit.get('drug_name','N/A')} &nbsp;|&nbsp;
                                📅 {cit.get('publication_year','N/A')}
                            </span><br>
                            <span style="font-size:0.75rem;color:#777;">📄 {cit.get('source_file','N/A')}</span><br>
                            <div style="margin-top:4px;">
                                <span style="font-size:0.75rem;color:#555;">Relevance: </span>
                                <div style="background:#e0e0e0;border-radius:4px;height:6px;width:100%;margin-top:2px;">
                                    <div style="background:#1a6eb5;height:6px;border-radius:4px;width:{bar_width}%;"></div>
                                </div>
                                <span style="font-size:0.72rem;color:#888;">{score:.1%}</span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Retrieved chunks
        if show_chunks and chunks:
            with st.expander(f"🔍 {len(chunks)} retrieved passage(s)", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    meta = chunk.get("metadata", {})
                    st.markdown(
                        f"**Passage {i}** · Score `{chunk.get('score',0):.3f}` · "
                        f"{meta.get('title','?')} · chunk {meta.get('chunk_index','?')}"
                    )
                    st.markdown(
                        f"<div style='background:#f5f5f5;padding:0.6rem;border-radius:6px;"
                        f"font-size:0.82rem;line-height:1.5;margin-bottom:0.5rem;'>"
                        f"{chunk['text'][:400]}{'…' if len(chunk['text'])>400 else ''}</div>",
                        unsafe_allow_html=True,
                    )
