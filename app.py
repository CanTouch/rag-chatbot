"""Streamlit UI for the RAG chatbot."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from full_pipeline import FullPipeline
from generate_answer import generate_answer, verify_citations
from hybrid import format_citation

st.set_page_config(page_title="RAG Chatbot", page_icon="📚")
st.title("📚 RAG Chatbot")
st.caption("Ask questions — answers are grounded in the loaded documents only.")

@st.cache_resource
def load_pipeline():
    return FullPipeline(rerank_threshold=-3.0)

pipeline = load_pipeline()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            hits = pipeline.retrieve(question, candidate_k=15, final_k=3)
            answer, used_hits = generate_answer(question, hits)

        st.markdown(answer)

        if used_hits:
            with st.expander("Sources"):
                for i, h in enumerate(used_hits, 1):
                    st.markdown(f"**[{i}]** `{format_citation(h)}`")
                    st.caption(h["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
