"""Streamlit UI for the RAG chatbot with PDF upload."""
import sys
import os
import tempfile
import pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from chunking import chunk_pdf
from full_pipeline import FullPipeline
from generate_answer import generate_answer
from hybrid import format_citation

st.set_page_config(page_title="KuppeLabs RAG Chatbot", page_icon="📚")
st.title("📚 KuppeLabs RAG Chatbot")
st.caption("Upload your documents and ask questions — answers are grounded in your documents only.")

DB_PATH = str(pathlib.Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "papers"
ef = embedding_functions.DefaultEmbeddingFunction()

def index_uploaded_pdf(pdf_bytes, filename):
    doc_id = pathlib.Path(filename).stem
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = pathlib.Path(tmp.name)

    chunks = chunk_pdf(doc_id, tmp_path)
    tmp_path.unlink()

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = client.get_collection(COLLECTION_NAME, embedding_function=ef)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    existing_ids = set(collection.get()["ids"])
    new_chunks = [c for c in chunks if c.chunk_id not in existing_ids]

    if new_chunks:
        collection.add(
            ids=[c.chunk_id for c in new_chunks],
            documents=[c.text for c in new_chunks],
            metadatas=[{"doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end} for c in new_chunks],
        )
    return len(new_chunks), doc_id

def get_pipeline():
    return FullPipeline(rerank_threshold=-3.0)

# Sidebar for PDF upload
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        any_new = False
        for f in uploaded_files:
            with st.spinner(f"Indexing {f.name}..."):
                n, doc_id = index_uploaded_pdf(f.read(), f.name)
            if n > 0:
                st.success(f"✅ {f.name}: {n} chunks indexed")
                any_new = True
            else:
                st.info(f"ℹ️ {f.name}: already indexed")

    st.divider()
    st.caption("Default corpus: Attention, BERT, RAG papers")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            pipeline = get_pipeline()
            hits = pipeline.retrieve(question, candidate_k=15, final_k=3)
            answer, used_hits = generate_answer(question, hits)

        st.markdown(answer)

        if used_hits:
            with st.expander("Sources"):
                for i, h in enumerate(used_hits, 1):
                    st.markdown(f"**[{i}]** `{format_citation(h)}`")
                    st.caption(h["text"][:300] + "...")

    st.session_state.messages.append({"role": "assistant", "content": answer})
