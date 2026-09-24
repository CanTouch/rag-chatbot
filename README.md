# KuppeLabs RAG Chatbot

A production-grade RAG chatbot that answers questions strictly from uploaded documents — no hallucinations.

**Live demo:** https://ragchatbot.kuppelabs.com

## Features
- Upload any PDF and ask questions about it
- Hybrid search: dense semantic + BM25 keyword search via RRF
- Cross-encoder reranking
- Citation-grounded answers — every claim references a source
- Refuses to answer out-of-corpus questions
- Groq API for fast inference

## Stack
- Embeddings: ChromaDB default (all-MiniLM-L6-v2)
- Vector store: ChromaDB
- BM25: rank-bm25
- Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
- LLM: Groq API (Qwen 27B)
- UI: Streamlit
- Deployment: Hetzner VPS + nginx + SSL

## Setup
- Clone the repo
- pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
- Add GROQ_API_KEY to .env
- Run: cd src && python3 index_corpus.py
- Run: streamlit run app.py

## Evaluation
- 17/17 on golden question set
- Adversarial stress tested
- Correctly refuses false-premise queries
