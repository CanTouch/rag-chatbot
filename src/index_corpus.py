"""Embed all chunks via Ollama and store them in ChromaDB."""
import pathlib
import requests
import chromadb
from chunking import chunk_pdf

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"
DB_PATH = str(pathlib.Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "papers"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL_NAME = "nomic-embed-text"

def embed_text(text):
    resp = requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "prompt": text})
    resp.raise_for_status()
    return resp.json()["embedding"]

def build_index():
    all_chunks = []
    for pdf_file in sorted(CORPUS_DIR.glob("*.pdf")):
        doc_id = pdf_file.stem
        chunks = chunk_pdf(doc_id, pdf_file)
        all_chunks.extend(chunks)
        print(f"{doc_id}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks via Ollama (this will take a bit)...")
    embeddings = []
    for i, c in enumerate(all_chunks):
        embeddings.append(embed_text(c.text))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(all_chunks)} embedded")

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        embeddings=embeddings,
        documents=[c.text for c in all_chunks],
        metadatas=[{"doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end} for c in all_chunks],
    )
    print(f"\nIndexed {collection.count()} chunks into '{COLLECTION_NAME}' at {DB_PATH}")

if __name__ == "__main__":
    build_index()
