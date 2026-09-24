"""Embed all chunks using ChromaDB's default embedding function and store in ChromaDB."""
import pathlib
import chromadb
from chromadb.utils import embedding_functions
from chunking import chunk_pdf

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"
DB_PATH = str(pathlib.Path(__file__).parent.parent / "chroma_db")
COLLECTION_NAME = "papers"

def build_index():
    all_chunks = []
    for pdf_file in sorted(CORPUS_DIR.glob("*.pdf")):
        doc_id = pdf_file.stem
        chunks = chunk_pdf(doc_id, pdf_file)
        all_chunks.extend(chunks)
        print(f"{doc_id}: {len(chunks)} chunks")

    print(f"\nEmbedding {len(all_chunks)} chunks (this will take a bit)...")

    ef = embedding_functions.DefaultEmbeddingFunction()

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME, embedding_function=ef)

    collection.add(
        ids=[c.chunk_id for c in all_chunks],
        documents=[c.text for c in all_chunks],
        metadatas=[{"doc_id": c.doc_id, "page_start": c.page_start, "page_end": c.page_end} for c in all_chunks],
    )
    print(f"\nIndexed {collection.count()} chunks into '{COLLECTION_NAME}' at {DB_PATH}")

if __name__ == "__main__":
    build_index()
