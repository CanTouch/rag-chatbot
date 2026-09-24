"""Query the index and return results with page citations and confidence gating."""
import chromadb
from index_corpus import embed_text, DB_PATH, COLLECTION_NAME

REJECTION_THRESHOLD = 360.0  # calibrated from eval: correct hits max ~303, adversarial min ~440

class Retriever:
    def __init__(self):
        client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = client.get_collection(COLLECTION_NAME)

    def retrieve(self, query, top_k=3):
        query_embedding = embed_text(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        hits = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            hits.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "doc_id": meta["doc_id"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "distance": results["distances"][0][i],
            })
        return hits

    def format_citation(self, hit):
        pages = f"p.{hit['page_start']}" if hit["page_start"] == hit["page_end"] else f"pp.{hit['page_start']}-{hit['page_end']}"
        return f"[{hit['doc_id']}, {pages}]"

def answer_query(query, top_k=3, threshold=REJECTION_THRESHOLD):
    retriever = Retriever()
    hits = retriever.retrieve(query, top_k)
    print(f"\nQuery: {query}\n")

    if not hits or hits[0]["distance"] > threshold:
        print(f"--- NO CONFIDENT MATCH (top distance={hits[0]['distance']:.1f} > threshold={threshold}) ---")
        print("This appears to be outside the corpus. Refusing to answer rather than guess.\n")
        return []

    for rank, hit in enumerate(hits, 1):
        citation = retriever.format_citation(hit)
        flag = "" if hit["distance"] <= threshold else "  [BELOW CONFIDENCE THRESHOLD]"
        print(f"--- Result {rank} {citation}  (distance={hit['distance']:.4f}){flag} ---")
        print(hit["text"][:300] + ("..." if len(hit["text"]) > 300 else ""))
        print()
    return hits

if __name__ == "__main__":
    test_queries = [
        "How many attention heads does the Transformer use?",
        "What is the capital of France?",
        "What retriever does the RAG model use?",
        "Who won the 2022 FIFA World Cup?",
    ]
    for q in test_queries:
        answer_query(q, top_k=2)
