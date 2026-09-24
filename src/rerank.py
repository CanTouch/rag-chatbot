"""Two-stage retrieval: fast dense retrieval for candidates, cross-encoder reranking for precision."""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from retrieve import Retriever

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        self.model.eval()

    def rerank(self, query, hits):
        if not hits:
            return hits
        pairs = [(query, h["text"]) for h in hits]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            scores = self.model(**inputs).logits.squeeze(-1).tolist()
        if isinstance(scores, float):
            scores = [scores]
        for h, s in zip(hits, scores):
            h["rerank_score"] = s
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)


def answer_query_reranked(query, retrieve_k=10, final_k=3):
    retriever = Retriever()
    reranker = Reranker()

    candidates = retriever.retrieve(query, top_k=retrieve_k)
    reranked = reranker.rerank(query, candidates)[:final_k]

    print(f"\nQuery: {query}\n")
    for rank, hit in enumerate(reranked, 1):
        citation = retriever.format_citation(hit)
        print(f"--- Result {rank} {citation}  (rerank_score={hit['rerank_score']:.2f}, dense_distance={hit['distance']:.1f}) ---")
        print(hit["text"][:250] + "...")
        print()
    return reranked


if __name__ == "__main__":
    # the exact query that failed before reranking
    answer_query_reranked("What is positional encoding based on in the Transformer model?", retrieve_k=10, final_k=3)
