"""Hybrid retrieval: dense (semantic) + BM25 (keyword) combined via reciprocal rank fusion."""
from rank_bm25 import BM25Okapi
import chromadb

from retrieve import Retriever
from index_corpus import DB_PATH, COLLECTION_NAME

RRF_K = 60  # standard RRF damping constant


class HybridRetriever:
    def __init__(self):
        self.dense = Retriever()
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        all_data = collection.get()

        self.chunk_ids = all_data["ids"]
        self.chunk_texts = all_data["documents"]
        self.chunk_metas = all_data["metadatas"]

        tokenized = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(tokenized)

    def _dense_ranking(self, query, k):
        hits = self.dense.retrieve(query, top_k=k)
        return [h["chunk_id"] for h in hits]

    def _bm25_ranking(self, query, k):
        scores = self.bm25.get_scores(query.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunk_ids[i] for i in ranked_idx]

    def retrieve(self, query, top_k=5, candidate_k=20):
        dense_ranked = self._dense_ranking(query, candidate_k)
        bm25_ranked = self._bm25_ranking(query, candidate_k)

        rrf_scores = {}
        for rank, chunk_id in enumerate(dense_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, chunk_id in enumerate(bm25_ranked):
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

        top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        results = []
        for cid in top_ids:
            idx = id_to_idx[cid]
            meta = self.chunk_metas[idx]
            results.append({
                "chunk_id": cid,
                "text": self.chunk_texts[idx],
                "doc_id": meta["doc_id"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "rrf_score": rrf_scores[cid],
                "in_dense": cid in dense_ranked,
                "in_bm25": cid in bm25_ranked,
            })
        return results


def format_citation(hit):
    pages = f"p.{hit['page_start']}" if hit["page_start"] == hit["page_end"] else f"pp.{hit['page_start']}-{hit['page_end']}"
    return f"[{hit['doc_id']}, {pages}]"


if __name__ == "__main__":
    retriever = HybridRetriever()
    # a query with a specific number -- exactly where keyword search should help
    test_queries = [
        "What is the dropout rate used in the Transformer model?",
        "What value of d_model does the Transformer use?",
    ]
    for query in test_queries:
        print(f"\nQuery: {query}\n")
        hits = retriever.retrieve(query, top_k=3, candidate_k=20)
        for rank, hit in enumerate(hits, 1):
            citation = format_citation(hit)
            sources = []
            if hit["in_dense"]:
                sources.append("dense")
            if hit["in_bm25"]:
                sources.append("bm25")
            print(f"--- Result {rank} {citation}  (rrf={hit['rrf_score']:.4f}, found_by={'+'.join(sources)}) ---")
            print(hit["text"][:250] + "...")
            print()
