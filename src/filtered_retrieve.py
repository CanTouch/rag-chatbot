"""Retrieval scoped to a specific document (or set of documents) via metadata filtering."""
from full_pipeline import FullPipeline, RERANK_THRESHOLD
from hybrid import format_citation


class FilteredPipeline(FullPipeline):
    def retrieve_scoped(self, query, doc_ids=None, candidate_k=20, final_k=3):
        """doc_ids: list of doc_id strings to restrict search to. None = search everything."""
        # Pull a larger raw candidate pool from BOTH dense and BM25, then filter by doc_id
        # BEFORE reranking -- filtering after would waste rerank compute on chunks we'll discard anyway.
        dense_ranked_ids = self.hybrid._dense_ranking(query, candidate_k * 3)
        bm25_ranked_ids = self.hybrid._bm25_ranking(query, candidate_k * 3)

        id_to_idx = {cid: i for i, cid in enumerate(self.hybrid.chunk_ids)}

        def passes_filter(chunk_id):
            if doc_ids is None:
                return True
            meta = self.hybrid.chunk_metas[id_to_idx[chunk_id]]
            return meta["doc_id"] in doc_ids

        dense_filtered = [cid for cid in dense_ranked_ids if passes_filter(cid)][:candidate_k]
        bm25_filtered = [cid for cid in bm25_ranked_ids if passes_filter(cid)][:candidate_k]

        rrf_scores = {}
        for rank, cid in enumerate(dense_filtered):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)
        for rank, cid in enumerate(bm25_filtered):
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1.0 / (60 + rank + 1)

        top_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:candidate_k]
        candidates = []
        for cid in top_ids:
            idx = id_to_idx[cid]
            meta = self.hybrid.chunk_metas[idx]
            candidates.append({
                "chunk_id": cid,
                "text": self.hybrid.chunk_texts[idx],
                "doc_id": meta["doc_id"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
            })

        reranked = self.reranker.rerank(query, candidates)
        if not reranked or reranked[0]["rerank_score"] < self.rerank_threshold:
            return []
        return reranked[:final_k]


if __name__ == "__main__":
    pipeline = FilteredPipeline(rerank_threshold=-3.0)

    # Ambiguous query -- "attention" appears meaningfully in ALL THREE papers
    # (Transformer's core mechanism, BERT builds on it, RAG's retriever uses attention internally)
    query = "How is attention used in this model?"

    print(f"Query: {query}\n")
    print("-- Unscoped (searches all 3 papers) --")
    for h in pipeline.retrieve(query, candidate_k=15, final_k=3):
        print(f"  {format_citation(h)}  score={h['rerank_score']:.2f}")

    print("\n-- Scoped to bert only --")
    for h in pipeline.retrieve_scoped(query, doc_ids=["bert"], candidate_k=15, final_k=3):
        print(f"  {format_citation(h)}  score={h['rerank_score']:.2f}")

    print("\n-- Scoped to rag_paper only --")
    for h in pipeline.retrieve_scoped(query, doc_ids=["rag_paper"], candidate_k=15, final_k=3):
        print(f"  {format_citation(h)}  score={h['rerank_score']:.2f}")
