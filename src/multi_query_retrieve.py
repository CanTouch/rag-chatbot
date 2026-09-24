"""Multi-query retrieval: rewrite query into variants, retrieve for each, merge, rerank once against the ORIGINAL query."""
from query_rewrite import rewrite_query
from hybrid import HybridRetriever, format_citation
from rerank import Reranker


class MultiQueryPipeline:
    def __init__(self):
        self.hybrid = HybridRetriever()
        self.reranker = Reranker()

    def retrieve(self, query, candidate_k=15, final_k=3, rerank_threshold=-3.0):
        variants = rewrite_query(query)
        print(f"  Using {len(variants)} query variant(s): {variants}")

        seen_ids = {}
        for variant in variants:
            hits = self.hybrid.retrieve(variant, top_k=candidate_k, candidate_k=candidate_k)
            for h in hits:
                if h["chunk_id"] not in seen_ids:
                    seen_ids[h["chunk_id"]] = h

        candidates = list(seen_ids.values())
        # IMPORTANT: rerank against the ORIGINAL query, not the rewrites --
        # relevance should be judged by what the user actually asked, not a paraphrase
        reranked = self.reranker.rerank(query, candidates)

        if not reranked or reranked[0]["rerank_score"] < rerank_threshold:
            return []
        return reranked[:final_k]


if __name__ == "__main__":
    pipeline = MultiQueryPipeline()
    casual_queries = [
        "how long did it take to train the big model",
        "what words did they use to fill in the blanks in BERT",
    ]
    for q in casual_queries:
        print(f"\nQuery: {q}")
        hits = pipeline.retrieve(q)
        for rank, h in enumerate(hits, 1):
            print(f"  {rank}. {format_citation(h)}  score={h['rerank_score']:.2f}")
            print(f"     {h['text'][:150]}...")
