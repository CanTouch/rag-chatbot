"""Full pipeline: hybrid retrieval (dense+BM25) -> cross-encoder rerank -> confidence threshold."""
from hybrid import HybridRetriever, format_citation
from rerank import Reranker

RERANK_THRESHOLD = -3.0  # calibrated below, then hardcoded once known


class FullPipeline:
    def __init__(self, rerank_threshold=RERANK_THRESHOLD):
        self.hybrid = HybridRetriever()
        self.reranker = Reranker()
        self.rerank_threshold = rerank_threshold

    def retrieve(self, query, candidate_k=20, final_k=3):
        candidates = self.hybrid.retrieve(query, top_k=candidate_k, candidate_k=candidate_k)
        reranked = self.reranker.rerank(query, candidates)

        if self.rerank_threshold is not None:
            if not reranked or reranked[0]["rerank_score"] < self.rerank_threshold:
                return []  # confident rejection: nothing clears the bar

        return reranked[:final_k]


if __name__ == "__main__":
    pipeline = FullPipeline(rerank_threshold=RERANK_THRESHOLD)
    hits = pipeline.retrieve("What are the beta1 and beta2 values for the Adam optimizer?")
    for rank, hit in enumerate(hits, 1):
        print(f"{rank}. {format_citation(hit)}  rerank_score={hit['rerank_score']:.2f}")
