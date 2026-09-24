"""Parent document retrieval: match on small chunks, but expand to surrounding
context (the 'parent' window) before returning results to the user/LLM.
Deduplicates overlapping text between adjacent chunks when merging."""
from filtered_retrieve import FilteredPipeline
from hybrid import format_citation


def merge_overlapping_texts(texts):
    """Join a sequence of chunk texts that may share an overlapping tail/head
    (since chunks were built with token overlap), without duplicating the
    overlapping words. Uses a simple longest-common-overlap check word by word."""
    if not texts:
        return ""
    merged_words = texts[0].split()
    for next_text in texts[1:]:
        next_words = next_text.split()
        # find the longest suffix of merged_words that matches a prefix of next_words
        max_check = min(len(merged_words), len(next_words), 60)  # our overlap is ~30 tokens, cap the search
        overlap_len = 0
        for k in range(max_check, 0, -1):
            if merged_words[-k:] == next_words[:k]:
                overlap_len = k
                break
        merged_words.extend(next_words[overlap_len:])
    return " ".join(merged_words)


class ParentRetrievalPipeline(FilteredPipeline):
    def __init__(self, *args, parent_window=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.parent_window = parent_window

        self.doc_chunk_order = {}
        for i, cid in enumerate(self.hybrid.chunk_ids):
            doc_id, chunk_part = cid.split("::chunk_")
            chunk_num = int(chunk_part)
            self.doc_chunk_order.setdefault(doc_id, {})[chunk_num] = i

    def expand_to_parent(self, hit):
        doc_id = hit["doc_id"]
        chunk_num = int(hit["chunk_id"].split("::chunk_")[1])
        order = self.doc_chunk_order[doc_id]

        neighbor_nums = sorted(
            n for n in range(chunk_num - self.parent_window, chunk_num + self.parent_window + 1)
            if n in order
        )

        texts, page_starts, page_ends = [], [], []
        for n in neighbor_nums:
            idx = order[n]
            texts.append(self.hybrid.chunk_texts[idx])
            meta = self.hybrid.chunk_metas[idx]
            page_starts.append(meta["page_start"])
            page_ends.append(meta["page_end"])

        return {
            **hit,
            "expanded_text": merge_overlapping_texts(texts),
            "expanded_page_start": min(page_starts),
            "expanded_page_end": max(page_ends),
            "n_chunks_merged": len(neighbor_nums),
        }

    def retrieve_with_context(self, query, doc_ids=None, candidate_k=15, final_k=3):
        if doc_ids:
            hits = self.retrieve_scoped(query, doc_ids=doc_ids, candidate_k=candidate_k, final_k=final_k)
        else:
            hits = self.retrieve(query, candidate_k=candidate_k, final_k=final_k)
        return [self.expand_to_parent(h) for h in hits]


if __name__ == "__main__":
    pipeline = ParentRetrievalPipeline(rerank_threshold=-3.0, parent_window=1)

    query = "What are the beta1 and beta2 values for the Adam optimizer?"
    print(f"Query: {query}\n")
    results = pipeline.retrieve_with_context(query, final_k=1)
    for r in results:
        pages = f"pp.{r['expanded_page_start']}-{r['expanded_page_end']}"
        print(f"[{r['doc_id']}, {pages}]  merged {r['n_chunks_merged']} chunks")
        print(f"\nMATCHED CHUNK ONLY ({len(r['text'])} chars):")
        print(r["text"])
        print(f"\nEXPANDED, DEDUPLICATED ({len(r['expanded_text'])} chars):")
        print(r["expanded_text"])
        # sanity check: verify no duplicated sentence
        dup_check = "machine with 8 NVIDIA P100 GPUs"
        count = r["expanded_text"].count(dup_check)
        print(f"\n[sanity check] '{dup_check}' appears {count} time(s) in expanded text (should be 1)")
