"""Cache embeddings for repeated queries to avoid redundant Ollama calls.
Persisted to disk (JSON) so the cache survives across script runs."""
import hashlib
import json
import pathlib
import time

from index_corpus import embed_text

CACHE_PATH = pathlib.Path(__file__).parent.parent / "embedding_cache.json"


class CachedEmbedder:
    def __init__(self):
        self.cache = {}
        if CACHE_PATH.exists():
            self.cache = json.loads(CACHE_PATH.read_text())
        self.hits = 0
        self.misses = 0

    def _key(self, text):
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, text):
        key = self._key(text)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        embedding = embed_text(text)
        self.cache[key] = embedding
        return embedding

    def save(self):
        CACHE_PATH.write_text(json.dumps(self.cache))

    def stats(self):
        total = self.hits + self.misses
        rate = self.hits / total if total else 0
        return f"{self.hits} hits, {self.misses} misses ({rate:.1%} hit rate)"


if __name__ == "__main__":
    embedder = CachedEmbedder()

    queries = [
        "What are the beta1 and beta2 values for the Adam optimizer?",
        "How many attention heads does the Transformer use?",
        "What are the beta1 and beta2 values for the Adam optimizer?",  # repeat -- should hit cache
        "How many attention heads does the Transformer use?",  # repeat -- should hit cache
        "What is the capital of France?",  # new
    ]

    for q in queries:
        start = time.perf_counter()
        embedder.embed(q)
        elapsed = time.perf_counter() - start
        print(f"{elapsed*1000:7.1f}ms  {q[:50]}")

    embedder.save()
    print(f"\nCache stats: {embedder.stats()}")
    print(f"Cache saved to {CACHE_PATH}")
