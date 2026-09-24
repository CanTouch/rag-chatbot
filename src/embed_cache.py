"""Cache embeddings for repeated queries to avoid redundant embedding calls.
Persisted to disk (JSON) so the cache survives across script runs."""
import hashlib
import json
import pathlib
import time

from chromadb.utils import embedding_functions

CACHE_PATH = pathlib.Path(__file__).parent.parent / "embedding_cache.json"
ef = embedding_functions.DefaultEmbeddingFunction()

def embed_text(text):
    return ef([text])[0]


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
