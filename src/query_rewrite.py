"""Rewrite a user's query into 2-3 search-friendly variants using a local LLM."""
import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
REWRITE_MODEL = "llama3.2:3b"

REWRITE_PROMPT = """You are a search query rewriter for a technical paper retrieval system.
Given a user's question, generate 2 alternative phrasings that use more precise,
technical vocabulary likely to appear in academic papers, while preserving the
original meaning exactly. Do not answer the question. Do not add new facts.

Return ONLY a JSON array of 2 strings, nothing else.

User question: {query}

JSON array:"""


def rewrite_query(query):
    prompt = REWRITE_PROMPT.format(query=query)
    resp = requests.post(OLLAMA_URL, json={
        "model": REWRITE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3},
    })
    resp.raise_for_status()
    raw = resp.json()["response"].strip()

    # LLMs sometimes wrap JSON in markdown fences -- strip those defensively
    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [query] + [v for v in variants if isinstance(v, str)]
    except json.JSONDecodeError:
        pass

    # If parsing fails, honestly fall back to just the original query rather than
    # silently guessing at malformed output
    print(f"  [rewrite parse failed, raw output: {raw!r}]")
    return [query]


if __name__ == "__main__":
    test_queries = [
        "how long did it take to train the big model",
        "what words did they use to fill in the blanks in BERT",
    ]
    for q in test_queries:
        print(f"\nOriginal: {q}")
        variants = rewrite_query(q)
        for v in variants:
            print(f"  -> {v}")
