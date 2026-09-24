"""Rewrite a user's query into 2-3 search-friendly variants using Groq."""
import json
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
REWRITE_MODEL = "qwen/qwen3.8-27b"

REWRITE_PROMPT = """You are a search query rewriter for a technical paper retrieval system.
Given a user's question, generate 2 alternative phrasings that use more precise,
technical vocabulary likely to appear in academic papers, while preserving the
original meaning exactly. Do not answer the question. Do not add new facts.

Return ONLY a JSON array of 2 strings, nothing else.

User question: {query}

JSON array:"""


def rewrite_query(query):
    prompt = REWRITE_PROMPT.format(query=query)
    response = client.chat.completions.create(
        model=REWRITE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.strip("`").replace("json", "", 1).strip()

    try:
        variants = json.loads(raw)
        if isinstance(variants, list):
            return [query] + [v for v in variants if isinstance(v, str)]
    except json.JSONDecodeError:
        pass

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
