"""Generate a grounded answer from retrieved chunks, using Groq API."""
import os
import re
from dotenv import load_dotenv
from groq import Groq

from full_pipeline import FullPipeline
from hybrid import format_citation

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GEN_MODEL = "qwen/qwen3.8-27b"

ANSWER_PROMPT = """You are answering questions using ONLY the numbered source excerpts below.
Rules:
- Every factual claim in your answer MUST be followed by a citation tag like [1] or [2] referring to the excerpt number.
- If the excerpts do NOT contain enough information to answer the question, say exactly:
  "The provided sources do not contain enough information to answer this question."
  and nothing else. Do not guess or use outside knowledge.

Sources:
{sources}

Question: {question}

Answer (with inline [N] citations):"""


def build_sources_block(hits):
    lines = []
    for i, h in enumerate(hits, 1):
        citation = format_citation(h)
        lines.append(f"[{i}] {citation}\n{h['text']}\n")
    return "\n".join(lines)


def generate_answer(question, hits):
    if not hits:
        return "The provided sources do not contain enough information to answer this question.", []

    sources_block = build_sources_block(hits)
    prompt = ANSWER_PROMPT.format(sources=sources_block, question=question)

    response = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    answer = response.choices[0].message.content.strip()
    return answer, hits


def verify_citations(answer, hits):
    cited_nums = set(int(n) for n in re.findall(r"\[(\d+)\]", answer))
    valid_nums = set(range(1, len(hits) + 1))
    invalid = cited_nums - valid_nums
    uncited_sources = valid_nums - cited_nums
    return {
        "cited": sorted(cited_nums),
        "invalid_citations": sorted(invalid),
        "unused_sources": sorted(uncited_sources),
        "has_any_citation": len(cited_nums) > 0,
    }


if __name__ == "__main__":
    pipeline = FullPipeline(rerank_threshold=-3.0)

    test_questions = [
        "What are the beta1 and beta2 values for the Adam optimizer?",
        "What attention mechanism does GPT-4 use?",
    ]

    for question in test_questions:
        print(f"\n{'='*70}\nQuestion: {question}\n{'='*70}")
        hits = pipeline.retrieve(question, candidate_k=15, final_k=3)
        answer, used_hits = generate_answer(question, hits)
        print(f"\nAnswer:\n{answer}\n")

        check = verify_citations(answer, used_hits)
        print(f"Citation check: {check}")
        if used_hits:
            print("\nSource key:")
            for i, h in enumerate(used_hits, 1):
                print(f"  [{i}] {format_citation(h)}")
