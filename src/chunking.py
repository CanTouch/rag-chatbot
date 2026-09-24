"""Chunk extracted PDF pages into overlapping, page-aware chunks."""

from dataclasses import dataclass
from extract import extract_pdf_pages


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int


def chunk_pdf(doc_id, pdf_path, chunk_size_tokens=180, overlap_tokens=30):
    pages = extract_pdf_pages(pdf_path)

    words = []
    word_pages = []
    for page_num, page_text in pages:
        for w in page_text.split():
            words.append(w)
            word_pages.append(page_num)

    stride = chunk_size_tokens - overlap_tokens
    chunks = []
    i = 0
    chunk_idx = 0
    while i < len(words):
        window = words[i : i + chunk_size_tokens]
        window_pages = word_pages[i : i + chunk_size_tokens]
        text = " ".join(window)
        chunks.append(Chunk(
            doc_id=doc_id,
            chunk_id=f"{doc_id}::chunk_{chunk_idx}",
            text=text,
            page_start=min(window_pages),
            page_end=max(window_pages),
        ))
        chunk_idx += 1
        if i + chunk_size_tokens >= len(words):
            break
        i += stride

    return chunks


if __name__ == "__main__":
    import pathlib
    CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"
    for pdf_file in sorted(CORPUS_DIR.glob("*.pdf")):
        doc_id = pdf_file.stem
        chunks = chunk_pdf(doc_id, pdf_file)
        print(f"{doc_id}: {len(chunks)} chunks")
        for c in chunks[:3]:
            print(f"  {c.chunk_id}  pages {c.page_start}-{c.page_end}")
