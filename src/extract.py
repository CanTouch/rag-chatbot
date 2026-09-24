"""Extract text from PDFs, preserving page numbers."""

import pathlib
from pypdf import PdfReader

CORPUS_DIR = pathlib.Path(__file__).parent.parent / "corpus"


def extract_pdf_pages(pdf_path):
    """Returns list of (page_number, page_text) tuples. Page numbers are 1-indexed."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():
            pages.append((i, text))
    return pages


if __name__ == "__main__":
    for pdf_file in sorted(CORPUS_DIR.glob("*.pdf")):
        pages = extract_pdf_pages(pdf_file)
        print(f"\n{pdf_file.name}: {len(pages)} pages with text")
        # print first 150 chars of page 1 as a sanity check
        if pages:
            page_num, text = pages[0]
            print(f"  Page {page_num} preview: {text[:150]!r}")
