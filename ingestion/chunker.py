"""text_chunker.py - Text Splitting & Chunking

Divides documents into manageable chunks for embedding and retrieval.

Key responsibilities:
  1. Split long documents into chunks (default: 512 tokens)
  2. Add overlap between chunks (default: 50 tokens) to preserve context
  3. Preserve semantic boundaries (don't split mid-sentence)
  4. Attach metadata to each chunk (source, page number, etc.)
  5. Prepare for embedding

Why chunking matters:
  - LLMs have context windows; documents are larger
  - Overlap ensures context isn't lost at chunk boundaries
  - Proper chunking dramatically improves retrieval quality

This is exactly how enterprise RAG pipelines handle long documents,
ensuring that retrieved context is relevant and contextually complete.
"""
"""
chunker.py

Splits documents into smaller chunks suitable for embedding
and retrieval in a RAG pipeline.
"""

from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


def _canonical_source(source: str) -> str:
    """Normalize source so duplicate uploads like 'file (1).pdf' map to the same name."""
    name = Path(source).name
    stem = Path(name).stem
    # strip Windows-style duplicate suffix " (1)" / " (2)" etc.
    if stem.endswith(")") and " (" in stem:
        base, maybe_num = stem.rsplit(" (", 1)
        if maybe_num[:-1].isdigit():
            stem = base
    return stem


def chunk_document(text, source):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = [
        chunk.strip()
        for chunk in splitter.split_text(text)
        if chunk and chunk.strip()
    ]

    documents = []

    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk,
                metadata={
                    "source": _canonical_source(source)
                }
            )
        )

    return documents
