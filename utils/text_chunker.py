"""
text_chunker.py
----------------
Split extracted document text into overlapping chunks suitable for
embedding and retrieval, using LangChain's RecursiveCharacterTextSplitter.
"""

from typing import Dict, List


def _get_splitter(chunk_size: int, chunk_overlap: int):
    """Import RecursiveCharacterTextSplitter, supporting both old and new
    LangChain package layouts."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Split a single block of raw text into overlapping chunks."""
    if not text or not text.strip():
        return []

    splitter = _get_splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


def chunk_documents(
    documents: Dict[str, str], chunk_size: int = 500, chunk_overlap: int = 50
) -> Dict[str, List[str]]:
    """
    Chunk a dict of {filename: full_text} into {filename: [chunk, chunk, ...]}.
    Convenience wrapper for batch processing multiple uploaded documents.
    """
    return {
        filename: chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for filename, text in documents.items()
    }
