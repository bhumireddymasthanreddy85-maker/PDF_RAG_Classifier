"""
pdf_parser.py
-------------
Extract text from PDF files using multiple free, local backends with
automatic fallback, in order of typical extraction quality:

    pdfplumber -> PyMuPDF (fitz) -> PyPDF2

Accepts a file path (str), raw bytes, or a file-like object (e.g. a
Streamlit ``UploadedFile``), so it works the same way whether you're
reading data/sample.pdf from disk or a file the user just uploaded.
"""

import io
from typing import Union


def _read_bytes(file_input: Union[str, bytes, "io.BufferedReader"]) -> bytes:
    """Normalize any supported input type into raw PDF bytes."""
    if isinstance(file_input, bytes):
        return file_input
    if isinstance(file_input, str):
        with open(file_input, "rb") as f:
            return f.read()
    # Assume a file-like object (e.g. Streamlit's UploadedFile)
    file_input.seek(0)
    data = file_input.read()
    file_input.seek(0)
    return data


def extract_text_pdfplumber(file_input) -> str:
    """Extract text using pdfplumber (best for text-based, well-formed PDFs)."""
    import pdfplumber

    data = _read_bytes(file_input)
    text_parts = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts).strip()


def extract_text_pymupdf(file_input) -> str:
    """Extract text using PyMuPDF / fitz (fast, handles many edge cases)."""
    import fitz  # PyMuPDF

    data = _read_bytes(file_input)
    text_parts = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts).strip()


def extract_text_pypdf2(file_input) -> str:
    """Extract text using PyPDF2 (fallback backend)."""
    from PyPDF2 import PdfReader

    data = _read_bytes(file_input)
    reader = PdfReader(io.BytesIO(data))
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts).strip()


def extract_text(file_input, min_chars: int = 20) -> dict:
    """
    Try each backend in order and return the first successful result.

    Returns
    -------
    dict with keys:
        text          - extracted text (empty string if all backends failed)
        backend_used  - name of the backend that succeeded, or None
        error         - last error message, if extraction failed entirely
    """
    backends = [
        ("pdfplumber", extract_text_pdfplumber),
        ("pymupdf", extract_text_pymupdf),
        ("PyPDF2", extract_text_pypdf2),
    ]

    last_error = None
    for name, fn in backends:
        try:
            text = fn(file_input)
            if text and len(text.strip()) >= min_chars:
                return {"text": text, "backend_used": name, "error": None}
        except Exception as e:  # noqa: BLE001 - we want to try the next backend
            last_error = f"{name}: {e}"
            continue

    return {
        "text": "",
        "backend_used": None,
        "error": last_error or "No extractable text found in this PDF.",
    }
