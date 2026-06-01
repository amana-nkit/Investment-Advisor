"""
loader/loaders.py
Secure document ingestion:
  - PDF: MIME validation, size cap, temp-file cleanup, chunking
  - Web: targeted market context from trusted sources

MIME detection uses a pure-Python magic-byte check (no native libmagic required),
so this works on Windows, macOS, and Linux without any system-level install.
"""
from __future__ import annotations

import hashlib
import os
import tempfile
from typing import Optional

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from core.config import get_settings
from models.schemas import InvestmentDocument
from utilities.logger import get_logger

log = get_logger(__name__)
settings = get_settings()

# PDF magic bytes: every valid PDF starts with %PDF-
_PDF_MAGIC = b"%PDF-"

TRUSTED_WEB_SOURCES = [
    "https://zerodha.com/varsity/chapter/introduction-to-personal-finance/",
    "https://zerodha.com/varsity/chapter/mutual-funds-basics/",
]


class PDFValidationError(ValueError):
    pass


def _is_pdf(data: bytes) -> bool:
    """
    Pure-Python PDF header check — no libmagic / system dependency.
    Checks the first 1024 bytes to handle PDFs with a leading BOM or whitespace.
    """
    return _PDF_MAGIC in data[:1024]


def validate_and_read_pdf(uploaded_file) -> bytes:
    """
    Read uploaded file bytes, enforce size cap, verify PDF magic bytes.
    Raises PDFValidationError on any violation.
    Pure-Python — works on Windows, macOS, and Linux without libmagic.
    """
    data = uploaded_file.read()
    max_bytes = settings.max_pdf_mb * 1024 * 1024

    if len(data) > max_bytes:
        raise PDFValidationError(
            f"File size {len(data) / 1024 / 1024:.1f} MB exceeds {settings.max_pdf_mb} MB limit."
        )

    if not _is_pdf(data):
        raise PDFValidationError(
            "File does not appear to be a valid PDF (missing %PDF- header). "
            "Only PDF files are accepted."
        )

    log.info("PDF validated: %.1f KB, sha256=%s", len(data) / 1024, hashlib.sha256(data).hexdigest()[:12])
    return data


def load_and_split_pdf(
    file_data: bytes,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Write PDF bytes to a secure temp file, load with PyPDFLoader,
    split into chunks, validate each chunk, clean up temp file.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_data)
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        raw_docs = loader.load()

        if not raw_docs:
            raise PDFValidationError("PDF appears to be empty or unreadable.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_docs = splitter.split_documents(raw_docs)
        log.info("PDF split into %d chunks (size=%d, overlap=%d)", len(split_docs), chunk_size, chunk_overlap)

        # Validate each chunk with Pydantic
        valid_docs: list[Document] = []
        for doc in split_docs:
            try:
                InvestmentDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "uploaded_pdf"),
                    doc_type="pdf",
                )
                valid_docs.append(doc)
            except Exception as e:
                log.debug("Chunk skipped (validation failed): %s", e)

        log.info("%d / %d chunks passed validation", len(valid_docs), len(split_docs))
        return valid_docs

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
            log.debug("Temp file cleaned up: %s", tmp_path)


def load_web_context() -> list[Document]:
    """
    Fetch market education content from trusted sources.
    Returns empty list on failure — web context is optional enrichment.
    """
    docs: list[Document] = []
    for url in TRUSTED_WEB_SOURCES:
        try:
            loader = WebBaseLoader(url)
            loaded = loader.load()
            for d in loaded:
                d.metadata["source_type"] = "web"
                d.metadata["source_url"] = url
            docs.extend(loaded[:2])  # cap per-source
            log.info("Loaded web context from %s", url)
        except Exception as e:
            log.warning("Web loader failed for %s: %s", url, e)
    return docs