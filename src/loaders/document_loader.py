"""
Document loader for PDF, DOCX, and TXT files.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Loads text content from PDF, DOCX, and TXT files."""

    SUPPORTED_FORMATS = {".pdf", ".docx", ".txt"}

    def load(self, file_path: str) -> dict:
        """
        Load a document and return its text content plus basic metadata.

        Returns a dict with keys: text, source_file, file_type, load_date
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported file format: {suffix}. Supported: {self.SUPPORTED_FORMATS}")

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if suffix == ".pdf":
            text = self._load_pdf(file_path)
        elif suffix == ".docx":
            text = self._load_docx(file_path)
        else:
            text = self._load_txt(file_path)

        return {
            "text": text,
            "source_file": str(path.name),
            "file_type": suffix.lstrip(".").upper(),
            "load_date": datetime.now().isoformat(),
            "file_size_kb": round(path.stat().st_size / 1024, 2),
        }

    def _load_pdf(self, file_path: str) -> str:
        try:
            import pdfplumber
            pages = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        pages.append(page_text)
            return "\n\n".join(pages)
        except ImportError:
            logger.warning("pdfplumber not available, trying PyMuPDF")
            return self._load_pdf_pymupdf(file_path)

    def _load_pdf_pymupdf(self, file_path: str) -> str:
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            pages = [page.get_text() for page in doc]
            doc.close()
            return "\n\n".join(pages)
        except ImportError:
            raise ImportError("Install pdfplumber or PyMuPDF: pip install pdfplumber")

    def _load_docx(self, file_path: str) -> str:
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

    def _load_txt(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def get_supported_formats(self) -> list:
        return list(self.SUPPORTED_FORMATS)
