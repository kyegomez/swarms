from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import IO, Dict, List, Optional

from pypdf import PdfReader

from swarms.utils.hash import str_to_hash


@dataclass
class TextArtifact:
    text: str


@dataclass
class PDFLoader:
    """
    A class for loading PDF files and extracting text artifacts.

    Args:
        tokenizer (str): The tokenizer to use for chunking the text.
        max_tokens (int): The maximum number of tokens per chunk.

    Methods:
        load(source, password=None, *args, **kwargs):
            Load a single PDF file and extract text artifacts.

        load_collection(sources, password=None, *args, **kwargs):
            Load a collection of PDF files and extract text artifacts.

    Private Methods:
        _load_pdf(stream, password=None):
            Load a PDF file and extract text artifacts.

    Attributes:
        tokenizer (str): The tokenizer used for chunking the text.
        max_tokens (int): The maximum number of tokens per chunk.
    """

    tokenizer: str
    max_tokens: int

    def __post_init__(self):
        self.chunker = PdfChunker(
            tokenizer=self.tokenizer, max_tokens=self.max_tokens
        )

    def load(
        self,
        source: str | IO | Path,
        password: Optional[str] = None,
        *args,
        **kwargs,
    ) -> List[TextArtifact]:
        return self._load_pdf(source, password)

    def load_collection(
        self,
        sources: List[str | IO | Path],
        password: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Dict[str, List[TextArtifact]]:
        return {
            str_to_hash(str(s)): self._load_pdf(s, password)
            for s in sources
        }

    def _load_pdf(
        self, stream: str | IO | Path, password: Optional[str]
    ) -> List[TextArtifact]:
        reader = PdfReader(stream, strict=True, password=password)
        return [
            TextArtifact(text=p.extract_text()) for p in reader.pages
        ]
