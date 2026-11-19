"""
In-memory document registry.

For this academic prototype a database is not needed: processed documents
(their pages, chunks and TF-IDF index) are kept in a Python dictionary keyed by
a UUID. This means:
  * questions can be asked repeatedly without re-uploading the PDF;
  * all documents are forgotten when the server restarts.

A lock protects the dictionary because FastAPI may handle requests in
several threads at the same time.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from app.exceptions import DocumentNotFoundError
from app.services.retrieval_service import TfidfRetriever
from app.utils.text_chunking import PageText, TextChunk

logger = logging.getLogger(__name__)


@dataclass
class StoredDocument:
    document_id: str
    filename: str
    file_path: Optional[Path]
    file_size_bytes: int
    page_count: int
    pages: List[PageText]
    empty_pages: List[int]
    chunks: List[TextChunk]
    retriever: TfidfRetriever
    status: str = "ready"
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def pages_with_text(self) -> int:
        return self.page_count - len(self.empty_pages)

    @property
    def total_characters(self) -> int:
        return sum(len(p.text) for p in self.pages)


class DocumentStore:
    def __init__(self) -> None:
        self._documents: Dict[str, StoredDocument] = {}
        self._lock = threading.Lock()

    def add(self, document: StoredDocument) -> None:
        with self._lock:
            self._documents[document.document_id] = document

    def get(self, document_id: str) -> StoredDocument:
        with self._lock:
            document = self._documents.get(document_id)
        if document is None:
            raise DocumentNotFoundError(f"No document was found with ID '{document_id}'.")
        return document

    def delete(self, document_id: str) -> StoredDocument:
        """Remove a document from the registry and delete its uploaded file."""
        with self._lock:
            document = self._documents.pop(document_id, None)
        if document is None:
            raise DocumentNotFoundError(f"No document was found with ID '{document_id}'.")
        self._remove_file(document)
        return document

    def clear(self, remove_files: bool = True) -> None:
        with self._lock:
            documents = list(self._documents.values())
            self._documents.clear()
        if remove_files:
            for document in documents:
                self._remove_file(document)

    def __len__(self) -> int:
        with self._lock:
            return len(self._documents)

    @staticmethod
    def _remove_file(document: StoredDocument) -> None:
        if document.file_path is None:
            return
        try:
            document.file_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete file for document %s: %s", document.document_id, exc)

