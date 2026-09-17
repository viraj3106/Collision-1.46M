import os
import re
from typing import List, Optional
from collision.rag.schemas import DocumentChunk

class DocumentChunker:
    """
    Deterministic document ingestion and chunker for text (.txt) and markdown (.md) documents.
    Preserves document source, chunk IDs, and token metadata without discarding content.
    """

    def __init__(self, chunk_size_words: int = 50, chunk_overlap_words: int = 10):
        self.chunk_size_words = max(10, chunk_size_words)
        self.chunk_overlap_words = max(0, min(chunk_overlap_words, self.chunk_size_words - 1))

    def clean_text(self, text: str) -> str:
        """Cleans and standardizes raw text."""
        if not text:
            return ""
        # Normalize newlines and whitespace
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def chunk_text(
        self,
        text: str,
        document_id: str,
        source: str
    ) -> List[DocumentChunk]:
        """
        Deterministically chunks text into overlapping word segments.
        """
        cleaned = self.clean_text(text)
        if not cleaned:
            return []

        words = cleaned.split()
        if not words:
            return []

        chunks: List[DocumentChunk] = []
        step = self.chunk_size_words - self.chunk_overlap_words
        chunk_idx = 0

        for start_idx in range(0, len(words), step):
            end_idx = min(start_idx + self.chunk_size_words, len(words))
            chunk_words = words[start_idx:end_idx]
            chunk_str = " ".join(chunk_words)

            if chunk_str.strip():
                chunk = DocumentChunk(
                    document_id=document_id,
                    source=source,
                    chunk_id=chunk_idx,
                    text=chunk_str,
                    token_count=len(chunk_words),
                    metadata={"word_start": start_idx, "word_end": end_idx}
                )
                chunks.append(chunk)
                chunk_idx += 1

            if end_idx >= len(words):
                break

        return chunks

    def ingest_file(
        self,
        filepath: str,
        document_id: Optional[str] = None
    ) -> List[DocumentChunk]:
        """Ingests and chunks a single .txt or .md file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Document file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in (".txt", ".md", ".markdown"):
            raise ValueError(f"Unsupported file format '{ext}'. Only .txt and .md supported.")

        filename = os.path.basename(filepath)
        doc_id = document_id if document_id is not None else filename

        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.chunk_text(content, document_id=doc_id, source=filename)

    def ingest_directory(
        self,
        dirpath: str,
        recursive: bool = True
    ) -> List[DocumentChunk]:
        """Ingests and chunks all .txt and .md files in a directory."""
        if not os.path.exists(dirpath):
            raise FileNotFoundError(f"Directory not found: {dirpath}")

        all_chunks: List[DocumentChunk] = []
        valid_extensions = {".txt", ".md", ".markdown"}

        for root, _, files in os.walk(dirpath):
            # Sort files deterministically
            for f in sorted(files):
                ext = os.path.splitext(f)[1].lower()
                if ext in valid_extensions:
                    full_path = os.path.join(root, f)
                    chunks = self.ingest_file(full_path)
                    all_chunks.extend(chunks)
            if not recursive:
                break

        return all_chunks
