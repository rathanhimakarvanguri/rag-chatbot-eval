"""Load Markdown documents and split them into section-level chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ragkit.text import split_sentences


@dataclass(frozen=True)
class Chunk:
    id: str  # "<doc-stem>#<section-slug>[-<n>]"
    doc: str
    title: str  # document title
    section: str  # section heading
    text: str

    @property
    def section_id(self) -> str:
        """Section-level id, without any sub-chunk suffix; used to match gold labels."""
        return re.sub(r"--\d+$", "", self.id)

    def render(self) -> str:
        return f"{self.title} > {self.section}\n{self.text}"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _pack(sentences: list[str], max_chars: int, overlap: int) -> list[str]:
    """Greedily pack sentences into windows of at most max_chars, overlapping by N sentences."""
    windows, current = [], []
    for sent in sentences:
        if current and len(" ".join(current + [sent])) > max_chars:
            windows.append(" ".join(current))
            current = current[-overlap:] if overlap else []
        current.append(sent)
    if current:
        windows.append(" ".join(current))
    return windows


def chunk_markdown(doc: str, markdown: str, max_chars: int = 900, overlap: int = 1) -> list[Chunk]:
    """Split on '## ' headings; long sections are packed into overlapping sentence windows.

    Text before the first '## ' heading (title and preamble) is dropped.
    """
    title_match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    title = title_match.group(1).strip() if title_match else doc
    parts = re.split(r"^##\s+(.+)$", markdown, flags=re.MULTILINE)
    chunks: list[Chunk] = []
    # parts = [preamble, heading1, body1, heading2, body2, ...]
    for heading, body in zip(parts[1::2], parts[2::2]):
        body = " ".join(body.split())
        if not body:
            continue
        base = f"{doc}#{slugify(heading)}"
        windows = _pack(split_sentences(body), max_chars, overlap)
        for i, window in enumerate(windows):
            cid = base if len(windows) == 1 else f"{base}--{i}"
            chunks.append(Chunk(cid, doc, title, heading.strip(), window))
    return chunks


def load_corpus(path: str | Path, max_chars: int = 900, overlap: int = 1) -> list[Chunk]:
    path = Path(path)
    files = sorted(path.glob("**/*.md")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"No .md files found under {path}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_markdown(f.stem, f.read_text(encoding="utf-8"), max_chars, overlap))
    return chunks
