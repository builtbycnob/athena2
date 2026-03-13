# src/athena/rag/ingestion/swiss.py
"""Swiss legal corpus ingestion from HuggingFace (rcds/swiss_legislation).

Loads the dataset, splits each law into article-level chunks,
embeds with BGE-M3, and upserts into LanceDB.
"""

from __future__ import annotations

import hashlib
import logging
import re
from html.parser import HTMLParser

import numpy as np

logger = logging.getLogger("athena.rag.ingestion.swiss")

# PDF regex: Art./§ preceded by double-space or start-of-string (section headers)
# Excludes cross-references like "gestützt auf Art. 46 des Bundesgesetzes"
_PDF_ARTICLE_RE = re.compile(
    r"(?:^|  )"  # start of string OR double-space (PDF section separator)
    r"((?:Art\.?|§)\s*\d+(?:bis|ter|quater|quinquies|sexies|septies|octies|[a-z])?)"
    r"\s+",  # must be followed by whitespace (title/body)
    re.IGNORECASE,
)


class _ArticleHTMLParser(HTMLParser):
    """Extract articles from structured Swiss legislation HTML.

    The HTML uses <div class="article"> with nested
    <div class="article_number"> and <div class="article_title">.
    """

    def __init__(self):
        super().__init__()
        self.articles: list[dict] = []
        self._in_article = False
        self._in_article_number = False
        self._current_number = ""
        self._current_text_parts: list[str] = []
        self._depth = 0
        self._capture_text = False

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        cls = attrs_dict.get("class", "")
        if tag == "div" and "article" == cls:
            # Flush previous article
            if self._current_number and self._current_text_parts:
                text = " ".join(self._current_text_parts).strip()
                if text:
                    self.articles.append(
                        {"article_number": self._current_number, "text": text}
                    )
            self._in_article = True
            self._in_article_number = False
            self._current_number = ""
            self._current_text_parts = []
            self._capture_text = False
        elif tag == "div" and "article_number" in cls:
            self._in_article_number = True
        elif self._in_article and tag == "div" and cls not in (
            "article_number",
            "article_title",
            "article_symbol",
        ):
            self._capture_text = True

    def handle_endtag(self, tag):
        if tag == "div" and self._in_article_number:
            self._in_article_number = False

    def handle_data(self, data):
        if self._in_article_number:
            # Collect article number text (e.g., "Art. 1", "§ 3")
            stripped = data.strip()
            if stripped and stripped not in ("Art.", "§"):
                self._current_number = f"Art. {stripped}" if stripped[0].isdigit() else stripped
            elif stripped in ("Art.", "§"):
                self._current_number = stripped
        elif self._in_article and not self._in_article_number:
            stripped = data.strip()
            if stripped:
                self._current_text_parts.append(stripped)

    def flush(self):
        if self._current_number and self._current_text_parts:
            text = " ".join(self._current_text_parts).strip()
            if text:
                self.articles.append(
                    {"article_number": self._current_number, "text": text}
                )


def _split_html_into_articles(html: str) -> list[dict]:
    """Split structured HTML into article-level chunks using the DOM."""
    parser = _ArticleHTMLParser()
    parser.feed(html)
    parser.flush()
    return parser.articles


def _split_pdf_into_articles(text: str) -> list[dict]:
    """Split PDF-extracted text into article-level chunks.

    PDF text has articles separated by double-spaces before Art./§ markers.
    Cross-references (Art. N in mid-sentence) are filtered by the double-space anchor.
    """
    matches = list(_PDF_ARTICLE_RE.finditer(text))
    if not matches:
        return []

    # Filter: keep only matches that look like section headers
    # (preceded by sentence-ending punctuation or section title, not mid-sentence)
    articles = []
    for i, match in enumerate(matches):
        art_num = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chunk_text = text[start:end].strip()
        if chunk_text and len(chunk_text) > 10:  # skip tiny fragments
            articles.append({"article_number": art_num, "text": chunk_text})

    return articles


def _split_into_articles(html: str, pdf: str, sr_number: str) -> list[dict]:
    """Split a law into article-level chunks.

    Prefers HTML parsing (structured) over PDF regex (heuristic).
    Falls back to whole-text chunking if no articles found.
    """
    # Try HTML first (structured, reliable)
    if html.strip():
        articles = _split_html_into_articles(html)
        if articles:
            return articles

    # Fall back to PDF regex
    if pdf.strip():
        articles = _split_pdf_into_articles(pdf)
        if articles:
            return articles

    # Last resort: whole text as single chunk
    text = (html or pdf).strip()
    if text:
        return [{"article_number": "full", "text": text}]
    return []


def _make_chunk_id(sr_number: str, article_number: str, language: str) -> str:
    """Create a deterministic chunk ID."""
    key = f"{sr_number}:{article_number}:{language}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word for German/French/Italian."""
    return max(1, int(len(text.split()) * 0.75))


def ingest_swiss_corpus(batch_size: int = 256) -> dict:
    """Ingest the Swiss legislation corpus from HuggingFace.

    Loads rcds/swiss_legislation, splits into articles, embeds, and stores.

    Returns:
        Dict with ingestion statistics.
    """
    from athena.rag.embedder import embed_dense
    from athena.rag.store import NormChunk, upsert_chunks

    try:
        from datasets import load_dataset
    except ImportError:
        return {"status": "error", "message": "datasets package not installed"}

    logger.info("Loading rcds/swiss_legislation from HuggingFace...")
    ds = load_dataset("rcds/swiss_legislation", split="train")

    total_laws = 0
    total_chunks = 0
    html_split = 0
    pdf_split = 0
    full_chunks = 0
    chunk_batch: list[NormChunk] = []
    text_batch: list[str] = []

    for row in ds:
        sr_number = row.get("sr_number", row.get("id", "unknown"))
        language = row.get("language", "de")
        html = row.get("html_content") or ""
        pdf = row.get("pdf_content") or ""
        title = row.get("title", "")
        valid_from = row.get("version_active_since", row.get("entry_into_force", None))

        if not html.strip() and not pdf.strip():
            continue

        articles = _split_into_articles(html, pdf, sr_number)
        if not articles:
            continue

        total_laws += 1
        # Track splitting method
        if len(articles) == 1 and articles[0]["article_number"] == "full":
            full_chunks += 1
        elif html.strip():
            html_split += 1
        else:
            pdf_split += 1

        for art in articles:
            chunk_id = _make_chunk_id(sr_number, art["article_number"], language)
            # For embedding, use plain text (strip HTML tags if needed)
            art_text = art["text"]
            chunk = NormChunk(
                chunk_id=chunk_id,
                jurisdiction="CH",
                sr_number=sr_number,
                article_number=art["article_number"],
                section_breadcrumb=title,
                language=language,
                text=art_text,
                valid_from=str(valid_from) if valid_from else None,
                token_count=_estimate_tokens(art_text),
            )
            chunk_batch.append(chunk)
            text_batch.append(f"{title} {art['article_number']}: {art_text}")

            # Flush batch
            if len(chunk_batch) >= batch_size:
                embeddings = embed_dense(text_batch)
                upsert_chunks(chunk_batch, embeddings, "CH")
                total_chunks += len(chunk_batch)
                logger.info(f"Ingested {total_chunks} chunks from {total_laws} laws...")
                chunk_batch = []
                text_batch = []

    # Final batch
    if chunk_batch:
        embeddings = embed_dense(text_batch)
        upsert_chunks(chunk_batch, embeddings, "CH")
        total_chunks += len(chunk_batch)

    logger.info(
        f"Swiss corpus ingestion complete: {total_laws} laws, {total_chunks} chunks "
        f"(html_split={html_split}, pdf_split={pdf_split}, full={full_chunks})"
    )
    return {
        "status": "ok",
        "laws_processed": total_laws,
        "chunks_created": total_chunks,
        "html_split": html_split,
        "pdf_split": pdf_split,
        "full_chunks": full_chunks,
    }
