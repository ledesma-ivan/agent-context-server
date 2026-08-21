"""BM25 full-text search over the vault, backed by SQLite FTS5.

Same incremental-by-content-hash pattern as rag.py's Chroma index: only
re-indexes pages whose content changed, and drops pages that no longer
exist. Kept in a module of its own (no chromadb/torch/FlagEmbedding
imports) so importing it never touches the native-DLL load-order
constraint documented at the top of rag.py.
"""

import hashlib
import sqlite3
from pathlib import Path

from vault_mcp.vault import VAULT_ROOT

FTS_DB_PATH = Path(__file__).parent / "fts_index.db"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(FTS_DB_PATH)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS pages USING fts5("
        "page_id UNINDEXED, content, content_hash UNINDEXED)"
    )
    return conn


def _safe_fts_query(query: str) -> str:
    """Quote every token as its own FTS5 phrase and AND them together
    (FTS5's default between bare tokens). Avoids syntax errors from
    hyphens (NOT-operator prefix), colons (column-filter syntax), and
    other FTS5 query-syntax characters showing up in ordinary text."""
    tokens = [t for t in query.split() if t]
    if not tokens:
        return '""'
    return " ".join('"' + t.replace('"', '""') + '"' for t in tokens)


def index_fts(dry_run: bool = True) -> str:
    """Walk every wiki/**/*.md page and (re)index it into the FTS5 table.

    Mirrors rag.index_vault's incremental strategy: compares each page's
    content hash against what's stored, only re-indexes new/changed
    pages, deletes pages that no longer exist in the vault.
    """
    pages = sorted((VAULT_ROOT / "wiki").rglob("*.md"))

    current: dict[str, str] = {}
    for page in pages:
        rel_path = page.relative_to(VAULT_ROOT / "wiki").as_posix()
        current[rel_path] = page.read_text(encoding="utf-8")

    if dry_run and not FTS_DB_PATH.exists():
        return (
            f"[dry_run FTS5] {len(current)} páginas nuevas/modificadas para indexar, "
            f"0 páginas obsoletas para borrar, 0 sin cambios (se saltean)."
        )

    conn = _get_connection()
    try:
        existing_hashes = dict(
            conn.execute("SELECT page_id, content_hash FROM pages").fetchall()
        )

        to_index = [
            rel_path
            for rel_path, text in current.items()
            if existing_hashes.get(rel_path) != _content_hash(text)
        ]
        stale = [rel_path for rel_path in existing_hashes if rel_path not in current]

        if dry_run:
            return (
                f"[dry_run FTS5] {len(to_index)} páginas nuevas/modificadas para indexar, "
                f"{len(stale)} páginas obsoletas para borrar, "
                f"{len(current) - len(to_index)} sin cambios (se saltean)."
            )

        for rel_path in stale:
            conn.execute("DELETE FROM pages WHERE page_id = ?", (rel_path,))

        if not to_index:
            conn.commit()
            return (
                f"Nada para indexar (FTS5) — las {len(current)} páginas ya estaban al día. "
                f"{len(stale)} obsoleta(s) borrada(s)."
            )

        for rel_path in to_index:
            conn.execute("DELETE FROM pages WHERE page_id = ?", (rel_path,))
            text = current[rel_path]
            conn.execute(
                "INSERT INTO pages (page_id, content, content_hash) VALUES (?, ?, ?)",
                (rel_path, text, _content_hash(text)),
            )
        conn.commit()

        return (
            f"Indexadas {len(to_index)} páginas nuevas/modificadas (FTS5), "
            f"{len(stale)} obsoleta(s) borrada(s), "
            f"{len(current) - len(to_index)} sin cambios (salteadas)."
        )
    finally:
        conn.close()


def search_wiki_ranked(query: str, max_results: int = 20) -> list[tuple[str, float]]:
    """Ranked BM25 search. Returns (page_id, score) sorted best-first,
    score higher = better (SQLite's raw bm25() is a cost, lower-is-better,
    so it's negated here for callers that expect higher-is-better, e.g.
    _rrf_fuse). Returns [] if the index is empty or unreachable."""
    if not FTS_DB_PATH.exists():
        return []
    conn = _get_connection()
    try:
        if conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0:
            return []
        fts_query = _safe_fts_query(query)
        rows = conn.execute(
            "SELECT page_id, bm25(pages) AS rank FROM pages "
            "WHERE pages MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, max_results),
        ).fetchall()
        return [(page_id, -rank) for page_id, rank in rows]
    finally:
        conn.close()


def search_wiki(query: str, max_results: int = 20) -> str:
    """Case-insensitive full-text search across every wiki/**/*.md file,
    ranked by BM25 (via SQLite FTS5) instead of file order.

    Same public signature and output shape as the substring-match version
    it replaces: a '# Resultados de búsqueda' header, one '## {page_id}'
    block per matching page, with a one-line snippet.
    """
    if not FTS_DB_PATH.exists():
        return "El índice FTS5 está vacío — correr index_vault(dry_run=False) primero."

    conn = _get_connection()
    try:
        if conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] == 0:
            return "El índice FTS5 está vacío — correr index_vault(dry_run=False) primero."

        fts_query = _safe_fts_query(query)
        rows = conn.execute(
            "SELECT page_id, snippet(pages, 1, '**', '**', '...', 12) AS snip, "
            "bm25(pages) AS rank FROM pages WHERE pages MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, max_results),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return f"Sin resultados para '{query}'"

    lines = [f"# Resultados de búsqueda: '{query}'", ""]
    for page_id, snip, _rank in rows:
        lines.append(f"## {page_id}")
        lines.append(" ".join(snip.split()))
        lines.append("")

    return "\n".join(lines)
