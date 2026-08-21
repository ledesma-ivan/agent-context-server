# search_hybrid Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `search_hybrid(query, n_results)` MCP tool that fuses BM25 (new, via SQLite FTS5), vector search (existing `search_semantic`), and graph relatedness (existing `get_related_pages`) with reciprocal rank fusion (RRF), and upgrade `search_wiki` from literal substring matching to real BM25 ranking.

**Architecture:** New module `src/vault_mcp/fts.py` owns a SQLite FTS5 index (same incremental-by-content-hash pattern as `rag.py`'s Chroma index) and replaces `search_wiki`. `rag.py` gets a new `search_semantic_ranked` helper and calls `fts.index_fts` from inside `index_vault` so both indexes stay in sync with one maintenance call. `vault.py` gets a pure `_rrf_fuse` function and the new `search_hybrid` tool, which lazily imports `rag`/`fts` (matching the existing lazy-import pattern at `vault.py:1292` that avoids the native-DLL load-order crash documented in `rag.py`).

**Tech Stack:** Python 3, stdlib `sqlite3` (FTS5 virtual tables), existing `chromadb`+`FlagEmbedding` (untouched), `unittest` (matches `tests/test_hot_serve_budget.py`).

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-21-search-hybrid-rrf-design.md` — follow it exactly; this plan implements it task-by-task.
- No new third-party dependencies (`sqlite3`/FTS5 is stdlib on the CPython builds this project already uses — verified in Task 1, Step 0).
- `search_wiki`'s public signature (`query: str, max_results: int = 20 -> str`) and output shape (`# Resultados de búsqueda: '{query}'` header, `## {page_id}` blocks) do not change — only the ranking/snippet quality improves.
- `index_vault`'s public signature (`dry_run: bool = True -> str`) does not change — it just also maintains the FTS5 index now.
- No changes to `get_related_pages` or its scoring weights (`RELATED_WEIGHT_*` in `vault.py`).
- Never import `rag` or `fts` at module top-level inside `vault.py` — use the lazy in-function import pattern already established at `vault.py:1292`, to avoid the native DLL load-order conflict documented at the top of `rag.py`.
- Windows dev machine, no GPU/model calls in tests — mock `_embed` wherever a test would otherwise trigger BGE-M3 loading.

---

### Task 1: `fts.py` — index + ranked search (internal)

**Files:**
- Create: `src/vault_mcp/fts.py`
- Test: `tests/test_fts.py`

**Interfaces:**
- Consumes: `vault_mcp.vault.VAULT_ROOT` (existing `Path`).
- Produces:
  - `index_fts(dry_run: bool = True) -> str`
  - `search_wiki_ranked(query: str, max_results: int = 20) -> list[tuple[str, float]]` (page_id, score — higher is better)
  - `search_wiki(query: str, max_results: int = 20) -> str`
  - `FTS_DB_PATH: Path` module constant

- [ ] **Step 0: Verify FTS5 is available in this Python's sqlite3 build**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -c "import sqlite3; c = sqlite3.connect(':memory:'); c.execute('CREATE VIRTUAL TABLE t USING fts5(x)'); print('FTS5 OK')"`
Expected: `FTS5 OK`. If this fails, stop and report back — the plan assumes stdlib FTS5 support, confirmed here before writing any code that depends on it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_fts.py`:

```python
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "fts.py"
spec = importlib.util.spec_from_file_location("vault_mcp_fts", MODULE_PATH)
fts = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_fts"] = fts
spec.loader.exec_module(fts)


class TestFtsIndexing(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.vault_root = Path(self.tmpdir.name)
        (self.vault_root / "wiki").mkdir()
        self.db_path = self.vault_root / "fts_index.db"

        # Point the module at our temp vault/db instead of the real ones.
        fts.VAULT_ROOT = self.vault_root
        fts.FTS_DB_PATH = self.db_path

        (self.vault_root / "wiki" / "gatos.md").write_text(
            "Los gatos duermen dieciseis horas por dia.", encoding="utf-8"
        )
        (self.vault_root / "wiki" / "perros.md").write_text(
            "Los perros necesitan paseos diarios y mucho ejercicio.", encoding="utf-8"
        )
        (self.vault_root / "wiki" / "ambos.md").write_text(
            "Gatos y perros pueden convivir si se los presenta con cuidado.", encoding="utf-8"
        )

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_index_fts_dry_run_reports_without_writing(self):
        report = fts.index_fts(dry_run=True)
        self.assertIn("3 páginas nuevas/modificadas", report)
        self.assertFalse(self.db_path.exists())

    def test_index_fts_writes_and_is_idempotent(self):
        first = fts.index_fts(dry_run=False)
        self.assertIn("3 páginas nuevas/modificadas", first)
        self.assertTrue(self.db_path.exists())

        second = fts.index_fts(dry_run=False)
        self.assertIn("Nada para indexar" if "Nada para indexar" in second else "0 páginas nuevas/modificadas", second)

    def test_index_fts_incremental_reindex(self):
        fts.index_fts(dry_run=False)
        (self.vault_root / "wiki" / "gatos.md").write_text(
            "Los gatos duermen dieciseis horas por dia y cazan de noche.", encoding="utf-8"
        )
        report = fts.index_fts(dry_run=False)
        self.assertIn("1 páginas nuevas/modificadas", report)

    def test_index_fts_deletes_stale(self):
        fts.index_fts(dry_run=False)
        (self.vault_root / "wiki" / "perros.md").unlink()
        report = fts.index_fts(dry_run=False)
        self.assertIn("1 obsoleta", report)

    def test_search_wiki_ranked_orders_by_relevance(self):
        fts.index_fts(dry_run=False)
        results = fts.search_wiki_ranked("gatos", max_results=10)
        page_ids = [r[0] for r in results]
        self.assertIn("gatos.md", page_ids)
        self.assertIn("ambos.md", page_ids)
        self.assertNotIn("perros.md", page_ids)
        # gatos.md mentions "gatos" more centrally/frequently than ambos.md
        self.assertEqual(page_ids[0], "gatos.md")

    def test_search_wiki_ranked_empty_index_returns_empty_list(self):
        results = fts.search_wiki_ranked("gatos", max_results=10)
        self.assertEqual(results, [])

    def test_search_wiki_public_format_and_empty_index_message(self):
        empty_msg = fts.search_wiki("gatos")
        self.assertIn("índice FTS5 está vacío", empty_msg)

        fts.index_fts(dry_run=False)
        out = fts.search_wiki("gatos")
        self.assertIn("# Resultados de búsqueda: 'gatos'", out)
        self.assertIn("## gatos.md", out)

    def test_search_wiki_no_results(self):
        fts.index_fts(dry_run=False)
        out = fts.search_wiki("elefantes")
        self.assertEqual(out, "Sin resultados para 'elefantes'")

    def test_search_wiki_ranked_handles_hyphenated_query_without_crash(self):
        fts.index_fts(dry_run=False)
        # Hyphens are FTS5 NOT-operator syntax if unescaped — must not raise.
        results = fts.search_wiki_ranked("llm-wiki gatos", max_results=10)
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_fts.py -v`
Expected: FAIL — `fts.py` does not exist yet (`ModuleNotFoundError` / `FileNotFoundError` from `spec.loader.exec_module`).

- [ ] **Step 3: Write `src/vault_mcp/fts.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_fts.py -v`
Expected: all PASS. If `test_index_fts_writes_and_is_idempotent` fails on the "Nada para indexar" branch text, check the exact string returned by Step 3's `index_fts` matches what the test asserts (both branches are written above — keep them in sync).

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Ivan/Proyectos/vault-mcp-server
git add src/vault_mcp/fts.py tests/test_fts.py
git commit -m "feat: add FTS5-backed BM25 search (fts.py)"
```

---

### Task 2: Remove old `search_wiki` from `vault.py`, wire `server.py` to `fts.search_wiki`

**Files:**
- Modify: `src/vault_mcp/vault.py:545-584` (delete the old `search_wiki` function)
- Modify: `src/vault_mcp/server.py:1-10, 96-98`

**Interfaces:**
- Consumes: `fts.search_wiki` (Task 1).
- Produces: no new interface — `search_wiki` MCP tool now backed by FTS5, same signature.

- [ ] **Step 1: Delete the old `search_wiki` from `vault.py`**

In `src/vault_mcp/vault.py`, delete lines 545-584 (the entire `def search_wiki(query: str, max_results: int = 20) -> str:` function, from its docstring through its final `return "\n".join(lines)`). Leave `get_index()` (the next function) untouched.

- [ ] **Step 2: Update `server.py`'s import and tool wiring**

In `src/vault_mcp/server.py`, change the top import from:

```python
from vault_mcp import rag, vault
```

to:

```python
from vault_mcp import fts, rag, vault
```

Then find the existing `search_wiki` tool (around line 96):

```python
@mcp.tool()
def search_wiki(query: str, max_results: int = 20) -> str:
    ...
    return vault.search_wiki(query, max_results=max_results)
```

Change the body's `return` line to call `fts` instead of `vault`:

```python
@mcp.tool()
def search_wiki(query: str, max_results: int = 20) -> str:
    """Case-insensitive full-text search across every wiki/**/*.md file,
    ranked by BM25 relevance."""
    return fts.search_wiki(query, max_results=max_results)
```

(Keep whatever exact docstring `server.py` already had, just update it to mention BM25/ranking if it currently says something about plain substring matching — check the current text before editing.)

- [ ] **Step 3: Run the full existing test suite to confirm nothing broke**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS (existing `test_hot_serve_budget.py` plus Task 1's `test_fts.py`; `vault.py` no longer defines `search_wiki`, and nothing else in the codebase references `vault.search_wiki` — confirm with the grep below before committing).

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && grep -rn "vault.search_wiki" src/ tests/`
Expected: no output (no remaining references to the deleted function).

- [ ] **Step 4: Commit**

```bash
cd C:/Users/Ivan/Proyectos/vault-mcp-server
git add src/vault_mcp/vault.py src/vault_mcp/server.py
git commit -m "refactor: search_wiki now backed by FTS5 (fts.py), remove substring version"
```

---

### Task 3: `rag.py` — `search_semantic_ranked` + wire FTS5 into `index_vault`

**Files:**
- Modify: `src/vault_mcp/rag.py`
- Test: `tests/test_rag_ranked.py`

**Interfaces:**
- Consumes: `fts.index_fts(dry_run: bool) -> str` (Task 1).
- Produces: `search_semantic_ranked(query: str, n_results: int = 5) -> list[tuple[str, float]]` (page_id, score — higher is better; cosine distance is inverted to similarity via `1 - distance`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag_ranked.py`:

```python
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "rag.py"
spec = importlib.util.spec_from_file_location("vault_mcp_rag", MODULE_PATH)
rag = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_rag"] = rag
spec.loader.exec_module(rag)


class TestSearchSemanticRanked(unittest.TestCase):
    def test_returns_similarity_scores_higher_is_better(self):
        fake_collection = MagicMock()
        fake_collection.count.return_value = 2
        fake_collection.query.return_value = {
            "ids": [["a.md", "b.md"]],
            "distances": [[0.1, 0.4]],
        }
        with patch.object(rag, "get_collection", return_value=fake_collection), \
             patch.object(rag, "_embed", return_value=[[0.0]]):
            result = rag.search_semantic_ranked("query", n_results=2)

        self.assertEqual(result, [("a.md", 0.9), ("b.md", 0.6)])

    def test_empty_collection_returns_empty_list(self):
        fake_collection = MagicMock()
        fake_collection.count.return_value = 0
        with patch.object(rag, "get_collection", return_value=fake_collection):
            result = rag.search_semantic_ranked("query", n_results=5)
        self.assertEqual(result, [])


class TestIndexVaultCallsFts(unittest.TestCase):
    def test_index_vault_dry_run_includes_fts_report(self):
        fake_collection = MagicMock()
        fake_collection.get.return_value = {"ids": [], "metadatas": []}
        with patch.object(rag, "get_collection", return_value=fake_collection), \
             patch("vault_mcp_rag.VAULT_ROOT") as fake_root:
            fake_root.__truediv__.return_value.rglob.return_value = []
            with patch.object(rag.fts, "index_fts", return_value="[dry_run FTS5] fake report") as mock_fts:
                report = rag.index_vault(dry_run=True)
        mock_fts.assert_called_once_with(dry_run=True)
        self.assertIn("fake report", report)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_rag_ranked.py -v`
Expected: FAIL — `rag.search_semantic_ranked` and `rag.fts` don't exist yet.

- [ ] **Step 3: Add `search_semantic_ranked` and wire `fts` into `index_vault`**

In `src/vault_mcp/rag.py`, add this import after the existing `from vault_mcp.vault import VAULT_ROOT` line:

```python
from vault_mcp import fts
```

Add this new function right after `search_semantic` (at the end of the file):

```python
def search_semantic_ranked(query: str, n_results: int = 5) -> list[tuple[str, float]]:
    """Ranked vector search. Returns (page_id, score) sorted best-first,
    score higher = better (Chroma's cosine distance is lower-is-better,
    inverted here to 1 - distance so callers like _rrf_fuse can treat
    every signal the same way). Returns [] if the index is empty."""
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = _embed([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    ids = results["ids"][0]
    distances = results["distances"][0]
    return [(page_id, 1 - distance) for page_id, distance in zip(ids, distances)]
```

Then, in `index_vault`, change the final `return` statement (currently the last lines of the function, the "Indexadas N páginas..." f-string) so its result is captured in a variable and the FTS5 report is appended. Replace:

```python
    return (
        f"Indexadas {len(to_index_ids)} páginas nuevas/modificadas, "
        f"{len(stale_ids)} obsoleta(s) borrada(s), "
        f"{len(current) - len(to_index_ids)} sin cambios (salteadas)."
    )
```

with:

```python
    chroma_report = (
        f"Indexadas {len(to_index_ids)} páginas nuevas/modificadas, "
        f"{len(stale_ids)} obsoleta(s) borrada(s), "
        f"{len(current) - len(to_index_ids)} sin cambios (salteadas)."
    )
    fts_report = fts.index_fts(dry_run=dry_run)
    return f"{chroma_report}\n{fts_report}"
```

Also find the earlier `dry_run` early-return in `index_vault` (the `if dry_run:` block near the top of the function that returns before touching the collection) and apply the same pattern — append `fts.index_fts(dry_run=True)`'s report to it, so a dry run reports both indexes too:

```python
    if dry_run:
        chroma_report = (
            f"[dry_run] {len(to_index_ids)} páginas nuevas/modificadas para embeber, "
            f"{len(stale_ids)} páginas obsoletas para borrar, "
            f"{len(current) - len(to_index_ids)} sin cambios (se saltean). "
            f"Ejemplo a embeber: {to_index_ids[:3]}"
        )
        fts_report = fts.index_fts(dry_run=True)
        return f"{chroma_report}\n{fts_report}"
```

And the `if not to_index_ids:` early-return branch (nothing new to embed) similarly:

```python
    if not to_index_ids:
        fts_report = fts.index_fts(dry_run=dry_run)
        return f"Nada para embeber — las {len(current)} páginas ya estaban al día. {len(stale_ids)} obsoleta(s) borrada(s).\n{fts_report}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_rag_ranked.py -v`
Expected: all PASS.

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS (no regression in `test_fts.py` or `test_hot_serve_budget.py`).

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Ivan/Proyectos/vault-mcp-server
git add src/vault_mcp/rag.py tests/test_rag_ranked.py
git commit -m "feat: search_semantic_ranked + index_vault also maintains FTS5 index"
```

---

### Task 4: `_rrf_fuse` pure function in `vault.py`

**Files:**
- Modify: `src/vault_mcp/vault.py` (add near the `get_related_pages`/graph section, after `get_related_pages`'s closing line)
- Test: `tests/test_rrf_fuse.py`

**Interfaces:**
- Produces: `_rrf_fuse(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rrf_fuse.py`:

```python
import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault_rrf", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault_rrf"] = vault
spec.loader.exec_module(vault)


class TestRrfFuse(unittest.TestCase):
    def test_single_list_preserves_order(self):
        result = vault._rrf_fuse([[("a", 9.0), ("b", 5.0), ("c", 1.0)]], k=60)
        self.assertEqual([p for p, _ in result], ["a", "b", "c"])

    def test_agreement_across_lists_wins(self):
        # "b" is #2 in both lists; "a" is #1 in one but absent from the other.
        list1 = [("a", 9.0), ("b", 5.0)]
        list2 = [("b", 8.0), ("c", 3.0)]
        result = vault._rrf_fuse([list1, list2], k=60)
        top = result[0][0]
        self.assertEqual(top, "b")

    def test_hand_computed_scores(self):
        # k=1 for easy hand math: score(page) = sum(1 / (1 + rank)) over
        # the lists it appears in, rank is 0-indexed.
        list1 = [("a", 1.0), ("b", 1.0)]  # a: rank 0, b: rank 1
        list2 = [("b", 1.0), ("a", 1.0)]  # b: rank 0, a: rank 1
        result = vault._rrf_fuse([list1, list2], k=1)
        scores = dict(result)
        # a: 1/(1+0) + 1/(1+1) = 1.0 + 0.5 = 1.5
        # b: 1/(1+1) + 1/(1+0) = 0.5 + 1.0 = 1.5
        self.assertAlmostEqual(scores["a"], 1.5)
        self.assertAlmostEqual(scores["b"], 1.5)

    def test_page_missing_from_one_list_still_scored(self):
        list1 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
        list2 = [("c", 9.0)]  # "a" and "b" absent here
        result = vault._rrf_fuse([list1, list2], k=60)
        page_ids = [p for p, _ in result]
        self.assertIn("a", page_ids)
        self.assertIn("b", page_ids)
        self.assertIn("c", page_ids)

    def test_empty_lists_return_empty(self):
        self.assertEqual(vault._rrf_fuse([[], []], k=60), [])
        self.assertEqual(vault._rrf_fuse([], k=60), [])

    def test_dedup_by_page_id(self):
        # Same page_id appearing more than once within a single list
        # (shouldn't happen from real callers, but must not double-count
        # silently into a broken score) — last occurrence's rank wins.
        result = vault._rrf_fuse([[("a", 9.0), ("a", 1.0)]], k=60)
        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_rrf_fuse.py -v`
Expected: FAIL — `vault._rrf_fuse` doesn't exist yet.

- [ ] **Step 3: Add `_rrf_fuse` to `vault.py`**

Add this function to `src/vault_mcp/vault.py`, directly after `get_related_pages`'s closing `return "\n".join(lines)` (end of that function):

```python
def _rrf_fuse(
    ranked_lists: list[list[tuple[str, float]]], k: int = 60
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion: combine N ranked (page_id, score) lists into
    one, using each item's *rank position* within its own list (not its raw
    score, which isn't comparable across signals like BM25/cosine-similarity/
    graph-weight). score(page) = sum(1 / (k + rank)) over every list it
    appears in, rank is 0-indexed. Standard k=60 (Cormack et al. 2009).

    A page_id repeated within a single input list is deduped to its last
    occurrence's rank before scoring."""
    scores: dict[str, float] = {}
    for ranked_list in ranked_lists:
        rank_by_page: dict[str, int] = {}
        for rank, (page_id, _score) in enumerate(ranked_list):
            rank_by_page[page_id] = rank
        for page_id, rank in rank_by_page.items():
            scores[page_id] = scores.get(page_id, 0.0) + 1.0 / (k + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_rrf_fuse.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
cd C:/Users/Ivan/Proyectos/vault-mcp-server
git add src/vault_mcp/vault.py tests/test_rrf_fuse.py
git commit -m "feat: add _rrf_fuse (reciprocal rank fusion) to vault.py"
```

---

### Task 5: `search_hybrid` tool — wire everything together

**Files:**
- Modify: `src/vault_mcp/vault.py` (add after `_rrf_fuse`)
- Modify: `src/vault_mcp/server.py` (register the new tool)
- Test: `tests/test_search_hybrid.py`

**Interfaces:**
- Consumes: `fts.search_wiki_ranked` (Task 1), `rag.search_semantic_ranked` (Task 3), `vault._rrf_fuse` (Task 4), `vault.get_related_pages` (existing).
- Produces: `search_hybrid(query: str, n_results: int = 5) -> str` (public tool).

- [ ] **Step 1: Write the failing test**

Create `tests/test_search_hybrid.py`:

`search_hybrid` lazy-imports `fts`/`rag` *inside* the function (`from vault_mcp import fts, rag` — see Step 3), so they never become attributes of the `vault` module object itself; `patch("vault_mcp_vault_hybrid.fts", ...)` would fail with `AttributeError` since that name doesn't exist there. The lazy import re-fetches whatever is currently in `sys.modules["vault_mcp.fts"]`/`sys.modules["vault_mcp.rag"]` each call, so the correct patch target is the *real, installed* `vault_mcp.fts`/`vault_mcp.rag` package modules (confirmed importable: `vault_mcp.__file__` resolves to this repo's own `src/vault_mcp/__init__.py`, i.e. it's installed editable in `.venv`) — not the ad-hoc module this test loads via `spec_from_file_location` to reach `search_hybrid` itself:

```python
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).parent.parent / "src" / "vault_mcp" / "vault.py"
spec = importlib.util.spec_from_file_location("vault_mcp_vault_hybrid", MODULE_PATH)
vault = importlib.util.module_from_spec(spec)
sys.modules["vault_mcp_vault_hybrid"] = vault
spec.loader.exec_module(vault)


class TestSearchHybrid(unittest.TestCase):
    def test_no_results_anywhere(self):
        with patch("vault_mcp.fts.search_wiki_ranked", return_value=[]), \
             patch("vault_mcp.rag.search_semantic_ranked", return_value=[]):
            result = vault.search_hybrid("query nada")
        self.assertEqual(result, "Sin resultados para 'query nada'")

    def test_fuses_bm25_and_vector_and_graph(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked",
            return_value=[("a.md", 5.0), ("b.md", 2.0)],
        ), patch(
            "vault_mcp.rag.search_semantic_ranked",
            return_value=[("b.md", 0.9), ("c.md", 0.5)],
        ), patch.object(vault, "get_related_pages") as mock_related:
            # get_related_pages is called once per stage-1 anchor page; return
            # a page not present in the BM25/vector results, to prove graph
            # results make it into the final fusion. Real format:
            # "- [[page]] (score) — breakdown", see vault.py:1473-1481.
            mock_related.return_value = "- [[d.md]] (3.0) — link directo"

            result = vault.search_hybrid("query", n_results=5)

        self.assertIn("d.md", result)
        self.assertIn("b.md", result)  # appears in both bm25 and vector, should rank high

    def test_only_one_signal_has_results(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked", return_value=[("a.md", 5.0)]
        ), patch(
            "vault_mcp.rag.search_semantic_ranked", return_value=[]
        ), patch.object(vault, "get_related_pages") as mock_related:
            mock_related.return_value = "- Ninguna señal de relación encontrada (página aislada)."

            result = vault.search_hybrid("solo bm25")

        self.assertIn("a.md", result)

    def test_anchor_page_that_does_not_resolve_does_not_crash(self):
        with patch(
            "vault_mcp.fts.search_wiki_ranked", return_value=[("orphan.md", 5.0)]
        ), patch(
            "vault_mcp.rag.search_semantic_ranked", return_value=[]
        ), patch.object(vault, "get_related_pages") as mock_related:
            # get_related_pages returns a warning string (its real behavior
            # for a nonexistent page), not an exception.
            mock_related.return_value = "⚠️ No existe ninguna página `orphan.md.md` en wiki/"

            result = vault.search_hybrid("orphan query")

        self.assertIn("orphan.md", result)


if __name__ == "__main__":
    unittest.main()
```

Note: patching `vault_mcp.rag.search_semantic_ranked` requires importing the real `vault_mcp.rag` module, which pulls in its heavy torch/chromadb import chain (documented at the top of `rag.py`) — this test file will be slower than the others as a result. That's an accepted cost, not a bug to fix.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_search_hybrid.py -v`
Expected: FAIL — `vault.search_hybrid` doesn't exist yet.

- [ ] **Step 3: Add `search_hybrid` to `vault.py`**

First, add a small parser for `get_related_pages`'s markdown-list output back into `(page_id, score)` tuples — it currently only returns formatted text, and `search_hybrid` needs the ranked pairs. `get_related_pages`'s real output format (confirmed by reading `src/vault_mcp/vault.py:1473-1481`) is one line per result: `- [[candidate]] (7.5) — link directo, co-citado por 2 fuente(s)` — the score sits directly in parens right after the wikilink, no `score:` label. Its "nothing found" case is the literal line `- Ninguna señal de relación encontrada (página aislada).`, which has no `[[...]]` and so naturally produces zero matches below. Add this helper right before `search_hybrid`:

```python
_RELATED_PAGE_LINE = re.compile(r"\[\[([^\]]+)\]\]\s*\(([0-9.]+)\)")


def _parse_related_pages_output(text: str) -> list[tuple[str, float]]:
    """Extract (page_id, score) pairs from get_related_pages' formatted
    output (lines like '- [[page]] (7.5) — breakdown'). Returns [] for
    its warning-string case (nonexistent page) or its no-signal case
    ('Ninguna señal de relación encontrada') — neither contains a
    [[wikilink]](score) pair, so search_hybrid treats both as "no graph
    signal from this anchor", not an error."""
    return [(name, float(score)) for name, score in _RELATED_PAGE_LINE.findall(text)]
```

Then add `search_hybrid` itself, after `_rrf_fuse`:

```python
def search_hybrid(query: str, n_results: int = 5) -> str:
    """Fuse BM25 (fts.search_wiki_ranked), vector search
    (rag.search_semantic_ranked), and graph relatedness (get_related_pages)
    with reciprocal rank fusion, per docs/superpowers/specs/
    2026-08-21-search-hybrid-rrf-design.md.

    Lazy-imports rag/fts (same pattern as the existing lazy import at the
    top of run_lint's index-drift check) to avoid the native DLL
    load-order conflict documented in rag.py if chromadb-adjacent imports
    happened at vault.py's module level.
    """
    from vault_mcp import fts, rag

    bm25_ranked = fts.search_wiki_ranked(query, max_results=20)
    vector_ranked = rag.search_semantic_ranked(query, n_results=20)

    stage1 = _rrf_fuse([bm25_ranked, vector_ranked], k=60)
    anchors = [page_id for page_id, _score in stage1[:5]]

    graph_ranked: list[tuple[str, float]] = []
    for anchor in anchors:
        related_text = get_related_pages(anchor, top_n=5)
        graph_ranked.extend(_parse_related_pages_output(related_text))

    final = _rrf_fuse([stage1, graph_ranked], k=60)[:n_results]

    if not final:
        return f"Sin resultados para '{query}'"

    bm25_pages = {p for p, _ in bm25_ranked}
    vector_pages = {p for p, _ in vector_ranked}
    graph_pages = {p for p, _ in graph_ranked}

    lines = [f"# Resultados híbridos (BM25+vectorial+grafo): '{query}'", ""]
    for page_id, score in final:
        signals = []
        if page_id in bm25_pages:
            signals.append("BM25")
        if page_id in vector_pages:
            signals.append("vectorial")
        if page_id in graph_pages:
            signals.append("grafo")
        lines.append(f"## {page_id} (score: {score:.4f}, señales: {', '.join(signals) or '?'})")
    return "\n".join(lines)
```

Add `import re` at the top of `vault.py` if it isn't already imported (check — `WIKILINK_PATTERN` at line 964 already uses `re.compile`, so `re` should already be imported; confirm before adding a duplicate import).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/test_search_hybrid.py -v`
Expected: all PASS.

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -m pytest tests/ -v`
Expected: all PASS (full suite, no regressions).

- [ ] **Step 5: Register `search_hybrid` as an MCP tool in `server.py`**

In `src/vault_mcp/server.py`, add (near the other search tools — `search_wiki`, `search_semantic`, `get_related_pages`):

```python
@mcp.tool()
def search_hybrid(query: str, n_results: int = 5) -> str:
    """Fused search: BM25 (full-text) + vector (semantic) + graph
    (relatedness via wikilinks), combined with reciprocal rank fusion.
    Use this instead of calling search_wiki/search_semantic/
    get_related_pages separately when you want one ranked answer that
    accounts for all three signals."""
    return vault.search_hybrid(query, n_results=n_results)
```

- [ ] **Step 6: Manual smoke test against the real vault**

Run: `cd C:/Users/Ivan/Proyectos/vault-mcp-server && .venv/Scripts/python.exe -c "
import sys; sys.path.insert(0, 'src')
from vault_mcp import rag, vault
rag.index_vault(dry_run=False)
print(vault.search_hybrid('vault-mcp second brain', n_results=5))
"`
Expected: real output with `## page_id (score: ..., señales: ...)` blocks, no traceback. This is the first real (non-mocked) run against the actual vault at `wiki/` — confirms the FTS5 index built from real content, the real `get_related_pages` output format matches what `_parse_related_pages_output` expects, and the whole chain works end-to-end outside of mocks.

- [ ] **Step 7: Commit**

```bash
cd C:/Users/Ivan/Proyectos/vault-mcp-server
git add src/vault_mcp/vault.py src/vault_mcp/server.py tests/test_search_hybrid.py
git commit -m "feat: add search_hybrid tool (RRF over BM25+vector+graph)"
```

---

## Post-plan note (not a task — informational)

After this plan lands, `vault-mcp`'s "Qué es" description in `wiki/entidades/vault-mcp.md` and the tool list in `CLAUDE.md`'s INICIO DE SESIÓN section will be stale (they enumerate the current tool set without `search_hybrid`, and describe `search_wiki` as plain text search). Updating those wiki pages is a normal `wiki/` edit, not part of this code plan — do it as a follow-up once `search_hybrid` is verified working (Task 5, Step 6), the same way `vault-mcp.md`'s "Ideas a futuro" section was already updated on 2026-08-21 to reference this spec.
