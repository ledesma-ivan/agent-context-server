"""MCP server entrypoint — exposes vault.py functions as tools."""

# vault_mcp (rag.py) must import before mcp.server.fastmcp: FastMCP's own
# dependency chain (pydantic-core et al.) triggers the same native DLL
# load-order conflict documented in rag.py if it loads first.
from vault_mcp import fts, rag, vault

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("vault-mcp")


@mcp.tool()
def get_hot() -> str:
    """Return the full contents of wiki/hot.md — the vault's active context."""
    return vault.get_hot()


@mcp.tool()
def get_prioridades() -> str:
    """Return wiki/hot.md's 'Cuellos de botella activos' section — the short,
    canonical list of active blockers. Call this before answering any
    synthesis question about priorities/bottlenecks/status instead of
    answering from conversational memory alone."""
    return vault.get_prioridades()


@mcp.tool()
def get_pendientes() -> str:
    """Return the 'Decisiones pendientes' section of wiki/hot.md, budgeted
    to ~28K chars and ranked by most-recent-date-mentioned — a short,
    focused slice for callers with a small context budget (e.g. the Hermes
    heartbeat), where get_hot()'s full ~100K chars gets truncated before
    reaching this section. Older items past the budget are listed by title
    only, with a pointer to search_semantic and pendientes-archivo.md."""
    return vault.get_pendientes()


@mcp.tool()
def archive_pendiente(marker: str) -> str:
    """Move one open (not yet resolved) item out of hot.md's 'Decisiones
    pendientes' into wiki/pendientes-archivo.md, where it stays real,
    searchable knowledge (indexable by search_semantic) without weighing
    down every get_pendientes() call. `marker` must be a unique substring
    of the item's text — refuses to act if it matches zero or several
    items."""
    return vault.archive_pendiente(marker)


@mcp.tool()
def add_pendiente(texto: str, force: bool = False) -> str:
    """Add a new open item to hot.md's 'Decisiones pendientes' section --
    counterpart to archive_pendiente, which only removes/relocates an item
    that already exists. Use this whenever the user asks to note, remember,
    or track something new as a pendiente (there was previously no way to
    create one via MCP, only to archive an existing one — a real gap found
    in Hermes' Telegram usage). If `texto` doesn't already include a
    dd/mm/yyyy date, today's date is appended automatically so it doesn't
    show up as stale/undated right away.

    Dedup check (added 1/8/2026): before writing, compares `texto` against
    every open item already in the section. If a near-duplicate exists, the
    write is skipped and the existing item's text is returned instead — set
    force=True to add anyway if it's genuinely a different item."""
    return vault.add_pendiente(texto, force=force)


@mcp.tool()
def add_tarea_diaria(texto: str) -> str:
    """Add a new open task to today's daily note ('## ✅ Tareas'), instead of
    hot.md. Counterpart to add_pendiente for items with a ~1-day execution
    horizon (convención adoptada 29/7/2026): use this for something
    actionable today/this week, use add_pendiente for an open-ended
    decision without a concrete action date. Fails if today's note doesn't
    exist yet or has no '## ✅ Tareas' section."""
    return vault.add_tarea_diaria(texto)


@mcp.tool()
def resolve_pendiente(marker: str) -> str:
    """Mark one 'Decisiones pendientes' item as resolved and move it to
    wiki/log.md — counterpart to archive_pendiente (which is for items that
    are still open but stale, not finished). Use this when the user says a
    pendiente is actually done. `marker` must be a unique substring of the
    item's text — refuses to act if it matches zero or several items."""
    return vault.resolve_pendiente(marker)


@mcp.tool()
def get_index() -> str:
    """Return the full contents of wiki/index.md — the vault's master index."""
    return vault.get_index()


@mcp.tool()
def search_wiki(query: str, max_results: int = 20) -> str:
    """Full-text search across every wiki/**/*.md file, ranked by BM25
    relevance (SQLite FTS5)."""
    return fts.search_wiki(query, max_results=max_results)


@mcp.tool()
def get_page(name: str) -> str:
    """Resolve a [[wikilink]] name (e.g. 'ivan-ledesma') to its file contents."""
    return vault.get_page(name)


@mcp.tool()
def get_backlinks(name: str) -> str:
    """Return every wiki page that links to [[name]] via wikilink — the
    reverse of get_page's forward resolution ('what links here')."""
    return vault.get_backlinks(name)


@mcp.tool()
def get_related_pages(name: str, top_n: int = 8) -> str:
    """Rank every other wiki page by weighted relevance to [[name]] (direct
    link, co-citation by a shared fuente, Adamic-Adar over common neighbors,
    same-type bonus) and return the top N. A heuristic for surfacing cluster/
    síntesis candidates beyond exact tag/entity matches — not exact like
    get_backlinks, always eyeball the result."""
    return vault.get_related_pages(name, top_n)


@mcp.tool()
def prune_hot(dry_run: bool = True) -> str:
    """Compact wiki/hot.md's 'Última sesión' section: entries older than 3
    days with a log.md pointer get collapsed to a 1-line stub (already
    preserved there in full). Entries within 3 days, or without a log.md
    pointer, are left untouched. Returns the report; only writes to hot.md
    when dry_run=False."""
    return vault.prune_hot(dry_run=dry_run)


@mcp.tool()
def prune_brechas(dry_run: bool = True) -> str:
    """Compact wiki/hot.md's '### Brechas documentadas' section (under
    '## Fallas abiertas y pendientes'): bullets marked '**cerrado/resuelto
    (DD/MM/YYYY)**' whose date has a matching '## YYYY-MM-DD' section in
    log.md, and that don't also mention an unresolved 'pendiente' elsewhere
    in the bullet, get collapsed to a 1-line stub pointing at log.md.
    Everything else is left untouched and reported separately. Returns the
    report; only writes to hot.md when dry_run=False."""
    return vault.prune_brechas(dry_run=dry_run)


@mcp.tool()
def get_local_events(target_date: str | None = None, dias: int = 1) -> str:
    """Eventos del calendario local (Calendario/, Full Calendar Remastered)
    para una fecha (YYYY-MM-DD, default hoy) y rango de días. 100% local,
    no usa Google Calendar."""
    return vault.get_local_events(target_date=target_date, dias=dias)


@mcp.tool()
def check_calendar_overlaps(target_date: str | None = None, dias: int = 14) -> str:
    """Chequea TODO el rango de días de una sola pasada por solapamientos
    entre eventos recurrentes/únicos de Calendario/, en vez de descubrirlos
    uno a la vez cuando ya chocaron. Solo reporta, no modifica nada."""
    return vault.check_calendar_overlaps(target_date=target_date, dias=dias)


@mcp.tool()
def create_page(page_type: str, name: str, content: str, metadata: dict) -> str:
    """Create a new wiki page (fuente/entidad/concepto/sintesis), enforcing
    the required frontmatter fields for that type. Refuses to overwrite."""
    return vault.create_page(page_type, name, content, metadata)


@mcp.tool()
def move_source_file(inbox_path: str, target_folder: str) -> str:
    """Move a file from _Inbox/ to its PARA destination (step 9 of INGERIR).
    target_folder must be relative to the vault root and rooted at one of
    Notas/Areas/Proyectos/Archivo/Recursos. Refuses to overwrite an existing
    file at the destination."""
    return vault.move_source_file(inbox_path, target_folder)


@mcp.tool()
def check_ingested(paths: list[str]) -> str:
    """Check which _Inbox/ files (paths relative to _Inbox/) were already
    evaluated in a previous INGERIR run, by content hash — not filename.
    Call once for the whole batch, before reading any file, at the start of
    INGERIR. Replaces the manual grep of wiki/log.md for prior verdicts;
    covers every verdict (pasa/descartado/fragmento_extraido/
    link_con_anotacion), not just discards."""
    return vault.check_ingested(paths)


@mcp.tool()
def register_ingested(path: str, veredicto: str, motivo: str, pagina: str | None = None) -> str:
    """Record the final INGERIR verdict for one _Inbox/ file in the manifest,
    keyed by content hash. veredicto must be one of: pasa, descartado,
    fragmento_extraido, link_con_anotacion. Call once per file when its
    verdict is final."""
    return vault.register_ingested(path, veredicto, motivo, pagina)


@mcp.tool()
def build_index(dry_run: bool = True) -> str:
    """Regenerate wiki/index.md from page frontmatter. Returns the generated
    markdown; only overwrites the file when dry_run=False."""
    return vault.build_index(dry_run=dry_run)


@mcp.tool()
def run_lint() -> str:
    """Run the vault LINT checks (broken links, unclosed contradictions,
    orphan pages) and return a text report."""
    return vault.run_lint()


@mcp.tool()
def fix_broken_links(dry_run: bool = True) -> str:
    """For every broken [[wikilink]] run_lint would flag, fuzzy-match the
    target against real page names (typos/renames only, not true aliases)
    and propose a fix. dry_run=True (default) only reports; dry_run=False
    rewrites the fixable links in place."""
    return vault.fix_broken_links(dry_run=dry_run)


@mcp.tool()
def doctor() -> str:
    """Health-check the vault-mcp infra itself (not content quality): wiki/
    subfolders, core files, ingest manifest, frontmatter validity, semantic
    index sync. Call this when something feels broken, or periodically to
    catch silent breakage before a session depends on it."""
    return vault.doctor()


@mcp.tool()
def index_vault(dry_run: bool = True) -> str:
    """(Re)build the semantic search index over wiki/**/*.md. Only writes
    to the Chroma collection when dry_run=False."""
    return rag.index_vault(dry_run=dry_run)


@mcp.tool()
def search_semantic(query: str, n_results: int = 5) -> str:
    """Semantic (meaning-based) search across the vault, as opposed to
    search_wiki's literal text match."""
    return rag.search_semantic(query, n_results=n_results)


@mcp.tool()
def search_hybrid(query: str, n_results: int = 5) -> str:
    """Fused search: BM25 (full-text) + vector (semantic) + graph
    (relatedness via wikilinks), combined with reciprocal rank fusion.
    Use this instead of calling search_wiki/search_semantic/
    get_related_pages separately when you want one ranked answer that
    accounts for all three signals."""
    return vault.search_hybrid(query, n_results=n_results)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
