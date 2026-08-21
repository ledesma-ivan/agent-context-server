"""Filesystem helpers for reading the Obsidian vault."""

import difflib
import hashlib
import json
import math
import os
import re
import shutil
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path

import frontmatter

# Path to the vault on disk. Reads VAULT_MCP_ROOT if set (portable across
# machines/vaults — needed once this repo is shared publicly, since a
# hardcoded personal path doesn't make sense for anyone else running it);
# falls back to Ivan's own vault so nothing breaks locally without a .env.
VAULT_ROOT = Path(os.environ.get("VAULT_MCP_ROOT", r"C:\Users\Ivan\Desktop\Diario"))

ARCHIVE_PATH = VAULT_ROOT / "wiki" / "pendientes-archivo.md"

# Items are parsed as "- [ ] ..." or "- [x] ..." up to the next top-level
# bullet or end of section — same shape used by run_lint's resolved-item scan.
ITEM_PATTERN = re.compile(r"^- \[[ xX]\].*?(?=\n- \[|\Z)", re.DOTALL | re.MULTILINE)
DATE_PATTERN = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

# Mismo formato de ruta que usan los scripts de Hermes en
# hermes/scripts/horas_autoasignar.py y sync_tareas_horas.py -- duplicado acá
# a propósito en vez de importar entre repos (mismo criterio ya usado por
# modo_bienestar.py para _matches_days_of_week, ver hot.md sesión 4/7/2026).
MESES_ES = {
    1: "Enero", 2: "Febrero", 3: "Marzo", 4: "Abril", 5: "Mayo", 6: "Junio",
    7: "Julio", 8: "Agosto", 9: "Septiembre", 10: "Octubre", 11: "Noviembre",
    12: "Diciembre",
}
TAREAS_SECTION_RE = re.compile(r"## ✅ Tareas\n(.*?)\n---", re.DOTALL)

# Servable budget for get_pendientes(). Kept below run_lint's 40k warn
# threshold and well under Hermes' ~8195-token real input ceiling (~32-33k
# chars) so a caller with a small context window never sees the truncation
# bug this tool was built to fix in the first place (see wiki/log.md,
# sesión 3/7/2026 — get_hot() truncated before reaching pendientes).
PENDIENTES_SERVE_BUDGET_CHARS = 28_000

# Un ítem "pendiente" sin fecha reciente mencionada es exactamente el patrón
# detectado en el análisis de comportamiento del log (8/7/2026): se anota
# pero nunca se cierra. Medir la edad en vez de confiar en que alguien lo
# note a ojo (mismo principio que el trust_ledger de hermes-swing-iem).
PENDIENTE_VENCIDO_DIAS = 14


# hot.md oscillates in a ~100K-150K character band (kept there by the
# weekly vault-prune-hot-auto cron — see wiki/log.md 2026-08-21) but never
# had its own serve budget: every session paid that full floor as its
# first read. Only '## Última sesión' gets trimmed here, since it already
# has its own compaction mechanism (prune_hot(), pointing to log.md) — a
# summarized entry there loses nothing that isn't already recoverable.
# Every other section is always served whole: they're smaller and more
# load-bearing, and truncating them top-down was the exact failure mode
# that motivated get_pendientes() as a separate tool in the first place.
HOT_SERVE_BUDGET_CHARS = 100_000


def get_hot() -> str:
    """Return wiki/hot.md, budgeted to HOT_SERVE_BUDGET_CHARS."""
    text = (VAULT_ROOT / "wiki" / "hot.md").read_text(encoding="utf-8")
    return _budget_hot_text(text)


def _budget_hot_text(text: str, budget: int = HOT_SERVE_BUDGET_CHARS) -> str:
    """Trim only '## Última sesión' so the whole file fits `budget` chars.

    Entries are ranked by their own most-recent mentioned date (most recent
    first) and included greedily until the per-session budget is spent —
    same pattern as get_pendientes(). Excluded entries are never dropped
    silently: they get a one-line summary with a pointer to wiki/log.md
    (where prune_hot() already sends anything it compacts) and to hot.md
    itself for the rare entry without a log.md pointer yet.
    """
    if len(text) <= budget:
        return text

    section_match = _ultima_sesion_section(text)
    if not section_match:
        return text

    sec_start, sec_end = section_match.span()
    section = text[sec_start:sec_end]
    resto_size = len(text) - len(section)
    budget_for_session = budget - resto_size

    header, _, body = section.partition("\n")
    body = body.strip("\n")
    if not body.strip():
        return text

    entries = [e.strip() for e in re.split(r"\n\n(?=\*\*Sesión )", body) if e.strip()]
    if not entries:
        return text

    ranked = sorted(entries, key=lambda e: _entry_own_date(e) or date.min, reverse=True)

    included: list[str] = []
    included_size = len(header) + 2
    excluded: list[str] = []
    for entry in ranked:
        if included and included_size + len(entry) + 2 > budget_for_session:
            excluded.append(entry)
            continue
        if not included and len(entry) + 2 > budget_for_session:
            excluded.append(entry)
            continue
        included.append(entry)
        included_size += len(entry) + 2

    if not excluded:
        return text

    included_in_order = [e for e in entries if e in included]

    lines = [header, ""]
    if included_in_order:
        lines.append("\n\n".join(included_in_order))
        lines.append("")
    lines.append(
        f"_{len(excluded)} entrada(s) más antigua(s) no incluidas por presupuesto de tamaño "
        f"({budget:,} caracteres totales). Nada se perdió — revisá `wiki/log.md` o "
        f"`wiki/hot.md` completo si necesitás el detalle:_"
    )
    for entry in excluded:
        entry_date = _entry_own_date(entry)
        fecha_txt = entry_date.isoformat() if entry_date else "sin fecha"
        lines.append(f"- {fecha_txt} — {_entry_summary_line(entry)}")

    new_section = "\n".join(lines).strip() + "\n"
    return text[:sec_start] + new_section + text[sec_end:]


def _cuellos_de_botella_section(text: str) -> re.Match | None:
    return re.search(r"^## Cuellos de botella activos\n.*?(?=\n## |\Z)", text, re.DOTALL | re.MULTILINE)


def get_prioridades() -> str:
    """Return wiki/hot.md's 'Cuellos de botella activos' section — the short,
    canonical list of active blockers (mudanza, inglés hablado, búsqueda
    laboral, etc.), deliberately capped at 3-5 items.

    Call this before answering any synthesis question about priorities,
    bottlenecks, or "what's going on" in a broad sense — instead of
    answering from conversational memory alone. Added 12/7/2026 after a real
    miss: a synthesis answer skipped 'búsqueda laboral' because it was
    documented but buried in narrative prose, not because the fact was
    missing. This doesn't guarantee the model calls it — it just makes the
    correct path (one cheap tool call) cheaper than the wrong one (re-reading
    all of hot.md or trusting memory).
    """
    text = get_hot()
    section_match = _cuellos_de_botella_section(text)
    if not section_match:
        return "No se encontró la sección 'Cuellos de botella activos' en hot.md."
    section = section_match.group(0).strip()
    # Strip a trailing "---" divider line if the section boundary swallowed
    # it (the divider before the next ## header lives inside the match).
    section = re.sub(r"\n+---\s*$", "", section).strip()
    return section


def _pendientes_section(text: str) -> re.Match | None:
    # "###" desde la reestructuración de hot.md (3/7/2026, 4 secciones ##
    # con subsecciones ###) — el regex viejo buscaba "##" exacto y quedó
    # roto en silencio desde entonces (bug real encontrado 8/7/2026, mismo
    # patrón que el healthcheck de infra: nadie llamó get_pendientes() para
    # notar que ya no matcheaba nada).
    return re.search(r"^### Decisiones pendientes\n.*?(?=\n## |\n### |\Z)", text, re.DOTALL | re.MULTILINE)


def _item_recency(item_text: str) -> date:
    """Most recent dd/mm/yyyy date mentioned in the item, or date.min if none.

    Items without a date sort as oldest — safer default than treating an
    undated item as urgent.
    """
    dates = []
    for day, month, year in DATE_PATTERN.findall(item_text):
        try:
            dates.append(date(int(year), int(month), int(day)))
        except ValueError:
            continue
    return max(dates) if dates else date.min


def _item_age_marker(item_text: str) -> str | None:
    """⚠️ marker for an item whose most recent mentioned date is stale, or
    that never had a date to begin with. None if it's recent enough."""
    recency = _item_recency(item_text)
    if recency == date.min:
        return "⚠️ SIN FECHA (revisar)"
    dias = (date.today() - recency).days
    if dias > PENDIENTE_VENCIDO_DIAS:
        return f"⚠️ VENCIDO ({dias}d)"
    return None


def _item_title(item_text: str) -> str:
    bold = re.search(r"\*\*(.+?)\*\*", item_text)
    if bold:
        return bold.group(1).strip()
    return item_text.splitlines()[0][:80].strip()


def get_pendientes() -> str:
    """Return the most relevant slice of wiki/hot.md's 'Decisiones pendientes'
    section, budgeted to PENDIENTES_SERVE_BUDGET_CHARS.

    hot.md as a whole is tens of thousands of characters — a small model
    with a limited context budget (e.g. Hermes' autonomous heartbeat) gets
    it truncated before reaching this section, which lives near the bottom.
    This gives callers that only care about open items a short, focused
    slice instead of the full narrative history.

    Items are ranked by the most recent date mentioned in their own text
    (most recent first) and included greedily until the budget is spent —
    older items get dropped from the response, not silently truncated
    mid-item, and are listed by title only with a pointer to search_semantic
    and wiki/pendientes-archivo.md so nothing is actually lost, just not
    force-fed into every call.
    """
    text = (VAULT_ROOT / "wiki" / "hot.md").read_text(encoding="utf-8")
    section_match = _pendientes_section(text)
    if not section_match:
        return "No se encontró la sección 'Decisiones pendientes' en hot.md."

    section = section_match.group(0).strip()
    header, _, body = section.partition("\n")
    items = [m.group(0) for m in ITEM_PATTERN.finditer(body)]

    if not items:
        return section

    ranked = sorted(items, key=_item_recency, reverse=True)

    included: list[str] = []
    included_size = len(header) + 2
    excluded: list[str] = []
    for item in ranked:
        if included and included_size + len(item) + 2 > PENDIENTES_SERVE_BUDGET_CHARS:
            excluded.append(item)
            continue
        included.append(item)
        included_size += len(item) + 2

    if not excluded:
        marked_lines = [header, ""]
        for item in items:
            marker = _item_age_marker(item)
            if marker:
                marked_lines.append(marker)
            marked_lines.append(item)
        return "\n".join(marked_lines).strip()

    # Re-sort included items back to their original document order so the
    # output reads the same as hot.md, not shuffled by recency.
    included_in_order = [item for item in items if item in included]

    lines = [header, ""]
    for item in included_in_order:
        marker = _item_age_marker(item)
        if marker:
            lines.append(f"{marker}")
        lines.append(item)
    lines.append("")
    lines.append(
        f"_{len(excluded)} ítem(s) más antiguo(s) no incluidos por presupuesto de tamaño "
        f"({PENDIENTES_SERVE_BUDGET_CHARS:,} caracteres). Nada se perdió — usá search_semantic "
        f"o revisá `wiki/pendientes-archivo.md` / `wiki/hot.md` completo si necesitás el detalle:_"
    )
    for item in excluded:
        marker = _item_age_marker(item)
        prefix = f"{marker} " if marker else ""
        lines.append(f"- {prefix}{_item_title(item)}")

    return "\n".join(lines).strip()


def archive_pendiente(marker: str) -> str:
    """Move one 'Decisiones pendientes' item out of hot.md into
    wiki/pendientes-archivo.md, keyed by a unique substring of its text.

    Unlike the [x]-resolved purge (which moves finished items to log.md),
    this is for open-but-stale items: content that should stop being served
    on every get_pendientes() call but is still real, searchable knowledge —
    it lands on its own indexable wiki page instead of hot.md's single
    oversized page, where index_vault's per-page embedding (max_length=8192
    tokens) would otherwise dilute or truncate it anyway.

    Refuses to guess: errors out if `marker` matches zero or more than one
    item.
    """
    hot_path = VAULT_ROOT / "wiki" / "hot.md"
    hot_text = hot_path.read_text(encoding="utf-8")
    section_match = _pendientes_section(hot_text)
    if not section_match:
        return "No se encontró la sección 'Decisiones pendientes' en hot.md."

    sec_start, sec_end = section_match.span()
    section = hot_text[sec_start:sec_end]
    items = [m.group(0) for m in ITEM_PATTERN.finditer(section)]
    matches = [item for item in items if marker in item]

    if not matches:
        return f"Ningún ítem contiene '{marker}' — no se movió nada."
    if len(matches) > 1:
        titles = ", ".join(_item_title(m) for m in matches)
        return f"'{marker}' matchea {len(matches)} ítems ({titles}) — usá un marcador más específico."

    item = matches[0]
    new_section = section.replace(item, "", 1)
    new_section = re.sub(r"\n{3,}", "\n\n", new_section).rstrip() + "\n"
    new_hot_text = hot_text[:sec_start] + new_section + hot_text[sec_end:]
    hot_path.write_text(new_hot_text, encoding="utf-8")

    if not ARCHIVE_PATH.exists():
        ARCHIVE_PATH.write_text(
            "---\n"
            "type: archivo\n"
            f"created: {date.today().isoformat()}\n"
            "estado: activo\n"
            "tags: [pendientes, archivo]\n"
            "---\n\n"
            "# Pendientes archivados\n\n"
            "Ítems abiertos (no resueltos) movidos fuera de `wiki/hot.md` → "
            "\"Decisiones pendientes\" por antigüedad/tamaño, vía `archive_pendiente()`. "
            "Siguen siendo válidos y buscables (`search_semantic`) — solo dejaron de "
            "ocupar espacio en el working set que se sirve en cada sesión/heartbeat. "
            "Ver también `wiki/log.md` para ítems ya *resueltos* (esos van ahí, no acá).\n",
            encoding="utf-8",
        )

    today_header = f"## Archivado {date.today().isoformat()}"
    archive_text = ARCHIVE_PATH.read_text(encoding="utf-8")
    entry = item.strip() + "\n"
    if today_header in archive_text:
        archive_text = archive_text.replace(today_header + "\n", today_header + "\n\n" + entry + "\n", 1)
    else:
        archive_text = archive_text.rstrip() + f"\n\n---\n\n{today_header}\n\n{entry}\n"
    ARCHIVE_PATH.write_text(archive_text, encoding="utf-8")

    return f"Movido a wiki/pendientes-archivo.md: \"{_item_title(item)}\" ({len(item):,} caracteres)."


DUPLICATE_PENDIENTE_RATIO = 0.6


def _normalize_pendiente_text(item_text: str) -> str:
    """Strip checkbox/date-added noise so similarity compares content, not
    formatting. Drops "- [ ] "/"- [x] " prefix, "(agregado dd/mm/yyyy)"
    suffixes, bold markers, and lowercases."""
    text = re.sub(r"^-\s*\[[ xX]\]\s*", "", item_text.strip())
    text = re.sub(r"\(agregado \d{1,2}/\d{1,2}/\d{4}\)", "", text)
    text = text.replace("**", "")
    return text.strip().lower()


def add_pendiente(texto: str, force: bool = False) -> str:
    """Add a new open item to hot.md's 'Decisiones pendientes' section.

    Counterpart to archive_pendiente (which only removes/relocates an item
    that already exists) -- added 29/7/2026 after a real gap found reviewing
    Hermes' Telegram sessions: the agent had no tool to create a new
    pendiente, so it kept retrying archive_pendiente against a marker that
    matched nothing (there was nothing to archive yet) and the request
    ("retirar autorización de obra social de Rocío en Constitución", sesión
    28/7/2026) never actually landed in hot.md -- had to be rescued by hand
    reading state.db directly. See wiki/hot.md, sesión 29/7/2026.

    `texto` becomes the item's full text (no "- [ ] " prefix needed -- added
    automatically). If it doesn't already mention a dd/mm/yyyy date, today's
    date gets appended in parens so it doesn't immediately show up as
    "SIN FECHA" via get_pendientes()'s age marker. Inserted right after the
    section header (most-recent-first, same place items get added by hand
    in this vault).

    Dedup check added 1/8/2026 (real incident same day: "Volver al gym"
    added twice, one already existed since 30/7 -- had to be archived by
    hand after the fact). Before writing, compares `texto` (normalized:
    checkbox/date/bold stripped, lowercased) against every currently open
    item in the section via difflib.SequenceMatcher. If any existing item
    scores >= DUPLICATE_PENDIENTE_RATIO, the write is skipped and the
    existing item's text is returned instead so the caller (Claude or
    Hermes via Telegram) can decide to update that one, or call again with
    force=True to add anyway.
    """
    texto = texto.strip()
    if not texto:
        return "El texto del pendiente no puede estar vacío."

    hot_path = VAULT_ROOT / "wiki" / "hot.md"
    hot_text = hot_path.read_text(encoding="utf-8")
    section_match = _pendientes_section(hot_text)
    if not section_match:
        return "No se encontró la sección 'Decisiones pendientes' en hot.md."

    if not force:
        section = section_match.group(0)
        _, _, body = section.partition("\n")
        existing_items = [m.group(0) for m in ITEM_PATTERN.finditer(body) if m.group(0).lower().startswith("- [ ]")]
        needle = _normalize_pendiente_text(texto)
        for existing in existing_items:
            hay = _normalize_pendiente_text(existing)
            ratio = difflib.SequenceMatcher(None, needle, hay).ratio()
            if ratio >= DUPLICATE_PENDIENTE_RATIO:
                return (
                    "No agregado -- ya existe un pendiente similar "
                    f"(similitud {ratio:.0%}): \"{existing.strip()}\". "
                    "Llamá de nuevo con force=True si de verdad es un ítem distinto."
                )

    if not DATE_PATTERN.search(texto):
        texto = f"{texto} (agregado {date.today().strftime('%d/%m/%Y')})"

    item_line = f"- [ ] {texto}\n"

    sec_start, sec_end = section_match.span()
    section = hot_text[sec_start:sec_end]
    header, sep, body = section.partition("\n")
    new_section = header + sep + item_line + body
    new_hot_text = hot_text[:sec_start] + new_section + hot_text[sec_end:]
    hot_path.write_text(new_hot_text, encoding="utf-8")

    return f'Agregado a hot.md → Decisiones pendientes: "{_item_title(item_line)}"'


def _today_note_path() -> Path:
    d = date.today()
    mes = f"{d.month:02d}-{MESES_ES[d.month]}"
    return VAULT_ROOT / "Notas" / str(d.year) / mes / f"{d.isoformat()}.md"


def add_tarea_diaria(texto: str) -> str:
    """Add a new open task to today's daily note, section '## ✅ Tareas'.

    Counterpart to add_pendiente for items with a ~1-day execution horizon
    instead of an open-ended decision -- convención adoptada 29/7/2026 (ver
    wiki/hot.md, sesión 29/7/2026): tareas accionables hoy/esta semana van a
    la nota diaria, decisiones abiertas sin fecha concreta van a
    add_pendiente. Rule of thumb, not enforced here -- the caller decides
    which of the two tools to use.

    Refuses to act if today's daily note doesn't exist yet (this tool only
    appends to an existing note, it doesn't create one from a template) or
    if the note has no '## ✅ Tareas' section in the expected shape.
    """
    texto = texto.strip()
    if not texto:
        return "El texto de la tarea no puede estar vacío."

    note_path = _today_note_path()
    if not note_path.exists():
        return f"No existe la nota de hoy todavía: {note_path}"

    note_text = note_path.read_text(encoding="utf-8")
    section_match = TAREAS_SECTION_RE.search(note_text)
    if not section_match:
        return "No se encontró la sección '## ✅ Tareas' en la nota de hoy."

    item_line = f"- [ ] {texto}"
    body = section_match.group(1)
    new_body = f"{item_line}\n{body}" if body.strip() else item_line
    new_note_text = (
        note_text[: section_match.start(1)]
        + new_body
        + note_text[section_match.end(1) :]
    )
    note_path.write_text(new_note_text, encoding="utf-8")

    return f'Agregado a la nota de hoy → ✅ Tareas: "{texto[:80]}"'


def resolve_pendiente(marker: str) -> str:
    """Mark one 'Decisiones pendientes' item as resolved and move it into
    wiki/log.md, out of hot.md.

    Counterpart to archive_pendiente: that one is for items that are still
    open but stale (deprioritized, not done); this one is for items that
    are actually finished. Added 29/7/2026 -- previously the only way to
    close a pendiente for real was Claude hand-editing hot.md/log.md each
    session. run_lint() already warns when resolved '- [x]' items pile up
    unpurged in hot.md, but nothing actually performed that purge as a
    callable tool.

    Refuses to guess: errors out if `marker` matches zero or more than one
    item (same convention as archive_pendiente). Matching is done against
    the item's original '- [ ] ...' text in hot.md, regardless of whether
    the caller already wrote it as done somewhere else.
    """
    hot_path = VAULT_ROOT / "wiki" / "hot.md"
    hot_text = hot_path.read_text(encoding="utf-8")
    section_match = _pendientes_section(hot_text)
    if not section_match:
        return "No se encontró la sección 'Decisiones pendientes' en hot.md."

    sec_start, sec_end = section_match.span()
    section = hot_text[sec_start:sec_end]
    items = [m.group(0) for m in ITEM_PATTERN.finditer(section)]
    matches = [item for item in items if marker in item]

    if not matches:
        return f"Ningún ítem contiene '{marker}' — no se movió nada."
    if len(matches) > 1:
        titles = ", ".join(_item_title(m) for m in matches)
        return f"'{marker}' matchea {len(matches)} ítems ({titles}) — usá un marcador más específico."

    item = matches[0]
    new_section = section.replace(item, "", 1)
    new_section = re.sub(r"\n{3,}", "\n\n", new_section).rstrip() + "\n"
    new_hot_text = hot_text[:sec_start] + new_section + hot_text[sec_end:]
    hot_path.write_text(new_hot_text, encoding="utf-8")

    resolved_item = re.sub(r"^- \[ \]", "- [x]", item.strip(), count=1)

    log_path = VAULT_ROOT / "wiki" / "log.md"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Wiki — Log de Actividad\n\n"
    today_header = f"## {date.today().isoformat()} (pendientes resueltos vía resolve_pendiente)"
    entry_line = resolved_item

    if today_header in log_text:
        anchor = today_header + "\n\n"
        idx = log_text.index(anchor) + len(anchor)
        log_text = log_text[:idx] + entry_line + "\n" + log_text[idx:]
    else:
        title_line = "# Wiki — Log de Actividad\n\n"
        insert_at = (
            log_text.index(title_line) + len(title_line) if title_line in log_text else 0
        )
        block = f"{today_header}\n\n{entry_line}\n\n"
        log_text = log_text[:insert_at] + block + log_text[insert_at:]

    log_path.write_text(log_text, encoding="utf-8")

    return f'Resuelto y movido a log.md: "{_item_title(item)}"'


def get_index() -> str:
    """Return the full contents of wiki/index.md as a string."""
    return (VAULT_ROOT / "wiki" / "index.md").read_text(encoding="utf-8")


WIKI_SUBFOLDERS = ["fuentes", "entidades", "conceptos", "sintesis"]


def get_page(name: str) -> str:
    """Resolve a [[wikilink]] name to its file and return its contents.

    Tries each of the four wiki subfolders in turn (fuentes, entidades,
    conceptos, sintesis) since a wikilink doesn't say which one it lives in.
    """
    for subfolder in WIKI_SUBFOLDERS:
        candidate = VAULT_ROOT / "wiki" / subfolder / f"{name}.md"
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    raise FileNotFoundError(f"No page named '{name}' in any wiki subfolder")


# type -> subfolder, matches the CLAUDE.md page types (fuente/entidad/concepto/sintesis)
TYPE_TO_FOLDER = {
    "fuente": "fuentes",
    "entidad": "entidades",
    "concepto": "conceptos",
    "sintesis": "sintesis",
}

# Frontmatter fields each page type requires, per the templates in CLAUDE.md.
REQUIRED_FRONTMATTER_FIELDS = {
    "fuente": {"type", "created", "fuente_original", "estado", "superseded_by", "fragmento_extraido", "tags", "atribucion"},
    "entidad": {"type", "created", "categoria", "estado", "tags"},
    "concepto": {"type", "created", "estado", "tags"},
    "sintesis": {"type", "created", "query", "estado", "tags"},
}

# Patrón de comportamiento encontrado 8/7/2026 (análisis de log): contenido
# ajeno se cita repetidamente como propio de Ivan sin marcarlo, y el error
# rebota entre sesiones con veredictos opuestos (caso real: "+25% OOS" de
# btc_ema_pullback, corregido el 4/7, revertido el 5/7). Campo obligatorio
# en vez de confiar en que se note a ojo.
ATRIBUCION_VALORES_VALIDOS = {"propio", "citado-de-tercero", "sin-verificar"}


def create_page(page_type: str, name: str, content: str, metadata: dict) -> str:
    """Create a new wiki page, enforcing the frontmatter contract from CLAUDE.md.

    `metadata` must already carry the required fields for `page_type` (the
    caller is expected to have written a proper source/entidad/concepto/
    sintesis body per the CLAUDE.md templates — this function only validates
    the frontmatter and writes the file, it doesn't invent template content).
    Refuses to overwrite an existing page.
    """
    if page_type not in TYPE_TO_FOLDER:
        raise ValueError(f"Unknown page_type '{page_type}', expected one of {list(TYPE_TO_FOLDER)}")

    metadata = dict(metadata)
    metadata.setdefault("type", page_type)
    metadata.setdefault("created", date.today().isoformat())

    missing = REQUIRED_FRONTMATTER_FIELDS[page_type] - metadata.keys()
    if missing:
        raise ValueError(f"metadata missing required fields for '{page_type}': {sorted(missing)}")

    if page_type == "fuente" and metadata["atribucion"] not in ATRIBUCION_VALORES_VALIDOS:
        raise ValueError(
            f"atribucion debe ser uno de {sorted(ATRIBUCION_VALORES_VALIDOS)}, "
            f"recibido: {metadata['atribucion']!r}"
        )

    path = VAULT_ROOT / "wiki" / TYPE_TO_FOLDER[page_type] / f"{name}.md"
    if path.exists():
        raise FileExistsError(f"Page '{name}' already exists at {path}")

    post = frontmatter.Post(content, **metadata)
    path.write_text(frontmatter.dumps(post), encoding="utf-8")
    return str(path)


# Valid PARA roots per the "Regla de reubicación del archivo original" in
# CLAUDE.md — the only destinations step 9 of INGERIR is allowed to move to.
PARA_ROOTS = {"Notas", "Areas", "Proyectos", "Archivo", "Recursos"}


def _resolve_inbox_path(inbox_path: str) -> Path:
    """Resolve a filename relative to _Inbox/ to an absolute Path, refusing
    traversal outside _Inbox/ and refusing a path that isn't an existing file.

    Shared by move_source_file, check_ingested and register_ingested — every
    tool that takes an "_Inbox-relative path" argument goes through this same
    validation instead of three copies of the same two checks.
    """
    source = (VAULT_ROOT / "_Inbox" / inbox_path).resolve()
    if VAULT_ROOT / "_Inbox" not in source.parents:
        raise ValueError(f"'{inbox_path}' escapes _Inbox/")
    if not source.is_file():
        raise FileNotFoundError(f"No file '{inbox_path}' in _Inbox/")
    return source


# Manifest de archivos de _Inbox/ ya evaluados por INGERIR (cualquier
# veredicto), indexados por hash MD5 de contenido — no por nombre de
# archivo, que no es estable entre re-clips del mismo tweet/URL. Vive junto
# al índice semántico (chroma_index/, ver rag.py) como estado de servidor,
# nunca dentro del vault Diario. Ver docs/superpowers/specs/
# 2026-07-15-manifest-archivos-ingeridos-design.md en el vault Diario.
MANIFEST_PATH = Path(__file__).parent / "ingested_manifest.json"

VEREDICTO_VALORES_VALIDOS = {"pasa", "descartado", "fragmento_extraido", "link_con_anotacion"}


def move_source_file(inbox_path: str, target_folder: str) -> str:
    """Move a file out of _Inbox/ into its PARA destination (step 9 of INGERIR).

    `inbox_path` is a filename relative to _Inbox/ (no traversal outside it).
    `target_folder` is a folder relative to VAULT_ROOT, rooted at one of the
    PARA_ROOTS (Notas/Areas/Proyectos/Archivo/Recursos) per the CLAUDE.md
    decision tree; created if it doesn't exist yet (e.g. a fresh month folder
    under Notas/YYYY/MM-Mes/). Refuses to overwrite an existing destination
    file — the caller (Claude, mid-INGERIR) is expected to have already
    resolved the destination per that decision tree; this just executes the
    move safely instead of a raw Bash `mv`.

    Refuses to move a file that has not been registered in the ingested
    manifest yet (register_ingested must run first). register_ingested hashes
    the file from its _Inbox/ location -- once moved, it is gone from there
    and the registration can no longer happen normally. This bit Claude twice
    (2026-07-15, 2026-07-26) despite CLAUDE.md documenting the required order
    in prose; a hard gate here makes the mistake impossible instead of just
    documented.
    """
    source = _resolve_inbox_path(inbox_path)

    manifest = _load_manifest()
    if _hash_file(source) not in manifest:
        raise ValueError(
            f"'{inbox_path}' no esta registrado en el manifest todavia -- "
            'llama a register_ingested(path, veredicto, motivo, pagina) antes de moverlo.'
        )

    dest_folder = (VAULT_ROOT / target_folder).resolve()
    if VAULT_ROOT not in dest_folder.parents and dest_folder != VAULT_ROOT:
        raise ValueError(f"'{target_folder}' escapes the vault")
    if dest_folder.relative_to(VAULT_ROOT).parts[0] not in PARA_ROOTS:
        raise ValueError(
            f"'{target_folder}' isn't under a PARA root {sorted(PARA_ROOTS)}"
        )

    dest = dest_folder / source.name
    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    dest_folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(dest))
    return str(dest)


# Full Calendar Remastered's own weekday code, indexed by Python's
# date.weekday() (Mon=0 ... Sun=6) — reverse-engineered from the plugin's
# real source (types/schema.ts: z.enum(['U','M','T','W','R','F','S'])).
_FCR_WEEKDAY_CODES = ["M", "T", "W", "R", "F", "S", "U"]


def _matches_days_of_week(d: date, days_of_week: list) -> bool:
    return _FCR_WEEKDAY_CODES[d.weekday()] in days_of_week


def _matches_repeat_on(d: date, week: int, weekday: int) -> bool:
    """`weekday` in FCR's repeatOn is 0=Sun..6=Sat; `week` is 1-4 for the
    Nth occurrence in the month, or -1 for the last occurrence."""
    target_py_weekday = (weekday - 1) % 7  # convert Sun=0-based to Mon=0-based
    if d.weekday() != target_py_weekday:
        return False
    if week == -1:
        return (d + timedelta(days=7)).month != d.month
    return (d.day - 1) // 7 + 1 == week


def _normalize_time(value) -> str:
    """YAML sin comillas tipo `13:29` se lee sexagesimal (PyYAML resuelve
    "H:MM" a un int de minutos, no a un string) — algunos eventos de
    Calendario/ tienen el horario entre comillas y otros no, así que el
    mismo campo puede llegar como str u int según el archivo. Normaliza
    ambos a "HH:MM" para que se puedan comparar/ordenar de forma consistente."""
    if isinstance(value, int):
        return f"{value // 60:02d}:{value % 60:02d}"
    return str(value) if value else "00:00"


def _resolve_local_events(start: date, dias: int) -> list[tuple[date, str, str, str]]:
    """Devuelve (día, hora_inicio, hora_fin, título) para cada evento de
    Calendario/ que ocurre en el rango [start, start+dias). Lógica de
    resolución de recurrencia compartida por get_local_events() y
    check_calendar_overlaps()."""
    calendario_dir = VAULT_ROOT / "Calendario"
    if not calendario_dir.exists():
        return []

    events = []
    for md_file in sorted(calendario_dir.glob("*.md")):
        if md_file.name == "README.md":
            continue
        try:
            post = frontmatter.load(md_file)
        except Exception:
            continue
        meta = post.metadata
        title = meta.get("title", md_file.stem)
        start_time = _normalize_time(meta.get("startTime"))
        end_time = _normalize_time(meta.get("endTime"))
        event_type = meta.get("type", "single")

        for offset in range(dias):
            day = start + timedelta(days=offset)
            occurs = False
            if event_type == "recurring":
                days_of_week = meta.get("daysOfWeek")
                repeat_on = meta.get("repeatOn")
                if days_of_week:
                    occurs = _matches_days_of_week(day, days_of_week)
                elif repeat_on:
                    occurs = _matches_repeat_on(day, repeat_on["week"], repeat_on["weekday"])
            else:
                occurs = meta.get("date") == day.isoformat()

            if occurs:
                events.append((day, start_time, end_time, title))

    return events


def get_local_events(target_date: str | None = None, dias: int = 1) -> str:
    """Eventos del calendario local (`Calendario/`, gestionado por el plugin
    Full Calendar Remastered de Obsidian) para una fecha o rango de días.

    `target_date` en formato YYYY-MM-DD (default: hoy). `dias` es cuántos
    días consecutivos reportar a partir de esa fecha. Soporta eventos
    únicos (`type: single`/sin `type`, campo `date`) y recurrentes
    (`type: recurring`, por `daysOfWeek` o por `repeatOn` — Nth día de la
    semana del mes). No usa Google Calendar ni ninguna fuente externa —
    100% local, pensado para el heartbeat autónomo de Hermes.
    """
    start = date.fromisoformat(target_date) if target_date else date.today()
    events = _resolve_local_events(start, dias)

    if not events:
        return f"Sin eventos locales entre {start.isoformat()} y {(start + timedelta(days=dias - 1)).isoformat()}."

    events.sort(key=lambda e: (e[0], e[1]))
    lines = [f"# Eventos locales ({start.isoformat()}, {dias} día(s))", ""]
    current_day = None
    for day, start_time, end_time, title in events:
        if day != current_day:
            lines.append(f"## {day.isoformat()} ({day.strftime('%A')})")
            current_day = day
        lines.append(f"- {start_time}: {title}")
    return "\n".join(lines)


def check_calendar_overlaps(target_date: str | None = None, dias: int = 14) -> str:
    """Patrón 2 del análisis de comportamiento (8/7/2026): en vez de arreglar
    un solapamiento del calendario cuando aparece y descubrir el siguiente
    en la próxima sesión (~14 fixes consecutivos documentados el 5/7/2026),
    chequea TODO el rango de una sola pasada. Reusa la misma resolución de
    recurrencia que get_local_events(); no modifica Calendario/, solo reporta.
    """
    start = date.fromisoformat(target_date) if target_date else date.today()
    events = _resolve_local_events(start, dias)
    if not events:
        return f"Sin eventos locales entre {start.isoformat()} y {(start + timedelta(days=dias - 1)).isoformat()}."

    by_day: dict[date, list[tuple[str, str, str]]] = {}
    for day, start_time, end_time, title in events:
        by_day.setdefault(day, []).append((start_time, end_time, title))

    overlaps: list[str] = []
    for day in sorted(by_day):
        same_day = sorted(by_day[day])
        # Todos los pares, no solo consecutivos tras ordenar — un evento
        # "contenedor" largo (ej. 10:00-16:00) puede pisar a un sub-evento
        # que no es su vecino inmediato en el orden por hora de inicio.
        for i in range(len(same_day)):
            for j in range(i + 1, len(same_day)):
                s1, e1, t1 = same_day[i]
                s2, e2, t2 = same_day[j]
                if s2 < e1 and s1 < e2:
                    overlaps.append(
                        f"{day.isoformat()} ({day.strftime('%A')}): "
                        f"\"{t1}\" ({s1}-{e1}) se pisa con \"{t2}\" ({s2}-{e2})"
                    )

    if not overlaps:
        return f"Sin solapamientos entre {start.isoformat()} y {(start + timedelta(days=dias - 1)).isoformat()} ({len(events)} evento(s) chequeado(s))."

    lines = [f"# Solapamientos de calendario ({len(overlaps)})", ""]
    lines.extend(f"- {o}" for o in overlaps)
    return "\n".join(lines)


def _iter_wiki_pages():
    """Yield (path, frontmatter.Post) for every page with parseable frontmatter.

    Pages whose YAML frontmatter fails to parse (e.g. an unquoted `[[wikilink]]`
    in a value, which YAML reads as a flow-sequence `[`) are skipped here and
    reported separately by `_frontmatter_errors()` — one bad page shouldn't
    crash every tool that walks the vault.
    """
    for md_file, _ in _iter_wiki_files():
        try:
            yield md_file, frontmatter.load(md_file)
        except Exception:
            continue


def _iter_wiki_files():
    """Yield (path, subfolder) for every .md file in the four content subfolders."""
    for subfolder in WIKI_SUBFOLDERS:
        folder = VAULT_ROOT / "wiki" / subfolder
        if not folder.exists():
            continue
        for md_file in sorted(folder.glob("*.md")):
            yield md_file, subfolder


def _frontmatter_errors() -> list[tuple[str, str]]:
    """Return (page_name, error_message) for every page with unparseable frontmatter."""
    errors = []
    for md_file, _ in _iter_wiki_files():
        try:
            frontmatter.load(md_file)
        except Exception as e:
            errors.append((md_file.stem, str(e).splitlines()[0]))
    return errors


def build_index(dry_run: bool = True) -> str:
    """Regenerate wiki/index.md from the frontmatter of every wiki page.

    Returns the generated markdown. This auto-generated index is simpler
    than a hand-curated one (no narrative "Tema" column — that requires
    an LLM/human summary, not something derivable from frontmatter alone),
    so by default this only returns the content (`dry_run=True`) instead of
    overwriting the existing wiki/index.md. Pass `dry_run=False` to write it.
    """
    section_labels = {
        "fuente": "Fuentes",
        "entidad": "Entidades",
        "concepto": "Conceptos",
        "sintesis": "Síntesis",
    }
    sections: dict[str, list[tuple[str, str, str]]] = {t: [] for t in section_labels}

    for md_file, post in _iter_wiki_pages():
        page_type = post.get("type", "?")
        if page_type not in sections:
            continue
        sections[page_type].append(
            (md_file.stem, post.get("estado", "?"), post.get("created", "?"))
        )

    lines = ["# Wiki — Índice Maestro (auto-generado)", "", f"Última actualización: {date.today().isoformat()}", ""]
    for page_type, label in section_labels.items():
        lines.append(f"## {label}")
        lines.append("")
        lines.append("| Página | Estado | Creado |")
        lines.append("|--------|--------|--------|")
        for name, estado, created in sections[page_type]:
            lines.append(f"| [[{name}]] | {estado} | {created} |")
        lines.append("")

    content = "\n".join(lines)
    if not dry_run:
        (VAULT_ROOT / "wiki" / "index.md").write_text(content, encoding="utf-8")
    return content


WIKILINK_PATTERN = re.compile(r"\[\[([^\]|#]+)")
# Inline code spans (`` `...` ``) and fenced code blocks (```...```) — a
# literal `[[wikilink]]` used as a syntax example inside backticks (common
# in pages that document the vault's own conventions, e.g.
# notas-atomicas-con-razon-de-conexion) is not a real link and must not be
# scanned as one. Found as a real false-positive bug (14/7/2026): run_lint
# flagged 8 "broken links" that were all syntax examples in backticks.
_CODE_SPAN_PATTERN = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)


def _strip_code_spans(text: str) -> str:
    """Blank out inline code / fenced code blocks before a wikilink scan,
    preserving line/character offsets (replaces with spaces, not deletes) so
    any caller doing line-number reporting on the original text still lines
    up."""
    return _CODE_SPAN_PATTERN.sub(lambda m: " " * len(m.group(0)), text)

# get_pendientes() worked fine at 38,675 chars and started risking truncation
# again at 94,485 chars (see wiki/hot.md, sesión 3/7/2026) against Hermes'
# ~8195-token input budget. Warn well before that point so the purge (move
# resolved [x] items to log.md) happens proactively instead of after a
# heartbeat silently loses content.
PENDIENTES_WARN_CHARS = 40_000
RESOLVED_ITEM_PATTERN = re.compile(r"^\s*- \[x\]", re.MULTILINE | re.IGNORECASE)


def run_lint() -> str:
    """Run the LINT workflow checks from CLAUDE.md and return a text report.

    Covers: broken links (CRÍTICO), orphan pages (ADVERTENCIA), active pages
    that appear in another page's superseded_by without their own estado
    reflecting it (CRÍTICO), and 'Decisiones pendientes' size creeping back
    toward truncation range (ADVERTENCIA). Cross-reference/staleness
    SUGERENCIA-level checks are not implemented — those need judgment calls
    a script can't make reliably.
    """
    pages = list(_iter_wiki_pages())
    # Use every .md file as a valid link target, even ones whose own frontmatter
    # is broken (Section above already flags those) — a page with bad frontmatter
    # is still a page other pages can validly link to.
    known_names = {md_file.stem for md_file, _ in _iter_wiki_files()}

    broken_links: list[tuple[str, str]] = []
    linked_to: set[str] = set()
    for md_file, post in pages:
        for match in WIKILINK_PATTERN.finditer(_strip_code_spans(post.content)):
            target = match.group(1).strip()
            linked_to.add(target)
            if target not in known_names:
                broken_links.append((md_file.stem, target))

    orphans = sorted(known_names - linked_to)

    # Per CLAUDE.md: superseded_by lives on the OLD page and points at the page
    # that replaced it. A page that has superseded_by populated but still says
    # estado: activo forgot to flip its own estado — that's the actual bug,
    # not a cross-reference to whatever it points at (which is expected to be
    # active — it's the current replacement).
    unclosed_contradictions: list[str] = []
    for md_file, post in pages:
        if (post.get("superseded_by") or []) and post.get("estado") == "activo":
            unclosed_contradictions.append(md_file.stem)

    fm_errors = _frontmatter_errors()

    # Patrón 8 del análisis de comportamiento (8/7/2026): fuentes sin marcar
    # si el contenido es propio o citado de un tercero, causa real de
    # atribuciones erróneas que rebotaron entre sesiones (btc_ema_pullback
    # +25% OOS). Campo nuevo, no retroactivo — solo aviso, no bloquea.
    fuentes_sin_atribucion = sorted(
        md_file.stem for md_file, post in pages
        if post.get("type") == "fuente" and not post.get("atribucion")
    )

    # Measure the real, full section — not get_pendientes()'s budgeted
    # output, which by design omits older items once the section grows past
    # PENDIENTES_SERVE_BUDGET_CHARS. Using the served (already-trimmed)
    # text here would hide the exact growth this check exists to catch.
    hot_text = (VAULT_ROOT / "wiki" / "hot.md").read_text(encoding="utf-8")
    section_match = _pendientes_section(hot_text)
    pendientes_text = section_match.group(0).strip() if section_match else ""
    pendientes_size = len(pendientes_text)
    resolved_count = len(RESOLVED_ITEM_PATTERN.findall(pendientes_text))

    lines = ["# Reporte de LINT", ""]

    lines.append(f"## CRÍTICO — frontmatter YAML inválido ({len(fm_errors)})")
    for name, error in fm_errors:
        lines.append(f"- `{name}`: {error}")
    if not fm_errors:
        lines.append("- Ninguno")
    lines.append("")

    lines.append(f"## CRÍTICO — links rotos ({len(broken_links)})")
    for source, target in broken_links:
        lines.append(f"- `{source}` enlaza a `[[{target}]]`, que no existe")
    if not broken_links:
        lines.append("- Ninguno")
    lines.append("")

    lines.append(f"## CRÍTICO — contradicciones sin cerrar ({len(unclosed_contradictions)})")
    for name in unclosed_contradictions:
        lines.append(f"- `{name}` tiene `superseded_by` poblado pero su `estado` sigue en `activo`")
    if not unclosed_contradictions:
        lines.append("- Ninguna")
    lines.append("")

    lines.append(f"## ADVERTENCIA — páginas huérfanas ({len(orphans)})")
    for name in orphans:
        lines.append(f"- `[[{name}]]` no tiene ningún link entrante")
    if not orphans:
        lines.append("- Ninguna")
    lines.append("")

    lines.append("## ADVERTENCIA — tamaño de 'Decisiones pendientes'")
    if pendientes_size > PENDIENTES_WARN_CHARS:
        lines.append(
            f"- La sección tiene {pendientes_size:,} caracteres "
            f"(umbral de aviso: {PENDIENTES_WARN_CHARS:,}) — riesgo de truncamiento "
            f"para consumidores con contexto limitado (ej. heartbeat de Hermes)."
        )
        lines.append(
            f"- {resolved_count} ítem(s) ya resuelto(s) (`- [x]`) siguen ahí — "
            "moverlos a `log.md` es la purga más simple para bajar el tamaño."
        )
        lines.append(
            "- Si ya no quedan resueltos por purgar, el siguiente paso es "
            "`archive_pendiente(marker)` sobre los ítems abiertos más viejos/menos "
            "urgentes, que los saca de `hot.md` hacia `wiki/pendientes-archivo.md` "
            "(siguen buscables por `search_semantic`, dejan de pesar en cada sesión)."
        )
    else:
        lines.append(
            f"- {pendientes_size:,} caracteres, dentro del umbral de aviso "
            f"({PENDIENTES_WARN_CHARS:,}). {resolved_count} ítem(s) resuelto(s) sin purgar."
        )
    lines.append("")

    lines.append(f"## SUGERENCIA — fuentes sin campo `atribucion` ({len(fuentes_sin_atribucion)})")
    if fuentes_sin_atribucion:
        lines.append(
            "- Campo nuevo (8/7/2026), no retroactivo — estas fuentes son de antes. "
            "Completar con `propio` / `citado-de-tercero` / `sin-verificar` cuando se editen."
        )
        for name in fuentes_sin_atribucion[:30]:
            lines.append(f"- `{name}`")
        if len(fuentes_sin_atribucion) > 30:
            lines.append(f"- ...y {len(fuentes_sin_atribucion) - 30} más")
    else:
        lines.append("- Ninguna")
    lines.append("")

    ultima_sesion_match = _ultima_sesion_section(hot_text)
    ultima_sesion_size = len(ultima_sesion_match.group(0)) if ultima_sesion_match else 0
    lines.append("## ADVERTENCIA — tamaño de 'Última sesión'")
    if ultima_sesion_size > ULTIMA_SESION_WARN_CHARS:
        lines.append(
            f"- La sección tiene {ultima_sesion_size:,} caracteres "
            f"(umbral de aviso: {ULTIMA_SESION_WARN_CHARS:,}) — correr `prune_hot()` para "
            f"compactar las entradas de hace más de {ULTIMA_SESION_RETENTION_DAYS} días con "
            "puntero a `log.md`."
        )
    else:
        lines.append(
            f"- {ultima_sesion_size:,} caracteres, dentro del umbral de aviso "
            f"({ULTIMA_SESION_WARN_CHARS:,})."
        )
    lines.append("")

    brechas_match = _brechas_section(hot_text)
    brechas_size = len(brechas_match.group(0)) if brechas_match else 0
    lines.append("## ADVERTENCIA — tamaño de 'Brechas documentadas'")
    if brechas_size > BRECHAS_WARN_CHARS:
        lines.append(
            f"- La sección tiene {brechas_size:,} caracteres "
            f"(umbral de aviso: {BRECHAS_WARN_CHARS:,}) — correr `prune_brechas()` para "
            "compactar las entradas cerradas con respaldo verificado en `log.md`."
        )
    else:
        lines.append(
            f"- {brechas_size:,} caracteres, dentro del umbral de aviso "
            f"({BRECHAS_WARN_CHARS:,})."
        )
    lines.append("")

    return "\n".join(lines)


FUZZY_MATCH_MIN_SIMILARITY = 0.72


def fix_broken_links(dry_run: bool = True) -> str:
    """For every broken [[wikilink]] run_lint's check would flag, propose (or
    apply) a fix by fuzzy-matching the broken target against every real page
    name — stdlib difflib, no new dependency.

    Adapted from `olw maintain --fix` (kytmanov/obsidian-llm-wiki-local, see
    wiki/fuentes/second-brain-wiki-local-ollama-2026.md), simplified: that
    tool builds an alias map at ingest time (e.g. "PC" -> "Program Counter")
    so it can repair links whose text doesn't resemble the real title at
    all. This vault has no such map, so this only catches broken links whose
    text is *similar* to a real page name (typos, renames, `-` vs `_`) — a
    true alias with no string overlap won't match and lands in "sin match
    razonable" for a manual fix.

    dry_run=True (default): report proposed fixes, touch nothing.
    dry_run=False: rewrite each fixable [[old]] to [[new]] in place. Uses the
    same wikilink scan as get_backlinks/run_lint, and replaces only the
    matched name span — never surrounding text, alias (`|`), or heading
    (`#`) suffixes.
    """
    known_names = {md_file.stem for md_file, _ in _iter_wiki_files()}

    # (start, end) are offsets into the STRIPPED content, which line up 1:1
    # with the original (see _strip_code_spans docstring — spaces, not
    # deletes) — safe to apply directly to post.content below.
    fixes_by_file: dict[Path, list[tuple[int, int, str, str]]] = {}
    unfixable: list[tuple[str, str]] = []

    for md_file, post in _iter_wiki_pages():
        stripped = _strip_code_spans(post.content)
        file_fixes = []
        for match in WIKILINK_PATTERN.finditer(stripped):
            target = match.group(1).strip()
            if target in known_names:
                continue
            candidates = difflib.get_close_matches(
                target, known_names, n=1, cutoff=FUZZY_MATCH_MIN_SIMILARITY
            )
            if candidates:
                file_fixes.append((match.start(1), match.end(1), target, candidates[0]))
            else:
                unfixable.append((md_file.stem, target))
        if file_fixes:
            fixes_by_file[md_file] = file_fixes

    total_fixes = sum(len(v) for v in fixes_by_file.values())
    lines = ["# Reparación de links rotos", ""]

    lines.append(f"## Arreglables por fuzzy match ({total_fixes})")
    for md_file, file_fixes in fixes_by_file.items():
        for _, _, old, new in file_fixes:
            lines.append(f"- `{md_file.stem}`: [[{old}]] → [[{new}]]")
    if not fixes_by_file:
        lines.append("- Ninguno")
    lines.append("")

    lines.append(f"## Sin match razonable ({len(unfixable)}) — requieren arreglo manual")
    for source, target in unfixable:
        lines.append(
            f"- `{source}` enlaza a `[[{target}]]`, sin candidato similar "
            "(¿alias real sin overlap de texto, o página que nunca existió?)"
        )
    if not unfixable:
        lines.append("- Ninguno")
    lines.append("")

    if dry_run:
        lines.append("(dry_run=True — nada escrito. Llamar con dry_run=False para aplicar los arreglables.)")
        return "\n".join(lines)

    applied = 0
    for md_file, file_fixes in fixes_by_file.items():
        fresh_post = frontmatter.load(md_file)
        content = fresh_post.content
        # Apply from the end backwards so earlier offsets in this file stay valid.
        for start, end, _old, new in sorted(file_fixes, key=lambda f: f[0], reverse=True):
            content = content[:start] + new + content[end:]
        fresh_post.content = content
        md_file.write_text(frontmatter.dumps(fresh_post), encoding="utf-8")
        applied += len(file_fixes)

    lines.append(f"✅ {applied} link(s) reparado(s) en {len(fixes_by_file)} página(s).")
    return "\n".join(lines)


def doctor() -> str:
    """Health-check for the vault + vault-mcp infra itself — not vault
    CONTENT quality (run_lint's job), but whether the plumbing underneath
    every other tool is intact: the 4 wiki/ subfolders, the core files
    (hot.md/index.md/log.md), the ingest manifest, frontmatter validity
    across every page, and whether the semantic index (rag.py) is populated
    and roughly in sync with the real page count.

    Adapted from `olw doctor` (kytmanov/obsidian-llm-wiki-local, see
    wiki/fuentes/second-brain-wiki-local-ollama-2026.md) — motivated by 2
    real incidents where vault-mcp broke silently and nobody noticed until a
    session tried to use it (uv sync losing the editable install, 15/7/2026;
    a cron job desynced on a stale model pin, caught by chance on 12/7/2026).
    """
    lines = ["# vault-mcp — doctor", ""]
    all_ok = True

    def check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        if not passed:
            all_ok = False
        symbol = "✅" if passed else "❌"
        lines.append(f"{symbol} {label}" + (f" — {detail}" if detail else ""))

    check("VAULT_ROOT existe", VAULT_ROOT.exists(), str(VAULT_ROOT))

    for subfolder in WIKI_SUBFOLDERS:
        folder = VAULT_ROOT / "wiki" / subfolder
        check(f"wiki/{subfolder}/ existe", folder.exists())

    for core_file in ("hot.md", "index.md", "log.md"):
        path = VAULT_ROOT / "wiki" / core_file
        exists = path.exists()
        non_empty = exists and path.stat().st_size > 0
        detail = "" if non_empty else ("no existe" if not exists else "existe pero está vacío")
        check(f"wiki/{core_file} existe y no está vacío", non_empty, detail)

    try:
        manifest = _load_manifest()
        check("ingested_manifest.json es JSON válido", True, f"{len(manifest)} entrada(s)")
    except Exception as e:
        check("ingested_manifest.json es JSON válido", False, str(e).splitlines()[0])

    fm_errors = _frontmatter_errors()
    check(
        "Frontmatter YAML válido en todas las páginas",
        not fm_errors,
        f"{len(fm_errors)} página(s) rota(s) — ver run_lint() para el detalle" if fm_errors else "",
    )

    page_count = sum(1 for _ in _iter_wiki_files())
    try:
        from vault_mcp import rag
        indexed_count = rag.get_collection().count()
        drift = abs(page_count - indexed_count)
        # Loose threshold on purpose: index_vault() embeds per-chunk, not
        # per-page (a page can produce 0, 1, or several chunks), so counts
        # rarely match 1:1 even when perfectly healthy. Only flag a gap big
        # enough to suggest the index is stale or never ran.
        check(
            "Índice semántico (search_semantic) poblado y no muy desalineado",
            indexed_count > 0 and drift < max(20, page_count * 0.5),
            f"{indexed_count} chunk(s) indexados vs. {page_count} páginas reales",
        )
    except Exception as e:
        check("Índice semántico (search_semantic) accesible", False, str(e).splitlines()[0])

    lines.append("")
    lines.append(
        "✅ Todo OK — vault-mcp e infra del vault en buen estado."
        if all_ok
        else "❌ Hay al menos un chequeo fallido arriba — revisar antes de confiar en tools que dependen de eso."
    )
    return "\n".join(lines)


def get_backlinks(name: str) -> str:
    """Return every page that links to [[name]], the reverse of get_page's
    forward resolution. Reuses the same wikilink scan as run_lint's orphan
    check, so a page counts as linked here iff it would count as non-orphan
    there — the two stay consistent by construction instead of by two
    separately-maintained regexes."""
    known_names = {md_file.stem for md_file, _ in _iter_wiki_files()}
    exists = name in known_names

    backlinks: set[str] = set()
    for md_file, post in _iter_wiki_pages():
        if md_file.stem == name:
            continue
        for match in WIKILINK_PATTERN.finditer(_strip_code_spans(post.content)):
            if match.group(1).strip() == name:
                backlinks.add(md_file.stem)
                break

    lines = [f"# Backlinks — [[{name}]]", ""]
    if not exists:
        lines.append(
            f"⚠️ No existe ninguna página `{name}.md` en wiki/ — puede ser un "
            "link roto o un nombre distinto al del archivo real."
        )
        lines.append("")
    lines.append(f"{len(backlinks)} página(s) enlazan a [[{name}]]:")
    for b in sorted(backlinks):
        lines.append(f"- [[{b}]]")
    if not backlinks:
        lines.append("- Ninguna (página huérfana, o todavía no citada por otra página)")

    return "\n".join(lines)


# Weighted relevance model for get_related_pages(), adapted from nashsu/llm_wiki's
# 4-signal design (found via INVESTIGAR, 2026-07-17, see
# wiki/fuentes/second-brain-wiki-local-ollama-2026.md): direct link, co-citation
# by a shared fuente, Adamic-Adar over common neighbors, and same-type bonus.
# Weights mirror the original ordering (co-citation > direct link > Adamic-Adar
# > type) — a fuente explicitly tying two pages together in the same paragraph
# is a stronger signal than an incidental link, which in turn beats a shared
# neighbor two hops away.
RELATED_WEIGHT_DIRECT_LINK = 3.0
RELATED_WEIGHT_CO_CITATION = 4.0
RELATED_WEIGHT_ADAMIC_ADAR = 1.5
RELATED_WEIGHT_TYPE_AFFINITY = 1.0


def _build_link_graph() -> tuple[dict[str, set[str]], dict[str, str], dict[str, set[str]]]:
    """Scan every wiki page once and return:
    - outlinks: page -> set of pages it wikilinks to (excludes self, dedups)
    - page_type: page -> its frontmatter `type` (fuente/entidad/concepto/sintesis)
    - fuente_links: fuente page -> set of pages it wikilinks to (subset of
      outlinks, only for type=fuente) — used for the co-citation signal, since
      a fuente linking two pages in the same paragraph is what "source overlap"
      means here (this vault has no `sources[]` frontmatter array like the
      tool that inspired this)."""
    outlinks: dict[str, set[str]] = {}
    page_type: dict[str, str] = {}
    fuente_links: dict[str, set[str]] = {}

    for md_file, post in _iter_wiki_pages():
        name = md_file.stem
        page_type[name] = post.get("type", "?")
        links = {
            m.group(1).strip()
            for m in WIKILINK_PATTERN.finditer(_strip_code_spans(post.content))
            if m.group(1).strip() != name
        }
        outlinks[name] = links
        if page_type[name] == "fuente":
            fuente_links[name] = links

    return outlinks, page_type, fuente_links


def get_related_pages(name: str, top_n: int = 8) -> str:
    """Rank every other wiki page by weighted relevance to [[name]] and return
    the top N, with a one-line breakdown of which signals fired.

    Signals (weights in RELATED_WEIGHT_*, see comment above the constants):
    - Direct link: name links to the other page, or vice versa.
    - Co-citation: some fuente page links to both name and the other page
      in its own body (the vault's equivalent of "shared source").
    - Adamic-Adar: pages sharing neighbors, weighted by 1/log(neighbor's
      degree) so a rare, specific shared neighbor counts more than a hub
      page like hot.md or a densely-linked entity.
    - Type affinity: bonus if both pages share the same frontmatter type.

    Unlike get_backlinks (exact reverse-link lookup), this is a ranked
    approximation meant for surfacing candidates for a cluster/síntesis —
    always eyeball the result, it's a heuristic, not a source of truth.
    """
    outlinks, page_type, fuente_links = _build_link_graph()
    known_names = set(outlinks)

    if name not in known_names:
        return (
            f"⚠️ No existe ninguna página `{name}.md` en wiki/ — no se puede "
            "calcular relevancia para un nombre que no resuelve a una página real."
        )

    # Undirected adjacency: A-B counts as linked if either links to the other.
    neighbors: dict[str, set[str]] = {p: set(links) for p, links in outlinks.items()}
    for p, links in outlinks.items():
        for other in links:
            neighbors.setdefault(other, set()).add(p)

    degree = {p: len(ns) for p, ns in neighbors.items()}

    direct = neighbors.get(name, set())

    co_citation: Counter[str] = Counter()
    for fuente, links in fuente_links.items():
        if name not in links:
            continue
        for other in links:
            if other != name:
                co_citation[other] += 1

    name_neighbors = neighbors.get(name, set())
    adamic_adar: Counter[float] = Counter()
    for candidate in known_names:
        if candidate == name:
            continue
        shared = name_neighbors & neighbors.get(candidate, set())
        score = sum(
            1.0 / math.log(degree[n]) for n in shared if degree.get(n, 0) > 1
        )
        if score > 0:
            adamic_adar[candidate] = score

    name_type = page_type.get(name, "?")

    scored: list[tuple[float, str, list[str]]] = []
    for candidate in known_names:
        if candidate == name:
            continue
        breakdown = []
        score = 0.0
        if candidate in direct:
            score += RELATED_WEIGHT_DIRECT_LINK
            breakdown.append("link directo")
        if co_citation[candidate]:
            score += RELATED_WEIGHT_CO_CITATION * co_citation[candidate]
            breakdown.append(f"co-citado por {co_citation[candidate]} fuente(s)")
        if adamic_adar[candidate]:
            score += RELATED_WEIGHT_ADAMIC_ADAR * adamic_adar[candidate]
            breakdown.append(f"vecinos en común (adamic-adar {adamic_adar[candidate]:.2f})")
        if page_type.get(candidate) == name_type and name_type != "?":
            score += RELATED_WEIGHT_TYPE_AFFINITY
            breakdown.append(f"mismo tipo ({name_type})")
        if score > 0:
            scored.append((score, candidate, breakdown))

    scored.sort(key=lambda t: (-t[0], t[1]))

    lines = [f"# Páginas relacionadas — [[{name}]]", ""]
    if not scored:
        lines.append("- Ninguna señal de relación encontrada (página aislada).")
        return "\n".join(lines)

    for score, candidate, breakdown in scored[:top_n]:
        lines.append(f"- [[{candidate}]] ({score:.1f}) — {', '.join(breakdown)}")

    return "\n".join(lines)


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


# hot.md's "Última sesión" grows the same way "Decisiones pendientes" used to
# before archive_pendiente/PENDIENTES_WARN_CHARS: dense prose, one paragraph
# per work session, never pruned. A manual step already existed for this
# (vault-maintenance skill, SÍNTESIS SEMANAL step 10, "Compactar hot.md") but
# text instructions get skipped when the easy path is available — same
# pattern already fixed elsewhere in this vault with hard hooks instead of
# prose reminders (image-check in INGERIR, dangerous-git blocking). prune_hot
# is that hook for this section.
ULTIMA_SESION_WARN_CHARS = 20_000
# Bajado de 7 a 3 (14/7/2026): medido el crecimiento real de hot.md via git
# history (63KB 4/7 -> 137KB 9/7 -> compactado a 111KB 12/7 -> ya en 129KB
# el 14/7, solo 2 dias despues) -- la ventana de 7 dias no daba abasto al
# ritmo real de esta semana. Ver wiki/log.md 2026-07-14.
ULTIMA_SESION_RETENTION_DAYS = 3

SESSION_ENTRY_HEADER_PATTERN = re.compile(r"^\*\*.+?\*\*", re.DOTALL)
LOG_MD_DATE_PATTERN = re.compile(r"log\.md`?\s*(\d{4}-\d{2}-\d{2})")


def _ultima_sesion_section(text: str) -> re.Match | None:
    return re.search(r"^## Última sesión\n.*?(?=\n## |\Z)", text, re.DOTALL | re.MULTILINE)


def _entry_own_date(entry_text: str) -> date | None:
    """The date in the entry's own header — the first dd/mm/yyyy in the text,
    since the header always comes first ('**Sesión del DD/M/YYYY ...**')."""
    m = DATE_PATTERN.search(entry_text)
    if not m:
        return None
    day, month, year = m.groups()
    try:
        return date(int(year), int(month), int(day))
    except ValueError:
        return None


def _entry_summary_line(entry_text: str) -> str:
    """First sentence-ish chunk of the entry body, after its bold header."""
    header_match = SESSION_ENTRY_HEADER_PATTERN.match(entry_text)
    body = entry_text[header_match.end():].strip() if header_match else entry_text
    first = re.split(r"(?<=[.:])\s", body, maxsplit=1)[0].strip()
    if len(first) > 160:
        first = first[:157].rstrip() + "..."
    return first


def prune_hot(dry_run: bool = True) -> str:
    """Compact hot.md's 'Última sesión' entries older than
    ULTIMA_SESION_RETENTION_DAYS into a 1-line stub, replacing the manual
    "compactar a mano" step of SÍNTESIS SEMANAL with a deterministic one.

    An entry is only compacted if it mentions 'log.md' — that's the signal
    its full detail already lives there, so collapsing it here loses nothing.
    Entries without that pointer are left untouched and flagged in the
    report, same defensive posture as archive_pendiente refusing an
    ambiguous marker: never silently drop content that isn't provably
    recoverable elsewhere.

    Returns the report; only writes to hot.md when dry_run=False.
    """
    hot_path = VAULT_ROOT / "wiki" / "hot.md"
    hot_text = hot_path.read_text(encoding="utf-8")
    section_match = _ultima_sesion_section(hot_text)
    if not section_match:
        return "No se encontró la sección 'Última sesión' en hot.md."

    sec_start, sec_end = section_match.span()
    section = hot_text[sec_start:sec_end]
    header, _, body = section.partition("\n")
    body = body.strip("\n")

    if not body.strip():
        return "La sección 'Última sesión' está vacía — nada para podar."

    entries = [e.strip() for e in re.split(r"\n\n(?=\*\*Sesión )", body) if e.strip()]
    cutoff = date.today() - timedelta(days=ULTIMA_SESION_RETENTION_DAYS)

    new_entries: list[str] = []
    compacted_titles: list[str] = []
    kept_window_titles: list[str] = []
    kept_no_pointer_titles: list[str] = []

    for entry in entries:
        header_match = SESSION_ENTRY_HEADER_PATTERN.match(entry)
        header_text = header_match.group(0) if header_match else entry.splitlines()[0]
        entry_date = _entry_own_date(entry)

        if entry_date is None or entry_date >= cutoff:
            new_entries.append(entry)
            kept_window_titles.append(header_text)
            continue

        if "log.md" not in entry:
            new_entries.append(entry)
            kept_no_pointer_titles.append(header_text)
            continue

        log_date_match = LOG_MD_DATE_PATTERN.search(entry)
        log_date = log_date_match.group(1) if log_date_match else entry_date.isoformat()
        stub = f"{header_text} {_entry_summary_line(entry)} → `log.md` {log_date}."
        new_entries.append(stub)
        compacted_titles.append(header_text)

    lines = ["# Reporte de poda de hot.md — 'Última sesión'", ""]
    lines.append(f"## COMPACTADAS ({len(compacted_titles)})")
    lines.extend(f"- {t}" for t in compacted_titles)
    if not compacted_titles:
        lines.append("- Ninguna")
    lines.append("")
    lines.append(f"## CONSERVADAS POR VENTANA — últimos {ULTIMA_SESION_RETENTION_DAYS} días ({len(kept_window_titles)})")
    lines.extend(f"- {t}" for t in kept_window_titles)
    if not kept_window_titles:
        lines.append("- Ninguna")
    lines.append("")
    lines.append(f"## CONSERVADAS SIN POINTER A log.md ({len(kept_no_pointer_titles)})")
    lines.extend(f"- {t}" for t in kept_no_pointer_titles)
    if not kept_no_pointer_titles:
        lines.append("- Ninguna")

    report = "\n".join(lines)

    if dry_run:
        return report

    if not compacted_titles:
        return report + "\n\nNada para compactar — hot.md no se tocó."

    new_body = "\n\n".join(new_entries)
    new_section = header + "\n\n" + new_body + "\n"
    new_hot_text = hot_text[:sec_start] + new_section + hot_text[sec_end:]
    hot_path.write_text(new_hot_text, encoding="utf-8")
    return report + "\n\nEscrito en wiki/hot.md."


# "Brechas documentadas" (## Fallas abiertas y pendientes) grows the same
# way "Última sesión" did before prune_hot: dense forensic bullets about
# fixes to code outside the vault (modo_bienestar.py, Cold Turkey, etc.),
# never pruned once the fix lands. Unlike "Última sesión", these bullets
# don't reliably say "log.md" — Ivan's own convention here is
# "**cerrado (DD/MM/YYYY)**" / "**resuelto (DD/MM/YYYY)**" without a pointer.
# So recoverability can't be trusted from a text marker alone: this checks
# log.md for an actual dated section header before collapsing anything.
# See wiki/log.md 2026-08-11 for the incident that motivated this (hot.md
# hit ~141k chars and get_hot()/Read both started failing on token limits).
BRECHAS_WARN_CHARS = 15_000
CERRADO_MARKER_PATTERN = re.compile(
    r"\*\*[^*]*?\b(?:[Cc]errad[oa]s?|[Rr]esuelt[oa]s?)\b[^*(]{0,60}?\((\d{1,2})/(\d{1,2})/(\d{4})\)"
)
PENDIENTE_WORD_PATTERN = re.compile(r"\bpendiente\b", re.IGNORECASE)
LOG_MD_HEADER_PATTERN = re.compile(r"^## (\d{4}-\d{2}-\d{2})", re.MULTILINE)


def _brechas_section(text: str) -> re.Match | None:
    return re.search(
        r"^### Brechas documentadas\n.*?(?=\n### |\n## |\Z)", text, re.DOTALL | re.MULTILINE
    )


def _brechas_entries(body: str) -> list[str]:
    """Top-level bullets (lines starting with '- ' at column 0), each
    including any indented continuation paragraphs that follow it — same
    idea as splitting 'Última sesión' on '**Sesión ', but bullet-based since
    this section has no per-entry header convention."""
    return [e.strip("\n") for e in re.split(r"\n(?=- )", body) if e.strip()]


def _entry_bold_title(entry_text: str) -> str:
    m = re.search(r"\*\*(.+?)\*\*", entry_text, re.DOTALL)
    return m.group(1).strip() if m else entry_text.splitlines()[0].strip("- ").strip()


def _log_md_has_date(log_text: str, iso_date: str) -> bool:
    return any(d == iso_date for d in LOG_MD_HEADER_PATTERN.findall(log_text))


def prune_brechas(dry_run: bool = True) -> str:
    """Compact wiki\\hot.md's '### Brechas documentadas' section (under
    '## Fallas abiertas y pendientes'): bullets whose most recent
    '**cerrado/resuelto (DD/MM/YYYY)**' marker has a matching dated section
    in log.md, and that don't also mention an unresolved 'pendiente'
    elsewhere in the bullet, get collapsed to a 1-line stub pointing at
    log.md. Everything else — no cierre marker, an embedded 'pendiente', or
    a cierre date log.md doesn't actually have a section for — is left
    untouched and reported separately, so nothing gets dropped without a
    verified copy elsewhere.

    Returns the report; only writes to hot.md when dry_run=False.
    """
    hot_path = VAULT_ROOT / "wiki" / "hot.md"
    hot_text = hot_path.read_text(encoding="utf-8")
    section_match = _brechas_section(hot_text)
    if not section_match:
        return "No se encontró la sección '### Brechas documentadas' en hot.md."

    sec_start, sec_end = section_match.span()
    section = hot_text[sec_start:sec_end]
    header, _, body = section.partition("\n")
    body = body.strip("\n")

    if not body.strip():
        return "La sección 'Brechas documentadas' está vacía — nada para podar."

    log_path = VAULT_ROOT / "wiki" / "log.md"
    log_text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""

    entries = _brechas_entries(body)

    new_entries: list[str] = []
    compacted_titles: list[str] = []
    kept_no_marker: list[str] = []
    kept_has_pendiente: list[str] = []
    kept_no_log_backup: list[str] = []

    for entry in entries:
        title = _entry_bold_title(entry)
        markers = list(CERRADO_MARKER_PATTERN.finditer(entry))

        if not markers:
            new_entries.append(entry)
            kept_no_marker.append(title)
            continue

        if PENDIENTE_WORD_PATTERN.search(entry):
            new_entries.append(entry)
            kept_has_pendiente.append(title)
            continue

        day, month, year = markers[-1].groups()
        try:
            iso_date = date(int(year), int(month), int(day)).isoformat()
        except ValueError:
            new_entries.append(entry)
            kept_no_marker.append(title)
            continue

        if not _log_md_has_date(log_text, iso_date):
            new_entries.append(entry)
            kept_no_log_backup.append(f"{title} (cerrado {day}/{month}/{year}, sin sección en log.md)")
            continue

        stub = f"- **{title}** — cerrado ({day}/{month}/{year}). Detalle completo en `log.md` {iso_date}."
        new_entries.append(stub)
        compacted_titles.append(title)

    lines = ["# Reporte de poda de hot.md — 'Brechas documentadas'", ""]
    lines.append(f"## COMPACTADAS ({len(compacted_titles)})")
    lines.extend(f"- {t}" for t in compacted_titles)
    if not compacted_titles:
        lines.append("- Ninguna")
    lines.append("")
    lines.append(f"## CONSERVADAS SIN MARCA DE CIERRE ({len(kept_no_marker)})")
    lines.extend(f"- {t}" for t in kept_no_marker)
    if not kept_no_marker:
        lines.append("- Ninguna")
    lines.append("")
    lines.append(f"## CONSERVADAS — TIENEN 'PENDIENTE' EMBEBIDO ({len(kept_has_pendiente)})")
    lines.extend(f"- {t}" for t in kept_has_pendiente)
    if not kept_has_pendiente:
        lines.append("- Ninguna")
    lines.append("")
    lines.append(f"## CONSERVADAS — CERRADO SIN RESPALDO VERIFICADO EN log.md ({len(kept_no_log_backup)})")
    lines.extend(f"- {t}" for t in kept_no_log_backup)
    if not kept_no_log_backup:
        lines.append("- Ninguna")

    report = "\n".join(lines)

    if dry_run:
        return report

    if not compacted_titles:
        return report + "\n\nNada para compactar — hot.md no se tocó."

    new_body = "\n".join(new_entries)
    new_section = header + "\n\n" + new_body + "\n"
    new_hot_text = hot_text[:sec_start] + new_section + hot_text[sec_end:]
    hot_path.write_text(new_hot_text, encoding="utf-8")
    return report + "\n\nEscrito en wiki/hot.md."


def _hash_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _save_manifest(data: dict) -> None:
    MANIFEST_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def check_ingested(paths: list[str]) -> str:
    """Check which _Inbox/ files (given as paths relative to _Inbox/) were
    already evaluated in a previous INGERIR run, by content hash (MD5) — not
    filename, which isn't stable across re-clips of the same tweet/URL.

    Call this once for the whole batch, before reading any file content, at
    the start of INGERIR. Replaces the manual grep of wiki/log.md for
    '[DESCARTADO]' entries — this covers every verdict (pasa/descartado/
    fragmento_extraido/link_con_anotacion), not just discards.
    """
    manifest = _load_manifest()
    already_seen: list[tuple[str, dict]] = []
    new: list[str] = []

    for rel_path in paths:
        file_path = _resolve_inbox_path(rel_path)
        file_hash = _hash_file(file_path)
        entry = manifest.get(file_hash)
        if entry:
            already_seen.append((rel_path, entry))
        else:
            new.append(rel_path)

    lines = ["# Chequeo de archivos ya ingeridos", ""]
    lines.append(f"## Ya evaluados — saltar ({len(already_seen)})")
    for rel_path, entry in already_seen:
        lines.append(
            f"- `{rel_path}` → {entry['veredicto']} ({entry['fecha']}): {entry['motivo']}"
            + (f" → [[{entry['pagina']}]]" if entry.get("pagina") else "")
        )
    if not already_seen:
        lines.append("- Ninguno")
    lines.append("")
    lines.append(f"## Nuevos — evaluar ({len(new)})")
    for rel_path in new:
        lines.append(f"- `{rel_path}`")
    if not new:
        lines.append("- Ninguno")

    return "\n".join(lines)


def register_ingested(path: str, veredicto: str, motivo: str, pagina: str | None = None) -> str:
    """Record the verdict for one _Inbox/ file in the manifest, keyed by its
    content hash (MD5). Call this once per file when its INGERIR verdict is
    final — whether it passed (created a page) or was discarded.
    """
    if veredicto not in VEREDICTO_VALORES_VALIDOS:
        raise ValueError(
            f"veredicto debe ser uno de {sorted(VEREDICTO_VALORES_VALIDOS)}, recibido: {veredicto!r}"
        )

    file_path = _resolve_inbox_path(path)
    file_hash = _hash_file(file_path)

    manifest = _load_manifest()
    manifest[file_hash] = {
        "filename": file_path.name,
        "fecha": date.today().isoformat(),
        "veredicto": veredicto,
        "motivo": motivo,
        "pagina": pagina,
    }
    _save_manifest(manifest)

    return f"Registrado: `{file_path.name}` → {veredicto} ({motivo})"


_RELATED_PAGE_LINE = re.compile(r"\[\[([^\]]+)\]\]\s*\(([0-9.]+)\)")


def _parse_related_pages_output(text: str) -> list[tuple[str, float]]:
    """Extract (page_id, score) pairs from get_related_pages' formatted
    output (lines like '- [[page]] (7.5) — breakdown'). Returns [] for
    its warning-string case (nonexistent page) or its no-signal case
    ('Ninguna señal de relación encontrada') — neither contains a
    [[wikilink]](score) pair, so search_hybrid treats both as "no graph
    signal from this anchor", not an error."""
    return [(name, float(score)) for name, score in _RELATED_PAGE_LINE.findall(text)]


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
