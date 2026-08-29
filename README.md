# agent-context-server

Servidor MCP que expone un vault de Obsidian (wiki en markdown con frontmatter YAML + `[[wikilinks]]`) como tools para un agente LLM (Claude Code). Reemplaza la carga manual de contexto (`Read` archivo por archivo) que un agente haría a mano cada sesión.

Es la capa 1 de un plan de "second brain autónomo". Capas 1 (MCP server) y 2 (RAG, `search_semantic`/`index_vault`) cerradas. El heartbeat autónomo (capa 3, antes pensado como un runtime propio en este repo) terminó resuelto por otra vía: crons de [Hermes Agent](https://github.com/NousResearch/hermes-agent) que llaman a estas mismas tools por MCP — no hay `agent_runtime/` en este repo.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e .
```

Requiere Python ≥3.11. Dependencias: `mcp`, `python-frontmatter`, `chromadb`, `FlagEmbedding` (declaradas en `pyproject.toml`) — `FlagEmbedding` trae `torch`/`transformers`/etc. como dependencias transitivas, ver el gotcha de orden de imports más abajo. `torch` necesita instalarse aparte con la build CUDA correcta (ver ese mismo gotcha) si hay GPU disponible.

## Registrar en Claude Code

```bash
claude mcp add vault-mcp -- ".venv/Scripts/python.exe" "-m" "vault_mcp.server"
```

Requiere reiniciar Claude Code para que la sesión reconozca el servidor nuevo.

## Arquitectura

- `src/vault_mcp/vault.py` — lógica de filesystem/vault, sin dependencia de MCP. Testeable de forma aislada.
- `src/vault_mcp/server.py` — glue MCP: usa `FastMCP`, registra cada función de `vault.py` como `@mcp.tool()`, corre por stdio.

Separados a propósito: el protocolo MCP puede cambiar independientemente de la lógica del vault.

`VAULT_ROOT` se lee de la variable de entorno `VAULT_MCP_ROOT` (obligatoria, sin fallback — el servidor no arranca sin ella) para poder correr contra cualquier vault de Obsidian con la misma estructura (`wiki/{fuentes,entidades,conceptos,sintesis}/`, frontmatter YAML, `[[wikilinks]]`).

```bash
export VAULT_MCP_ROOT="/path/to/tu/vault"
```

## Tools

20 tools registradas (`src/vault_mcp/server.py`), agrupadas por lo que hacen:

**Contexto de sesión (lectura barata, primero en la cadena de retrieval)**

| Tool | Qué hace |
|------|----------|
| `get_hot()` | Contenido completo de `wiki/hot.md` |
| `get_prioridades()` | Solo la sección "Cuellos de botella activos" de `hot.md` — lista corta y acotada (3-5 ítems), para preguntas de síntesis/prioridades sin traer el archivo entero |
| `get_pendientes()` | Slice de "Decisiones pendientes" acotado a un presupuesto de caracteres (`hot.md` completo puede truncarse antes de llegar a esa sección en un modelo con contexto chico) — ítems viejos quedan listados por título con puntero a `search_semantic`/`pendientes-archivo.md`, nunca truncados a mitad |
| `get_index()` | Contenido de `wiki/index.md` |
| `get_page(name)` | Resuelve un `[[wikilink]]` probando las 4 subcarpetas (`fuentes/`, `entidades/`, `conceptos/`, `sintesis/`) |
| `get_backlinks(name)` | Inverso de `get_page` — qué páginas enlazan hacia `name` (misma lógica de escaneo de wikilinks que el chequeo de huérfanas de `run_lint`, para que ambos queden consistentes por construcción) |
| `get_related_pages(name, top_n=8)` | A diferencia de `get_backlinks` (exacto), esto es una **aproximación heurística**: rankea páginas por relevancia ponderada a `name` con 4 señales — link directo (×3.0), co-citación por una fuente compartida (×4.0), Adamic-Adar sobre vecinos en común (×1.5, pondera más un vecino específico que un hub como `hot.md`), y bonus de mismo tipo (×1.0). Diseño adaptado del modelo de relevancia de [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) (ver `wiki/fuentes/second-brain-wiki-local-ollama-2026.md` del vault), 17/7/2026. Pensado para sugerir candidatos de clúster/síntesis más allá de tags/entidades compartidas — siempre revisar el resultado a ojo, no es una fuente de verdad exacta |

**Búsqueda (texto completo + semántica)**

| Tool | Qué hace |
|------|----------|
| `search_wiki(query, max_results=20)` | Full-text case-insensitive sobre todo `wiki/**/*.md` |
| `search_semantic(query, n_results=5)` | Búsqueda semántica (por significado, no texto literal) contra el índice de `index_vault` — embeddings BGE-M3 + ChromaDB |
| `index_vault(dry_run=True)` | (Re)indexa `wiki/**/*.md` en la colección Chroma (`src/vault_mcp/rag.py`). Incremental desde el 14/7/2026: compara hash de contenido y solo embebe páginas nuevas/modificadas (antes re-embebía las ~500+ páginas del vault en cada corrida, ~1h por CPU); `dry_run=True` (default) solo cuenta qué cambiaría. |

**Escritura (crea/mueve, nunca sobreescribe sin querer)**

| Tool | Qué hace |
|------|----------|
| `create_page(page_type, name, content, metadata)` | Crea una página nueva, valida que `metadata` tenga los campos de frontmatter requeridos por `page_type` (ver `CLAUDE.md` del vault). Nunca sobreescribe. |
| `move_source_file(inbox_path, target_folder)` | Mueve un archivo de `_Inbox/` a su destino PARA (paso de reubicación de `INGERIR`) — valida que no escape de `_Inbox/`, nunca sobreescribe un destino existente. |
| `archive_pendiente(marker)` | Saca un ítem abierto-pero-viejo de "Decisiones pendientes" hacia `wiki/pendientes-archivo.md` (página propia, indexable, en vez de diluirse en la página gigante de `hot.md`). Falla si `marker` no matchea exactamente un ítem. |
| `prune_hot(dry_run=True)` | Compacta las entradas de "Última sesión" en `hot.md` más viejas que la ventana de retención a un stub de 1 línea con puntero a `log.md` — reemplaza el paso manual de compactación de `SÍNTESIS SEMANAL`. |

**Mantenimiento**

| Tool | Qué hace |
|------|----------|
| `build_index(dry_run=True)` | Regenera `index.md` desde el frontmatter de todas las páginas. |
| `run_lint()` | Frontmatter YAML inválido, links rotos, contradicciones `superseded_by` sin cerrar, páginas huérfanas, tamaño de "Decisiones pendientes". |
| `fix_broken_links(dry_run=True)` | Para cada link roto que detectaría `run_lint()`, busca por fuzzy match (`difflib`, sin dependencia nueva) el nombre de página real más parecido y propone (o aplica, con `dry_run=False`) el reemplazo. Solo atrapa typos/renames — un alias real sin overlap de texto (ej. "PC" → "Program Counter") no matchea, queda para arreglo manual. Adaptado de `olw maintain --fix` (kytmanov/obsidian-llm-wiki-local, ver `wiki/fuentes/second-brain-wiki-local-ollama-2026.md`), 17/7/2026 |
| `doctor()` | Health-check de la infra del propio `vault-mcp` (no de contenido — eso es `run_lint()`): subcarpetas de `wiki/`, archivos core, manifest de ingesta, frontmatter, y si el índice semántico está poblado y no muy desalineado del conteo real de páginas. Adaptado de `olw doctor` (mismo repo de inspiración), motivado por 2 incidentes reales donde `vault-mcp` se rompió en silencio (`uv sync` perdiendo el install editable 15/7, un cron desincronizado por un pin de modelo viejo) |

**Calendario local (solo lectura, sin Google Calendar ni nada externo)**

| Tool | Qué hace |
|------|----------|
| `get_local_events(target_date=None, dias=1)` | Eventos de `Calendario/` (plugin Full Calendar Remastered de Obsidian) para una fecha o rango — soporta eventos únicos y recurrentes (por día de semana o Nth-día-del-mes) |
| `check_calendar_overlaps(target_date=None, dias=14)` | Chequea solapamientos en todo un rango de una sola pasada, en vez de descubrirlos uno a uno sesión tras sesión |

## Known gotchas

- **YAML frontmatter + wikilinks**: `fuente_original: [[a]] + [[b]]` rompe el parser (el `[` inicial se lee como lista de flujo) — quotear el valor. Tolerado por archivo: no crashea el resto de las tools, pero esa página queda invisible para `build_index`/`run_lint` hasta corregirse.
- **Windows: orden de imports.** `torch`/`pandas`/`transformers` deben importarse antes que `chromadb`/`FastMCP` en `rag.py`/`server.py` — otro orden produce `access violation` al cargar `FlagEmbedding` (conflicto de carga de DLLs nativas entre `pyarrow.lib` y el resto, no un problema de versiones). `torch` debe ser build CUDA `>=2.6` (`+cu124`, requisito de `transformers` por CVE-2025-32434).

## Testing

33 tests (`unittest`, `tests/`) sobre la lógica de `vault.py`/`fts.py`/`rag.py` en directorios temporales aislados. Sin suite de integración contra el vault real todavía.
