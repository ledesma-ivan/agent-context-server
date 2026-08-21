# search_hybrid — fusión BM25 + vectorial + grafo con RRF

## Contexto

`vault-mcp` ya tiene 2 de las 3 patas de búsqueda que "LLM Wiki v2" (extensión externa
del patrón de Karpathy, ver `wiki/fuentes/second-brain-estado-arte-agosto-2026.md` del
vault) recomienda para vaults que pasaron ~200-500 páginas: `search_semantic`
(vectorial, BGE-M3+Chroma, desde 8/7/2026) y una tercera señal de grafo que **ya
existe pero no está identificada como tal**: `get_related_pages` (Adamic-Adar +
co-citación + link directo + afinidad de tipo, adaptado de `nashsu/llm_wiki`,
17/7/2026). Lo que falta es (a) que `search_wiki` sea BM25 real en vez de substring
plano sin ranking, y (b) una tool que fusione las tres con reciprocal rank fusion
(RRF), como hace el proyecto externo que motivó esta investigación.

Con el vault en 793+ páginas indexadas, ya está en la zona donde el índice plano
(`index.md`) deja de alcanzar como mecanismo primario de búsqueda — motivo original
por el que existe `search_semantic`, y ahora el mismo argumento aplica al resto.

## Objetivo

1. Reemplazar `search_wiki` por una implementación BM25 real (SQLite FTS5), misma
   firma pública.
2. Nueva tool `search_hybrid(query, n_results)` que fusiona BM25 + vectorial + grafo
   vía RRF y devuelve un resultado único rankeado.
3. Mantenimiento del índice FTS5 integrado a `index_vault()` (mismo paso que ya
   mantiene el índice de Chroma).

## No-objetivos

- No se toca el algoritmo de `get_related_pages` (ya validado, adaptado de una fuente
  externa, con tests implícitos de uso real desde 17/7).
- No se reemplaza `index.md` como catálogo legible por humanos — sigue existiendo,
  solo deja de ser el mecanismo primario de retrieval para el agente.
- No se benchmarkea contra LongMemEval ni ningún benchmark externo — el 95.2% citado
  en la fuente que motivó esto es autoreportado por el autor del proyecto que lo
  inspira, sin corroboración independiente (ver fuente citada arriba). Este spec no
  persigue ese número, solo el patrón arquitectónico.

## Arquitectura

### 1. `src/vault_mcp/fts.py` (nuevo módulo)

Mismo patrón que `rag.py`: índice persistente en disco junto a `chroma_index/`
(`fts_index.db`), tabla virtual SQLite FTS5 `pages(content, page_id UNINDEXED)`.

- `index_fts(dry_run: bool = True) -> str` — reconstrucción incremental: hash de
  contenido por página (mismo criterio que `rag.index_vault`), solo re-indexa
  nuevo/modificado, borra páginas obsoletas. Mismo formato de mensaje de retorno que
  `index_vault` para consistencia.
- `search_wiki_ranked(query: str, max_results: int) -> list[tuple[str, float]]` —
  función interna, devuelve `(page_id, bm25_score)` rankeado. Usa el `bm25()` nativo
  de FTS5, `snippet()` nativo para preview.
- `search_wiki(query: str, max_results: int = 20) -> str` — reemplaza la función
  actual en `vault.py` (se mueve acá). Misma firma pública, mismo formato de salida
  agrupado por archivo con snippets — nada que ya llama `search_wiki` (INICIO DE
  SESIÓN, CONSULTAR) cambia de comportamiento observable, solo mejora el ranking.

`server.py` actualiza su import de `search_wiki` de `vault` a `fts`.

### 2. `rag.py` — mantenimiento combinado

`index_vault(dry_run)` pasa a llamar también `fts.index_fts(dry_run)` y concatena
ambos reportes en el string de retorno. Un solo paso de mantenimiento para los dos
índices — no cambia la firma ni el paso 13.b de `INGERIR`.

Se extrae también `search_semantic_ranked(query, n_results) -> list[tuple[str,
float]]` (page_id + distancia) como función interna reusable, separada del
formateo a string que hace `search_semantic`.

### 3. `vault.py` — `search_hybrid(query, n_results=5) -> str` (nueva tool)

1. `bm25_ranked = fts.search_wiki_ranked(query, max_results=20)`
2. `vec_ranked = rag.search_semantic_ranked(query, n_results=20)`
3. `stage1 = _rrf_fuse([bm25_ranked, vec_ranked], k=60)` → top 3-5 page_ids
4. Para cada page_id de `stage1`: `get_related_pages(page_id, top_n=5)` → páginas
   relacionadas por grafo (se descarta silenciosamente si `page_id` no resuelve a
   página real — ya devuelve string de warning, no excepción).
5. `graph_ranked` = lista combinada de resultados de grafo, rankeada por su propio
   score ya calculado por `get_related_pages`.
6. `final = _rrf_fuse([stage1, graph_ranked], k=60)` → top `n_results`.
7. Formatea salida: página, score final, qué señal(es) la trajeron (BM25/vector/
   grafo) — mismo estilo que `search_semantic` (bloques `## page_id`).

`_rrf_fuse(ranked_lists: list[list[tuple[str, float]]], k: int = 60) -> list[tuple[str,
float]]` — función pura, `score(page) = Σ 1/(k + rank_en_esa_lista)` sobre las listas
donde aparece, ordena descendente. Sin dependencias nuevas.

### Edge cases

- Índice FTS5 o Chroma vacío → mismo mensaje que hoy usa `search_semantic`
  ("correr index_vault(dry_run=False) primero"), sin excepción.
- Sin resultados en ningún motor → `"Sin resultados para '{query}'"`.
- Resultado en un solo motor → RRF lo maneja solo (score 0 en las listas donde no
  aparece).
- Página repetida entre señales → RRF dedupea por construcción (suma scores por
  page_id).
- Ancla de grafo que no resuelve → se skipea, no frena el resto.
- Costo: el paso más caro (embedding de la query) ya lo paga hoy `search_semantic`
  solo — `search_hybrid` no agrega costo por encima de llamar las 3 tools por
  separado.

## Testing

Convención existente del repo: `unittest`, carga de módulo por path (sin instalar el
paquete), sin red ni modelo real.

- `tests/test_fts.py`: índice FTS5 sobre vault temporal (3-4 `.md` con contenido
  conocido) → ranking correcto (no por orden de archivo); reindexado incremental
  solo re-embebe lo modificado.
- `tests/test_search_hybrid.py`: `_rrf_fuse` con casos calculables a mano; página con
  score en un solo motor no rompe la fusión; ancla de grafo sin resolver no frena el
  resto; query sin resultados en ningún motor → mensaje esperado sin excepción.
- No se testea `search_semantic`/embeddings reales (BGE-M3 sería lento sin GPU en
  CI) — se mockea `_embed` donde haga falta.

## Migración / compatibilidad

- `search_wiki` cambia de implementación, no de firma ni de forma de output — nada
  que la invoque hoy (workflows de `CLAUDE.md`) necesita cambios.
- `index_vault` sigue siendo el único paso de mantenimiento de índices que
  `INGERIR` paso 13.b ya llama — sin cambios en el workflow del vault.
- `search_hybrid` es aditiva: no reemplaza `search_wiki`/`search_semantic`/
  `get_related_pages` como tools independientes, todas siguen expuestas.
