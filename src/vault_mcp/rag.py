"""Semantic search over the vault, backed by an embedded ChromaDB collection."""

import hashlib
from pathlib import Path

# Import order matters here on Windows: loading chromadb (or FlagEmbedding)
# cold, before torch/pandas/sklearn/transformers/datasets are already
# resident, crashes the interpreter with an access violation inside
# pyarrow.lib — a native DLL load-order conflict, not a version mismatch.
# Pre-importing these five first, in this order, at module load time (not
# inside a function — chromadb below would otherwise poison the DLL state
# first) makes every later import of these libraries a no-op against an
# already-consistent state.
import torch  # noqa: F401
import pandas  # noqa: F401
import sklearn  # noqa: F401
import transformers  # noqa: F401
import datasets  # noqa: F401

import chromadb

from vault_mcp.vault import VAULT_ROOT

INDEX_PATH = Path(__file__).parent / "chroma_index"


_model = None


def _get_model():
    """Lazily load BGE-M3 (first call only — the model stays resident for
    the life of the MCP server process, ~2GB VRAM on Ivan's RTX 3060)."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
    return _model


def _embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Dense embeddings only — BGE-M3 also produces sparse/ColBERT vectors,
    not needed at this vault's scale (~230 pages).

    Batched with progress printed per batch — silent for single-query calls
    (search_semantic), useful once index_vault's page count grows enough
    that a run takes real minutes instead of seconds.
    """
    model = _get_model()
    total = len(texts)
    vecs: list[list[float]] = []
    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        output = model.encode(batch, max_length=8192)
        vecs.extend(output["dense_vecs"].tolist())
        if total > batch_size:
            print(f"[rag] embeddings: {min(start + batch_size, total)}/{total}", flush=True)
    return vecs


def get_collection():
    """Return the persistent Chroma collection, creating it if missing.

    Cosine distance to match BGE-M3, whose dense vectors are trained/compared
    for cosine similarity (Chroma's default is L2, which would be wrong here).
    """
    client = chromadb.PersistentClient(path=str(INDEX_PATH))
    return client.get_or_create_collection("wiki", metadata={"hnsw:space": "cosine"})


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def index_vault(dry_run: bool = True) -> str:
    """Walk every wiki/**/*.md page and (re)index it into the collection.

    Incremental (agregado 14/7/2026): compara el hash de contenido de cada
    página contra el que ya está guardado en la colección (metadata
    `content_hash`) y solo embebe páginas nuevas o modificadas — el
    embedding es el paso caro (BGE-M3 por CPU, ~1 hora para las ~522
    páginas del vault a mediados de julio 2026), releer todo sin cambios
    era el motivo real de esa demora. También borra de la colección las
    páginas que ya no existen en el vault (movidas/renombradas/borradas).

    Corrido por GPU desde el 17/7/2026 (antes corría por CPU sin que nadie
    lo notara — el venv tenía `torch==2.13.0+cpu`, el wheel default de
    PyPI para Windows, en vez de una build CUDA; `pyproject.toml` ahora fija
    `torch` como dependencia directa apuntando al índice
    `download.pytorch.org/whl/cu126` vía `[tool.uv.sources]`, necesario
    porque uv solo aplica overrides de índice a dependencias directas, no
    transitivas). Con RTX 3060: ~0.75s para 32 textos, vs. la ~1h por CPU
    de la nota de arriba.

    One entry per page (not per chunk) — matches the vault's existing
    short, structured page convention.
    """
    collection = get_collection()
    pages = sorted((VAULT_ROOT / "wiki").rglob("*.md"))

    current: dict[str, str] = {}
    for page in pages:
        rel_path = page.relative_to(VAULT_ROOT / "wiki").as_posix()
        current[rel_path] = page.read_text(encoding="utf-8")

    existing = collection.get(include=["metadatas"])
    existing_hashes = {
        id_: (meta or {}).get("content_hash")
        for id_, meta in zip(existing["ids"], existing["metadatas"])
    }

    to_index_ids = [
        rel_path
        for rel_path, text in current.items()
        if existing_hashes.get(rel_path) != _content_hash(text)
    ]
    stale_ids = [id_ for id_ in existing_hashes if id_ not in current]

    if dry_run:
        return (
            f"[dry_run] {len(to_index_ids)} páginas nuevas/modificadas para embeber, "
            f"{len(stale_ids)} páginas obsoletas para borrar, "
            f"{len(current) - len(to_index_ids)} sin cambios (se saltean). "
            f"Ejemplo a embeber: {to_index_ids[:3]}"
        )

    if stale_ids:
        collection.delete(ids=stale_ids)

    if not to_index_ids:
        return f"Nada para embeber — las {len(current)} páginas ya estaban al día. {len(stale_ids)} obsoleta(s) borrada(s)."

    # Upsert por tanda (no todo junto al final): si el proceso se corta a
    # mitad de camino (pasó una vez, corriendo por CPU casi 1h20), lo ya
    # embebido queda guardado en disco — la próxima corrida solo re-embebe
    # lo que faltó, no arranca de cero.
    batch_size = 32
    done = 0
    for start in range(0, len(to_index_ids), batch_size):
        batch_ids = to_index_ids[start : start + batch_size]
        batch_docs = [current[rel_path] for rel_path in batch_ids]
        batch_embeddings = _embed(batch_docs, batch_size=batch_size)
        batch_metadatas = [{"content_hash": _content_hash(doc)} for doc in batch_docs]
        collection.upsert(
            ids=batch_ids, documents=batch_docs, embeddings=batch_embeddings, metadatas=batch_metadatas
        )
        done += len(batch_ids)
        print(f"[rag] upsert: {done}/{len(to_index_ids)}", flush=True)

    return (
        f"Indexadas {len(to_index_ids)} páginas nuevas/modificadas, "
        f"{len(stale_ids)} obsoleta(s) borrada(s), "
        f"{len(current) - len(to_index_ids)} sin cambios (salteadas)."
    )


def search_semantic(query: str, n_results: int = 5) -> str:
    """Semantic search: find the pages whose meaning is closest to `query`."""
    collection = get_collection()

    if collection.count() == 0:
        return "El índice semántico está vacío — correr index_vault(dry_run=False) primero."

    query_embedding = _embed([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    ids = results["ids"][0]
    documents = results["documents"][0]
    distances = results["distances"][0]

    if not ids:
        return f"Sin resultados para '{query}'"

    lines = [f"# Resultados semánticos: '{query}'", ""]
    for page_id, document, distance in zip(ids, documents, distances):
        preview = " ".join(document.strip().splitlines())[:300]
        lines.append(f"## {page_id} (distancia: {distance:.3f})")
        lines.append(preview)
        lines.append("")

    return "\n".join(lines)
