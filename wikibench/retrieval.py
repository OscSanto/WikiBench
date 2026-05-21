"""Retrieval backends for wikibench.

wikembed DB schema (relevant tables):
  articles(id, title, path, source)
  chunks(id, article_id, chunk_idx, text, section, depth, embedded)
  chunks_fts  — contentless FTS5 (content=''), rowid = chunks.id
               BM25 scores are NEGATIVE — ORDER BY rank ASC for best first.

FAISS index stores chunk IDs as vector IDs (via IndexIDMap).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Sequence

import faiss

from . import embed as embed_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_index(index_dir: str | Path) -> faiss.Index:
    path = Path(index_dir) / "faiss.index"
    idx  = faiss.read_index(str(path))
    # Set efSearch for HNSW inner index
    inner = getattr(idx, "index", idx)
    if hasattr(inner, "hnsw"):
        inner.hnsw.efSearch = 64
    return idx


def _open_chunks_db(index_dir: str | Path) -> sqlite3.Connection:
    path = Path(index_dir) / "chunks.db"
    con  = sqlite3.connect(str(path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def _fetch_chunks(con: sqlite3.Connection, chunk_ids: Sequence[int]) -> list[dict]:
    if not chunk_ids:
        return []
    placeholders = ",".join("?" * len(chunk_ids))
    rows = con.execute(
        f"""SELECT c.id, c.text, c.section, c.depth, a.title, a.path
              FROM chunks c JOIN articles a ON a.id = c.article_id
             WHERE c.id IN ({placeholders})""",
        list(chunk_ids),
    ).fetchall()
    by_id = {r["id"]: dict(r) for r in rows}
    return [by_id[cid] for cid in chunk_ids if cid in by_id]


# ── RRF ───────────────────────────────────────────────────────────────────────

def _rrf_fuse(ranked_lists: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal Rank Fusion. Returns merged list of IDs, best first."""
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, cid in enumerate(lst, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ── Retriever ─────────────────────────────────────────────────────────────────

class Retriever:
    """Single-instance retriever; open once per run, reuse across questions."""

    def __init__(self, index_dir: str | Path, embedding_model: str, top_k: int = 5):
        self.index_dir       = Path(index_dir)
        self.embedding_model = embedding_model
        self.top_k           = top_k
        self._idx = _load_index(index_dir)
        self._con = _open_chunks_db(index_dir)

    def close(self) -> None:
        self._con.close()

    # ── Dense ─────────────────────────────────────────────────────────────────

    def dense(self, query: str) -> list[dict]:
        q_vec = embed_mod.encode_one(query, self.embedding_model)
        _, ids = self._idx.search(q_vec, self.top_k)
        chunk_ids = [int(i) for i in ids[0] if i >= 0]
        return _fetch_chunks(self._con, chunk_ids)

    # ── Sparse (BM25 via FTS5) ────────────────────────────────────────────────

    def sparse(self, query: str) -> list[dict]:
        # chunks_fts is contentless; rowid == chunks.id
        rows = self._con.execute(
            """SELECT c.id, c.text, c.section, c.depth, a.title, a.path
                 FROM chunks_fts
                 JOIN chunks  c ON c.id = chunks_fts.rowid
                 JOIN articles a ON a.id = c.article_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank          -- rank is negative BM25; ASC = best first
                LIMIT ?""",
            (query, self.top_k),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Hybrid RRF ────────────────────────────────────────────────────────────

    def hybrid_rrf(self, query: str) -> list[dict]:
        fetch_n = self.top_k * 3  # over-fetch before fusion

        # Dense
        q_vec = embed_mod.encode_one(query, self.embedding_model)
        _, ids = self._idx.search(q_vec, fetch_n)
        dense_ids = [int(i) for i in ids[0] if i >= 0]

        # Sparse
        rows = self._con.execute(
            """SELECT chunks_fts.rowid AS id
                 FROM chunks_fts
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?""",
            (query, fetch_n),
        ).fetchall()
        sparse_ids = [r["id"] for r in rows]

        fused = _rrf_fuse([dense_ids, sparse_ids])[:self.top_k]
        return _fetch_chunks(self._con, fused)

    # ── Structured ────────────────────────────────────────────────────────────

    def structured(self, query: str) -> list[dict]:
        """Lead sections first (depth=0), then infobox, then dense fallback."""
        q_vec = embed_mod.encode_one(query, self.embedding_model)
        _, ids = self._idx.search(q_vec, self.top_k * 4)
        all_ids = [int(i) for i in ids[0] if i >= 0]
        chunks  = _fetch_chunks(self._con, all_ids)

        seen     = set()
        lead, infobox, rest = [], [], []
        for c in chunks:
            cid = c["id"]
            if c.get("depth") == 0:
                lead.append(c)
                seen.add(cid)
            elif "infobox" in (c.get("section") or "").lower():
                infobox.append(c)
                seen.add(cid)
            else:
                rest.append(c)

        ordered  = lead + infobox + rest
        return ordered[:self.top_k]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def retrieve(self, method: str, query: str) -> list[dict]:
        if method == "dense":
            return self.dense(query)
        if method == "sparse":
            return self.sparse(query)
        if method == "hybrid_rrf":
            return self.hybrid_rrf(query)
        if method == "structured":
            return self.structured(query)
        raise ValueError(f"Unknown retrieval method: {method!r}")
