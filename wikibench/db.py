"""SQLite helpers for wikibench benchmark runs."""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path


def open_db(db_path: Path) -> sqlite3.Connection:
    schema = Path(__file__).parent / "schema.sql"
    con = sqlite3.connect(str(db_path), check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(schema.read_text())
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.commit()
    return con


# ── Runs ──────────────────────────────────────────────────────────────────────

def find_incomplete_run(con: sqlite3.Connection, config_hash: str) -> int | None:
    """Return run_id of an incomplete run with matching config, or None."""
    row = con.execute(
        "SELECT id FROM runs WHERE config_hash = ? AND completed_at IS NULL"
        " ORDER BY id DESC LIMIT 1",
        (config_hash,),
    ).fetchone()
    return row["id"] if row else None


def create_run(con: sqlite3.Connection, config: dict, config_hash: str) -> int:
    cur = con.execute(
        """INSERT INTO runs
           (model, embedding_model, retrieval, index_dir, dataset_path,
            n_questions, started_at, config_hash, config_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            config["model"],
            config.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            config["retrieval"],
            config["index_dir"],
            config["dataset_path"],
            config.get("n_questions"),
            time.strftime("%Y-%m-%dT%H:%M:%S"),
            config_hash,
            json.dumps(config),
        ),
    )
    con.commit()
    return cur.lastrowid


def complete_run(con: sqlite3.Connection, run_id: int, accuracy: float) -> None:
    con.execute(
        "UPDATE runs SET completed_at = ?, accuracy = ? WHERE id = ?",
        (time.strftime("%Y-%m-%dT%H:%M:%S"), accuracy, run_id),
    )
    con.commit()


def get_completed_indices(con: sqlite3.Connection, run_id: int) -> set[int]:
    """Return set of question_idx already recorded for this run (for resume)."""
    rows = con.execute(
        "SELECT question_idx FROM questions WHERE run_id = ?", (run_id,)
    ).fetchall()
    return {r["question_idx"] for r in rows}


def get_run_summary(con: sqlite3.Connection, run_id: int) -> dict | None:
    row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def list_runs(con: sqlite3.Connection) -> list[dict]:
    rows = con.execute(
        "SELECT * FROM runs ORDER BY id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── Questions ─────────────────────────────────────────────────────────────────

def insert_question(
    con:           sqlite3.Connection,
    run_id:        int,
    question_idx:  int,
    question_text: str,
    options:       dict,
    correct:       str,
) -> int:
    cur = con.execute(
        """INSERT OR IGNORE INTO questions
           (run_id, question_idx, question_text, options_json, correct_answer)
           VALUES (?, ?, ?, ?, ?)""",
        (run_id, question_idx, question_text, json.dumps(options), correct),
    )
    if cur.lastrowid:
        con.commit()
        return cur.lastrowid
    return con.execute(
        "SELECT id FROM questions WHERE run_id = ? AND question_idx = ?",
        (run_id, question_idx),
    ).fetchone()["id"]


# ── Results ───────────────────────────────────────────────────────────────────

def insert_result(
    con:          sqlite3.Connection,
    run_id:       int,
    question_id:  int,
    predicted:    str | None,
    is_correct:   bool,
    chunks:       list[dict],
    latency_ms:   float,
    raw_response: str,
) -> int:
    cur = con.execute(
        """INSERT INTO results
           (question_id, run_id, predicted_answer, is_correct,
            context_json, n_chunks_retrieved, retrieval_latency_ms, raw_response)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            question_id, run_id, predicted, int(is_correct),
            json.dumps([{k: v for k, v in c.items() if k != "text"} for c in chunks]),
            len(chunks), latency_ms, raw_response,
        ),
    )
    con.commit()
    return cur.lastrowid


def insert_llm_metrics(
    con:       sqlite3.Connection,
    result_id: int,
    run_id:    int,
    m:         dict,
) -> None:
    con.execute(
        """INSERT INTO llm_metrics
           (result_id, run_id,
            total_duration_ms, load_duration_ms,
            prompt_eval_count, prompt_eval_duration_ms,
            eval_count, eval_duration_ms,
            ttft_ms, tokens_per_sec, prefill_rate, cold_start,
            ram_before_mb, ram_after_mb, cpu_pct, pi_temp_c)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            result_id, run_id,
            m.get("total_duration_ms"),   m.get("load_duration_ms"),
            m.get("prompt_eval_count"),   m.get("prompt_eval_duration_ms"),
            m.get("eval_count"),          m.get("eval_duration_ms"),
            m.get("ttft_ms"),             m.get("tokens_per_sec"),
            m.get("prefill_rate"),        int(m.get("cold_start", False)),
            m.get("ram_before_mb"),       m.get("ram_after_mb"),
            m.get("cpu_pct"),             m.get("pi_temp_c"),
        ),
    )
    con.commit()


# ── Export queries ────────────────────────────────────────────────────────────

def get_run_rows(con: sqlite3.Connection, run_id: int) -> list[dict]:
    """Full join of questions + results + metrics for a run (for export)."""
    rows = con.execute(
        """SELECT
               q.question_idx, q.question_text, q.options_json, q.correct_answer,
               r.predicted_answer, r.is_correct,
               r.n_chunks_retrieved, r.retrieval_latency_ms,
               m.total_duration_ms, m.load_duration_ms,
               m.prompt_eval_count, m.prompt_eval_duration_ms,
               m.eval_count, m.eval_duration_ms,
               m.ttft_ms, m.tokens_per_sec, m.prefill_rate, m.cold_start,
               m.ram_before_mb, m.ram_after_mb, m.cpu_pct, m.pi_temp_c
           FROM questions q
           JOIN results r        ON r.question_id = q.id
           LEFT JOIN llm_metrics m ON m.result_id  = r.id
           WHERE q.run_id = ?
           ORDER BY q.question_idx""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]
