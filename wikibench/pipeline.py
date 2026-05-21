"""BenchmarkRun — main pipeline tying retrieval, LLM, and DB together."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import yaml

from . import db, datasets as ds, metrics as met, ollama
from .retrieval import Retriever


# ── Config hash ───────────────────────────────────────────────────────────────

def config_hash(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


# ── Load config ───────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


_REQUIRED = {"model", "retrieval", "index_dir", "dataset_path"}


def _validate(cfg: dict) -> None:
    missing = _REQUIRED - cfg.keys()
    if missing:
        raise ValueError(f"Config missing required keys: {missing}")
    valid_retrieval = {"dense", "sparse", "hybrid_rrf", "structured"}
    if cfg["retrieval"] not in valid_retrieval:
        raise ValueError(f"retrieval must be one of {valid_retrieval}")


# ── Main class ────────────────────────────────────────────────────────────────

class BenchmarkRun:
    def __init__(
        self,
        config:      dict,
        db_path:     str | Path,
        tick_cb:     Callable | None = None,
        ollama_url:  str             = "http://localhost:11434",
        warmup:      bool            = True,
        top_k:       int             = 5,
    ):
        self.config      = config
        self.db_path     = Path(db_path)
        self.tick_cb     = tick_cb
        self.ollama_url  = ollama_url
        self.do_warmup   = warmup
        self.top_k       = top_k
        self._c_hash     = config_hash(config)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tick(self, **kw) -> None:
        if self.tick_cb:
            self.tick_cb(kw)

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self) -> dict:
        cfg = self.config
        con = db.open_db(self.db_path)

        # Resume or new run
        run_id = db.find_incomplete_run(con, self._c_hash)
        if run_id is not None:
            print(f"Resuming run {run_id} …")
        else:
            run_id = db.create_run(con, cfg, self._c_hash)
            print(f"Started run {run_id}")

        completed = db.get_completed_indices(con, run_id)

        # Load dataset
        questions = ds.load_medqa(cfg["dataset_path"], n=cfg.get("n_questions"))
        total     = len(questions)
        remaining = [q for q in questions if q["question_idx"] not in completed]
        print(f"  {total} questions total, {len(remaining)} remaining")

        # Setup retriever
        retriever = Retriever(
            index_dir       = cfg["index_dir"],
            embedding_model = cfg.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            top_k           = self.top_k,
        )

        # Warmup
        if self.do_warmup and remaining:
            print("  Warming up model…", flush=True)
            try:
                ollama.warmup(cfg["model"], base_url=self.ollama_url)
            except Exception as e:
                print(f"  Warmup failed (non-fatal): {e}")

        # Main loop
        n_correct = 0
        n_done    = len(completed)

        for q in remaining:
            idx   = q["question_idx"]
            qtext = q["question_text"]
            opts  = q["options"]
            ans   = q["answer"]

            # Retrieval
            t_ret_start = time.perf_counter()
            try:
                chunks = retriever.retrieve(cfg["retrieval"], qtext)
            except Exception as e:
                print(f"  [Q{idx}] retrieval error: {e}")
                chunks = []
            latency_ms = (time.perf_counter() - t_ret_start) * 1000

            # Build prompt
            messages = ds.format_prompt(qtext, opts, chunks)

            # System snapshot before
            before = met.snapshot_before()
            cpu_mon = met.CpuMonitor(interval=0.5)

            # LLM call
            try:
                result = ollama.chat(
                    cfg["model"],
                    messages,
                    base_url = self.ollama_url,
                )
            except Exception as e:
                cpu_mon.stop()
                print(f"  [Q{idx}] LLM error: {e}")
                continue

            cpu_pct = cpu_mon.stop()
            after   = met.snapshot_after(before)

            raw_response = result["content"]
            predicted    = ds.parse_answer(raw_response, set(opts.keys()))
            is_correct   = predicted == ans
            if is_correct:
                n_correct += 1

            # Write to DB
            q_id = db.insert_question(con, run_id, idx, qtext, opts, ans)
            r_id = db.insert_result(
                con, run_id, q_id,
                predicted, is_correct, chunks, latency_ms, raw_response,
            )
            db.insert_llm_metrics(con, r_id, run_id, {
                **result,
                "cpu_pct":       cpu_pct,
                "ram_before_mb": before.ram_mb,
                "ram_after_mb":  after.ram_mb,
                "pi_temp_c":     after.temp_c,
            })

            n_done += 1
            acc_so_far = n_correct / (n_done - len(completed)) if (n_done - len(completed)) > 0 else 0.0
            self._tick(
                done      = n_done,
                total     = total,
                correct   = n_correct,
                accuracy  = acc_so_far,
                question  = idx,
            )

            if n_done % 10 == 0:
                print(
                    f"  [{n_done}/{total}] acc={acc_so_far:.1%}"
                    f"  predicted={predicted}  correct={ans}",
                    flush=True,
                )

        retriever.close()

        # Final accuracy (includes previously completed questions)
        all_results = con.execute(
            "SELECT is_correct FROM results WHERE run_id = ?", (run_id,)
        ).fetchall()
        accuracy = sum(r["is_correct"] for r in all_results) / len(all_results) if all_results else 0.0

        db.complete_run(con, run_id, accuracy)
        con.close()

        summary = {
            "run_id":    run_id,
            "accuracy":  accuracy,
            "n_total":   total,
            "n_correct": sum(r["is_correct"] for r in all_results),
            "model":     cfg["model"],
            "retrieval": cfg["retrieval"],
        }
        print(f"\nRun {run_id} complete — accuracy {accuracy:.1%}")
        return summary
