"""Export wikibench run data to CSV or JSONL."""
from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

from .db import get_run_rows, get_run_summary


def to_csv(con: sqlite3.Connection, run_id: int, out_path: str | Path) -> Path:
    rows = get_run_rows(con, run_id)
    if not rows:
        raise ValueError(f"No data for run {run_id}")
    out = Path(out_path)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return out


def to_jsonl(con: sqlite3.Connection, run_id: int, out_path: str | Path) -> Path:
    rows = get_run_rows(con, run_id)
    if not rows:
        raise ValueError(f"No data for run {run_id}")
    out = Path(out_path)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return out


def export_run(
    con:      sqlite3.Connection,
    run_id:   int,
    out_dir:  str | Path,
    fmt:      str = "both",   # "csv" | "jsonl" | "both"
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"run_{run_id}"
    result = {}
    if fmt in ("csv", "both"):
        result["csv"] = to_csv(con, run_id, out / f"{stem}.csv")
    if fmt in ("jsonl", "both"):
        result["jsonl"] = to_jsonl(con, run_id, out / f"{stem}.jsonl")
    return result
