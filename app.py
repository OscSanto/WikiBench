"""Streamlit dashboard for wikibench."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="wikibench", page_icon="📊", layout="wide")
st.title("📊 wikibench")
st.caption("Retrieval benchmark suite for wikembed indexes")

# ── DB path ───────────────────────────────────────────────────────────────────

db_path_str = st.sidebar.text_input(
    "Database path",
    value="wikibench.db",
    help="Path to the SQLite database created by wikibench.",
)

if not Path(db_path_str).exists():
    st.info("Enter a valid wikibench.db path in the sidebar, or run a benchmark first.")
    st.stop()

@st.cache_resource
def _get_con(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

con = _get_con(db_path_str)

# ── Sidebar — run new benchmark ───────────────────────────────────────────────

st.sidebar.divider()
st.sidebar.header("Run benchmark")

config_path = st.sidebar.text_input(
    "Config YAML path",
    placeholder="/path/to/configs/example.yaml",
)
ollama_url = st.sidebar.text_input("Ollama URL", value="http://localhost:11434")
top_k      = st.sidebar.slider("Top-K chunks", 1, 20, 5)
no_warmup  = st.sidebar.checkbox("Skip warmup")

if st.sidebar.button("▶ Run", type="primary", disabled=not config_path):
    from wikibench.pipeline import BenchmarkRun, load_config

    progress = st.sidebar.progress(0, text="Starting…")
    status   = st.sidebar.empty()

    def _tick(info: dict) -> None:
        done  = info.get("done", 0)
        total = info.get("total", 1) or 1
        acc   = info.get("accuracy", 0.0)
        progress.progress(min(done / total, 1.0), text=f"{done}/{total} · acc {acc:.1%}")
        status.caption(f"Q{info.get('question', '?')} — predicted {info.get('predicted', '?')}")

    try:
        cfg = load_config(config_path)
        run = BenchmarkRun(
            config     = cfg,
            db_path    = db_path_str,
            tick_cb    = _tick,
            ollama_url = ollama_url,
            warmup     = not no_warmup,
            top_k      = top_k,
        )
        summary = run.run()
        progress.progress(1.0, text="Done!")
        st.sidebar.success(f"Accuracy: {summary['accuracy']:.2%}")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Run failed: {e}")

# ── Runs table ────────────────────────────────────────────────────────────────

from wikibench.db import list_runs, get_run_summary, get_run_rows

runs = list_runs(con)

if not runs:
    st.info("No runs yet. Configure a run in the sidebar.")
    st.stop()

st.subheader("All runs")

import pandas as pd

runs_df = pd.DataFrame(runs)
runs_df["accuracy"] = runs_df["accuracy"].apply(
    lambda x: f"{x:.1%}" if x is not None else "—"
)
display_cols = ["id", "model", "retrieval", "n_questions", "accuracy", "started_at", "completed_at"]
st.dataframe(
    runs_df[[c for c in display_cols if c in runs_df.columns]],
    use_container_width=True,
    hide_index=True,
)

# ── Per-run detail ────────────────────────────────────────────────────────────

st.subheader("Run detail")

run_ids  = [r["id"] for r in runs]
selected = st.selectbox("Select run", run_ids, format_func=lambda i: f"Run {i}")

if selected is not None:
    summary = get_run_summary(con, selected)
    rows    = get_run_rows(con, selected)

    col1, col2, col3, col4 = st.columns(4)
    acc = summary.get("accuracy")
    col1.metric("Accuracy",   f"{acc:.1%}" if acc else "—")
    col2.metric("Model",      summary.get("model", "—").split(":")[-1])
    col3.metric("Retrieval",  summary.get("retrieval", "—"))
    col4.metric("Questions",  len(rows))

    if rows:
        df = pd.DataFrame(rows)

        # ── Correctness breakdown
        st.markdown("#### Results")
        correct_pct = df["is_correct"].mean() * 100 if "is_correct" in df.columns else 0
        st.bar_chart(df["is_correct"].value_counts().rename({0: "Wrong", 1: "Correct"}))

        # ── Timing distributions
        if "tokens_per_sec" in df.columns and df["tokens_per_sec"].notna().any():
            st.markdown("#### LLM performance")
            perf_cols = st.columns(3)
            perf_cols[0].metric(
                "Avg tokens/s",
                f"{df['tokens_per_sec'].mean():.1f}" if df['tokens_per_sec'].notna().any() else "—",
            )
            perf_cols[1].metric(
                "Avg TTFT (ms)",
                f"{df['ttft_ms'].mean():.0f}" if "ttft_ms" in df.columns and df['ttft_ms'].notna().any() else "—",
            )
            perf_cols[2].metric(
                "Avg retrieval (ms)",
                f"{df['retrieval_latency_ms'].mean():.0f}" if df['retrieval_latency_ms'].notna().any() else "—",
            )
            st.line_chart(df[["tokens_per_sec"]].dropna())

        # ── Full table
        with st.expander("Full results table"):
            st.dataframe(df, use_container_width=True, hide_index=True)

        # ── Export
        st.markdown("#### Export")
        from wikibench.export import to_csv, to_jsonl
        import io

        col_csv, col_jsonl = st.columns(2)

        csv_buf = io.StringIO()
        import csv as _csv
        writer = _csv.DictWriter(csv_buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        col_csv.download_button(
            "⬇ Download CSV",
            data      = csv_buf.getvalue().encode(),
            file_name = f"run_{selected}.csv",
            mime      = "text/csv",
        )

        import json
        jsonl_str = "\n".join(json.dumps(r) for r in rows)
        col_jsonl.download_button(
            "⬇ Download JSONL",
            data      = jsonl_str.encode(),
            file_name = f"run_{selected}.jsonl",
            mime      = "application/jsonl",
        )

# ── Run comparison ────────────────────────────────────────────────────────────

if len(runs) > 1:
    st.subheader("Run comparison")
    compare_ids = st.multiselect(
        "Select runs to compare", run_ids, default=run_ids[:min(4, len(run_ids))]
    )
    if len(compare_ids) > 1:
        comp_rows = []
        for rid in compare_ids:
            s = get_run_summary(con, rid)
            if s:
                comp_rows.append({
                    "Run":       rid,
                    "Model":     s.get("model", "—"),
                    "Retrieval": s.get("retrieval", "—"),
                    "Accuracy":  s.get("accuracy"),
                })
        comp_df = pd.DataFrame(comp_rows).set_index("Run")
        comp_df["Accuracy %"] = comp_df["Accuracy"].apply(
            lambda x: round(x * 100, 1) if x is not None else None
        )
        st.bar_chart(comp_df[["Accuracy %"]])
        st.dataframe(comp_df.drop(columns=["Accuracy"]), use_container_width=True)
