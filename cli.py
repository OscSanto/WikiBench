"""wikibench CLI — run benchmarks, export results."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _cmd_run(args: argparse.Namespace) -> None:
    from wikibench.pipeline import BenchmarkRun, load_config
    from wikibench import db

    cfg = load_config(args.config)
    run = BenchmarkRun(
        config     = cfg,
        db_path    = args.db,
        ollama_url = args.ollama_url,
        warmup     = not args.no_warmup,
        top_k      = args.top_k,
    )
    summary = run.run()
    print(f"\nAccuracy: {summary['accuracy']:.2%}  ({summary['n_correct']}/{summary['n_total']})")


def _cmd_export(args: argparse.Namespace) -> None:
    import sqlite3
    from wikibench.export import export_run
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    paths = export_run(con, args.run_id, args.out_dir, fmt=args.fmt)
    con.close()
    for k, p in paths.items():
        print(f"{k}: {p}")


def _cmd_beir(args: argparse.Namespace) -> None:
    from wikibench.beir_eval import (
        download_dataset, corpus_to_chunks, evaluate, print_results
    )
    import subprocess, sys

    data_dir  = Path(args.data_dir)
    index_dir = Path(args.index_dir) if args.index_dir else data_dir / args.dataset / "index"

    # Step 1 — download dataset
    if not args.skip_download:
        print(f"\nStep 1 — Downloading BEIR/{args.dataset} …")
        dataset_dir = download_dataset(args.dataset, data_dir)
    else:
        dataset_dir = data_dir / args.dataset

    # Step 2 — convert corpus to chunks.jsonl
    chunks_path = dataset_dir / "chunks.jsonl"
    if not chunks_path.exists():
        print(f"\nStep 2 — Converting corpus to chunks.jsonl …")
        n = corpus_to_chunks(
            dataset_dir / "corpus.jsonl",
            chunks_path,
            chunk_size = args.chunk_size,
            overlap    = args.overlap,
        )
        print(f"  {n:,} chunks written to {chunks_path}")
    else:
        print(f"\nStep 2 — chunks.jsonl already exists, skipping conversion")

    # Step 3 — embed with wikembed (unless already done or skipped)
    done_path = index_dir / ".done"
    if done_path.exists():
        print(f"\nStep 3 — Index already built at {index_dir}")
    elif args.skip_embed:
        print(f"\nStep 3 — Skipping embedding (--skip-embed). Index must exist at {index_dir}")
    else:
        print(f"\nStep 3 — Embedding with wikembed …")
        result = subprocess.run(
            [sys.executable, "-m", "wikembed.cli", str(chunks_path),
             "--model", args.embedding_model, "--out", str(index_dir)],
            check=False,
        )
        if result.returncode != 0:
            print(f"\nwikembed failed. Run manually:")
            print(f"  wikembed {chunks_path} --model {args.embedding_model} --out {index_dir}")
            raise SystemExit(1)

    # Step 4 — evaluate
    k_values = [int(k) for k in args.k_values.split(",")]
    print(f"\nStep 4 — Evaluating retrieval={args.retrieval} …")
    results = evaluate(
        index_dir       = index_dir,
        dataset_dir     = dataset_dir,
        retrieval       = args.retrieval,
        embedding_model = args.embedding_model,
        top_k           = max(k_values),
        k_values        = k_values,
    )
    print_results(args.dataset, args.retrieval, results, k_values)


def _cmd_list(args: argparse.Namespace) -> None:
    from wikibench import db
    import sqlite3
    con = sqlite3.connect(str(args.db))
    con.row_factory = sqlite3.Row
    runs = db.list_runs(con)
    con.close()
    if not runs:
        print("No runs in database.")
        return
    header = f"{'ID':>4}  {'Model':<30}  {'Retrieval':<12}  {'N':>6}  {'Acc':>6}  {'Completed'}"
    print(header)
    print("-" * len(header))
    for r in runs:
        acc = f"{r['accuracy']:.1%}" if r["accuracy"] is not None else "—"
        comp = r["completed_at"] or "incomplete"
        print(f"{r['id']:>4}  {r['model']:<30}  {r['retrieval']:<12}  "
              f"{(r['n_questions'] or 0):>6}  {acc:>6}  {comp}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wikibench",
        description="Retrieval benchmark suite for wikembed indexes.",
    )
    parser.add_argument("--db", default="wikibench.db", metavar="PATH",
                        help="SQLite database file (default: wikibench.db)")

    sub = parser.add_subparsers(dest="command", required=True)

    # run
    p_run = sub.add_parser("run", help="Execute a benchmark run")
    p_run.add_argument("config", help="Path to YAML config file")
    p_run.add_argument("--ollama-url", default="http://localhost:11434",
                       help="Ollama base URL (default: %(default)s)")
    p_run.add_argument("--no-warmup", action="store_true",
                       help="Skip model warmup query")
    p_run.add_argument("--top-k", type=int, default=5,
                       help="Number of chunks to retrieve (default: %(default)s)")
    p_run.set_defaults(func=_cmd_run)

    # export
    p_exp = sub.add_parser("export", help="Export a run to CSV/JSONL")
    p_exp.add_argument("run_id", type=int, help="Run ID to export")
    p_exp.add_argument("--out-dir", default=".", metavar="DIR")
    p_exp.add_argument("--fmt", choices=["csv", "jsonl", "both"], default="both")
    p_exp.set_defaults(func=_cmd_export)

    # list
    p_list = sub.add_parser("list", help="List all runs")
    p_list.set_defaults(func=_cmd_list)

    # beir
    p_beir = sub.add_parser("beir", help="Run BEIR retrieval evaluation")
    p_beir.add_argument("dataset",
                        choices=["nfcorpus", "bioasq", "scifact", "msmarco",
                                 "trec-covid", "arguana", "fiqa", "hotpotqa"],
                        help="BEIR dataset name")
    p_beir.add_argument("--retrieval", default="dense",
                        choices=["dense", "sparse", "hybrid_rrf", "structured"],
                        help="Retrieval method (default: dense)")
    p_beir.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5",
                        metavar="MODEL",
                        help="Embedding model — must match wikembed (default: %(default)s)")
    p_beir.add_argument("--data-dir", default="./beir_data", metavar="DIR",
                        help="Directory to download datasets into (default: %(default)s)")
    p_beir.add_argument("--index-dir", default=None, metavar="DIR",
                        help="wikembed index directory (default: <data-dir>/<dataset>/index)")
    p_beir.add_argument("--chunk-size", type=int, default=500,
                        help="Chunk size in chars for corpus conversion (default: %(default)s)")
    p_beir.add_argument("--overlap", type=int, default=50,
                        help="Chunk overlap in chars (default: %(default)s)")
    p_beir.add_argument("--k-values", default="1,3,5,10,100",
                        help="Comma-separated k values for nDCG/Recall/MAP (default: %(default)s)")
    p_beir.add_argument("--skip-download", action="store_true",
                        help="Skip dataset download (assume already in --data-dir)")
    p_beir.add_argument("--skip-embed", action="store_true",
                        help="Skip wikembed step (assume index already exists)")
    p_beir.set_defaults(func=_cmd_beir)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
