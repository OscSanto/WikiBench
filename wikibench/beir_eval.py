"""BEIR evaluation adapter for wikibench.

Flow:
  1. `prepare`  — download BEIR dataset + convert corpus to chunks.jsonl
  2. (user runs wikembed on the chunks.jsonl to build the index)
  3. `evaluate` — query wikibench Retriever against BEIR qrels, compute metrics

Recommended datasets for medical RAG:
  nfcorpus   3.6k docs   Biomedical literature — smallest, good smoke test
  bioasq     hundreds    Biomedical QA from PubMed
  scifact    5k docs     Scientific fact verification
  msmarco    8.8M docs   General web — large, skip unless you have time

BEIR nDCG@10 baselines (BM25):
  nfcorpus: 0.325   bioasq: 0.465   scifact: 0.665   msmarco: 0.228
"""
from __future__ import annotations

import csv
import json
import re
import textwrap
from pathlib import Path
from typing import Sequence


# ── Dataset download ──────────────────────────────────────────────────────────

_BEIR_HF = {
    "nfcorpus":    ("BeIR/nfcorpus",     "BeIR/nfcorpus-qrels"),
    "bioasq":      ("BeIR/bioasq",       "BeIR/bioasq-qrels"),
    "scifact":     ("BeIR/scifact",      "BeIR/scifact-qrels"),
    "msmarco":     ("BeIR/msmarco",      "BeIR/msmarco-qrels"),
    "trec-covid":  ("BeIR/trec-covid",   "BeIR/trec-covid-qrels"),
    "arguana":     ("BeIR/arguana",      "BeIR/arguana-qrels"),
    "fiqa":        ("BeIR/fiqa",         "BeIR/fiqa-qrels"),
    "hotpotqa":    ("BeIR/hotpotqa",     "BeIR/hotpotqa-qrels"),
}


def download_dataset(name: str, data_dir: str | Path) -> Path:
    """Download a BEIR dataset from HuggingFace into data_dir.

    Returns the dataset directory containing:
      corpus.jsonl, queries.jsonl, qrels/test.tsv
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install HuggingFace datasets to download BEIR: pip install datasets"
        )

    if name not in _BEIR_HF:
        raise ValueError(
            f"Unknown dataset {name!r}. Known: {list(_BEIR_HF)}"
        )

    corpus_hf, qrels_hf = _BEIR_HF[name]
    out = Path(data_dir) / name
    out.mkdir(parents=True, exist_ok=True)
    qrels_dir = out / "qrels"
    qrels_dir.mkdir(exist_ok=True)

    # Corpus
    corpus_path = out / "corpus.jsonl"
    if not corpus_path.exists():
        print(f"  Downloading corpus from {corpus_hf} …")
        ds = load_dataset(corpus_hf, "corpus", split="corpus")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({
                    "_id":   row["_id"],
                    "title": row.get("title", ""),
                    "text":  row.get("text", ""),
                }) + "\n")
        print(f"  Wrote {corpus_path}")

    # Queries
    queries_path = out / "queries.jsonl"
    if not queries_path.exists():
        print(f"  Downloading queries …")
        ds = load_dataset(corpus_hf, "queries", split="queries")
        with open(queries_path, "w", encoding="utf-8") as f:
            for row in ds:
                f.write(json.dumps({
                    "_id":  row["_id"],
                    "text": row.get("text", ""),
                }) + "\n")
        print(f"  Wrote {queries_path}")

    # Qrels
    qrels_path = qrels_dir / "test.tsv"
    if not qrels_path.exists():
        print(f"  Downloading qrels from {qrels_hf} …")
        ds = load_dataset(qrels_hf, split="test")
        with open(qrels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["query-id", "corpus-id", "score"])
            for row in ds:
                writer.writerow([row["query-id"], row["corpus-id"], row["score"]])
        print(f"  Wrote {qrels_path}")

    return out


# ── Corpus → chunks.jsonl ─────────────────────────────────────────────────────

def corpus_to_chunks(
    corpus_path: str | Path,
    chunks_path: str | Path,
    chunk_size:  int = 500,
    overlap:     int = 50,
) -> int:
    """Convert BEIR corpus.jsonl to wikichunk-compatible chunks.jsonl.

    BEIR docs are flat (no section hierarchy). Each doc becomes one or more
    chunks using a sliding window. Returns number of chunks written.
    """
    def _sliding_window(text: str) -> list[str]:
        words  = text.split()
        chunks = []
        i      = 0
        while i < len(words):
            j = i + 1
            while j < len(words) and len(" ".join(words[i:j + 1])) <= chunk_size:
                j += 1
            chunk = " ".join(words[i:j])
            if chunk.strip():
                chunks.append(chunk)
            if j >= len(words):
                break
            if overlap > 0:
                next_i = j
                for k in range(j - 1, i, -1):
                    if len(" ".join(words[k:j])) >= overlap:
                        next_i = k
                        break
                i = max(next_i, i + 1)
            else:
                i = j
        return chunks

    n = 0
    with open(corpus_path, encoding="utf-8") as src, \
         open(chunks_path, "w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            doc_id = doc["_id"]
            title  = doc.get("title", "").strip()
            text   = doc.get("text",  "").strip()
            full   = f"{title}\n\n{text}" if title else text
            if not full.strip():
                continue
            windows = _sliding_window(full)
            if not windows:
                windows = [full[:chunk_size]]
            for idx, window in enumerate(windows):
                dst.write(json.dumps({
                    "article":     title or doc_id,
                    "path":        doc_id,      # BEIR doc_id — used for qrel matching
                    "section":     "",
                    "depth":       0,
                    "chunk_index": idx,
                    "text":        window,
                    "char_count":  len(window),
                }, ensure_ascii=False) + "\n")
                n += 1

    return n


# ── Load queries + qrels ──────────────────────────────────────────────────────

def load_queries(queries_path: str | Path) -> dict[str, str]:
    """Returns {query_id: query_text}."""
    queries = {}
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            queries[row["_id"]] = row["text"]
    return queries


def load_qrels(qrels_path: str | Path) -> dict[str, dict[str, int]]:
    """Returns {query_id: {doc_id: relevance}}."""
    qrels: dict[str, dict[str, int]] = {}
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            rel = int(row["score"])
            qrels.setdefault(qid, {})[did] = rel
    return qrels


# ── Retrieval + evaluation ────────────────────────────────────────────────────

def _chunks_to_doc_scores(chunks: list[dict]) -> dict[str, float]:
    """Aggregate per-chunk results to per-document scores (max pooling)."""
    doc_scores: dict[str, float] = {}
    for rank, chunk in enumerate(chunks, start=1):
        doc_id = chunk.get("path", "")
        score  = 1.0 / rank          # reciprocal rank as score for pytrec_eval
        if doc_id not in doc_scores or score > doc_scores[doc_id]:
            doc_scores[doc_id] = score
    return doc_scores


def evaluate(
    index_dir:       str | Path,
    dataset_dir:     str | Path,
    retrieval:       str           = "dense",
    embedding_model: str           = "BAAI/bge-small-en-v1.5",
    top_k:           int           = 100,
    k_values:        list[int]     = (1, 3, 5, 10, 100),
    split:           str           = "test",
) -> dict:
    """Run BEIR evaluation.

    Returns {metric: {k: score}} dict, e.g.:
      {"ndcg": {10: 0.412}, "recall": {100: 0.785}, "map": {10: 0.198}}
    """
    try:
        import pytrec_eval
    except ImportError:
        raise ImportError(
            "Install pytrec_eval for BEIR scoring: pip install pytrec-eval-terrier"
        )

    from .retrieval import Retriever

    dataset_dir = Path(dataset_dir)
    queries     = load_queries(dataset_dir / "queries.jsonl")
    qrels_path  = dataset_dir / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        qrels_path = dataset_dir / "qrels" / "test.tsv"
    qrels = load_qrels(qrels_path)

    # Only evaluate queries that have qrels
    eval_qids = [qid for qid in queries if qid in qrels]
    print(f"  {len(eval_qids)} queries with qrels / {len(queries)} total")

    retriever = Retriever(
        index_dir       = index_dir,
        embedding_model = embedding_model,
        top_k           = top_k,
    )

    # Build run dict: {query_id: {doc_id: score}}
    run: dict[str, dict[str, float]] = {}
    for i, qid in enumerate(eval_qids):
        if i % 50 == 0:
            print(f"  Querying {i}/{len(eval_qids)} …", end="\r", flush=True)
        chunks         = retriever.retrieve(retrieval, queries[qid])
        run[qid]       = _chunks_to_doc_scores(chunks)
    print(f"  Queried {len(eval_qids)}/{len(eval_qids)}")

    retriever.close()

    # pytrec_eval measures
    measures = set()
    for k in k_values:
        measures.add(f"ndcg_cut_{k}")
        measures.add(f"recall_{k}")
        measures.add(f"map_cut_{k}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query  = evaluator.evaluate(run)

    # Average across queries
    totals: dict[str, dict[int, float]] = {
        "ndcg": {}, "recall": {}, "map": {}
    }
    counts: dict[str, dict[int, int]] = {
        "ndcg": {}, "recall": {}, "map": {}
    }
    for _, scores in per_query.items():
        for k in k_values:
            for metric, prefix in [("ndcg", "ndcg_cut"), ("recall", "recall"), ("map", "map_cut")]:
                key = f"{prefix}_{k}"
                if key in scores:
                    totals[metric][k] = totals[metric].get(k, 0.0) + scores[key]
                    counts[metric][k] = counts[metric].get(k, 0) + 1

    averages: dict[str, dict[int, float]] = {}
    for metric in totals:
        averages[metric] = {
            k: totals[metric][k] / counts[metric][k]
            for k in totals[metric]
            if counts[metric].get(k, 0) > 0
        }

    return averages


def print_results(dataset: str, retrieval: str, results: dict, k_values: list[int]) -> None:
    bar = "─" * 52
    print(f"\n{bar}")
    print(f"  BEIR — {dataset}   retrieval={retrieval}")
    print(bar)
    header = f"  {'Metric':<12}" + "".join(f"  @{k:<6}" for k in k_values)
    print(header)
    print(f"  {'─'*12}" + "".join(f"  {'─'*7}" for _ in k_values))
    for metric in ("ndcg", "map", "recall"):
        row = f"  {metric.upper():<12}"
        for k in k_values:
            val = results.get(metric, {}).get(k)
            row += f"  {val:.4f} " if val is not None else "  —      "
        print(row)
    print(bar)
