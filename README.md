# wikibench

Retrieval benchmark suite for wikembed indexes. Two evaluation modes:

- **MedQA mode** — end-to-end pipeline accuracy on your ZIM-based knowledge base using a local Ollama model
- **BEIR mode** — retrieval-only quality on standardised corpora with pre-labelled ground truth

```
wikichunk → wikembed → wikibench
   ZIM         FAISS       Benchmarks
```

---

## Installation

### 1. Python environment

```bash
conda create -n wikibench python=3.11 -y
conda activate wikibench
```

Or with venv:

```bash
python3.11 -m venv .venv && source .venv/bin/activate
```

### 2. Install wikibench

```bash
cd ~/Documents/GitHub/wikibench
pip install -e .
```

### 3. Install wikembed (required for BEIR mode)

BEIR mode calls wikembed to embed the benchmark corpus. If you only run MedQA mode, this is not needed.

```bash
cd ~/Documents/GitHub/wikembed
pip install -e .
```

### 4. Install optional extras

```bash
# Streamlit dashboard
pip install streamlit

# BEIR evaluation
pip install -e ".[beir]"   # from wikibench directory
# installs: datasets>=2.19, pytrec-eval-terrier>=0.5
```

### 5. Install and start Ollama (MedQA mode only)

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b
ollama list   # confirm model is ready
```

### 6. Verify installation

```bash
wikibench --help
```

---

## End-to-end pipeline

There are two paths depending on whether you are using Google Colab for indexing or running everything locally.

### Path A — Colab indexing + Pi benchmarking (recommended for large ZIMs)

Run wikichunk + wikembed on Colab (GPU, free tier). Transfer the index to your Pi. Run wikibench on the Pi.

```bash
# ── On Colab ────────────────────────────────────────────────────────────────
# Open wikembed/COLAB.ipynb
# Cell 3: paste your Kiwix ZIM URL
# Cell 4: configure strategy, model, FAISS type → click ▶ Run Pipeline
# Cell 4: click ⬇ Download Index → saves wikembed_output.zip

# ── Transfer to Pi ───────────────────────────────────────────────────────────
scp wikembed_output.zip pi@raspberrypi.local:~/
ssh pi@raspberrypi.local
unzip ~/wikembed_output.zip -d ~/wikembed_output

# ── On Pi ────────────────────────────────────────────────────────────────────
# Install Ollama and wikibench (see Installation above)

# Write a config
cat > ~/Documents/GitHub/wikibench/configs/my_run.yaml << 'EOF'
model:           "llama3.2:3b"
embedding_model: "BAAI/bge-small-en-v1.5"
retrieval:       "hybrid_rrf"
index_dir:       "~/wikembed_output/wikipedia_en_medicine/section_500_50"
dataset_path:    "~/medqa/test_us.jsonl"
n_questions:     100
top_k:           5
EOF

# Run
cd ~/Documents/GitHub/wikibench
wikibench run configs/my_run.yaml

# View
wikibench list
streamlit run app.py
```

---

### Path B — Fully local (no Colab)

Run the entire pipeline on one machine. Suitable for smaller ZIMs or if you have a capable local machine.

```bash
# ── Step 1: Chunk the ZIM ────────────────────────────────────────────────────
cd ~/Documents/GitHub/wikichunk
wikichunk /path/to/wikipedia_en_medicine.zim \
    --strategy section \
    --chunk-size 500 \
    --overlap 50 \
    --out ~/wikembed_output

# Output: ~/wikembed_output/wikipedia_en_medicine/section_500_50/chunks/chunks.jsonl

# ── Step 2: Embed into FAISS index ───────────────────────────────────────────
cd ~/Documents/GitHub/wikembed
wikembed ~/wikembed_output/wikipedia_en_medicine/section_500_50/chunks/chunks.jsonl \
    --model BAAI/bge-small-en-v1.5

# Output: ~/wikembed_output/wikipedia_en_medicine/section_500_50/faiss.index
#         ~/wikembed_output/wikipedia_en_medicine/section_500_50/chunks.db

# ── Step 3: Run benchmark ────────────────────────────────────────────────────
cd ~/Documents/GitHub/wikibench
wikibench run configs/my_run.yaml
```

---

## Two evaluation modes

### Mode 1 — MedQA (end-to-end, ZIM-based)

For each question in the dataset:

1. Retrieve top-K chunks from your wikembed index using the configured method
2. Build a prompt containing the retrieved context and send to Ollama (streaming)
3. Parse the predicted answer letter from the model's response
4. Compare against the known correct answer and record the result

**What changes the score:** Chunking quality, embedding model, retrieval method, prompt design, and model capability all affect accuracy. A single number cannot tell you which component is responsible for a given result.

**Ground truth:** Answer letters (`A`–`E`) provided by the MedQA dataset, written by medical professionals.

**Answer parsing:** The response is parsed with a three-tier cascade — standalone letter on line 1 → `A.` / `A)` prefix → first letter found anywhere in the response. If no valid letter is found, the answer is recorded as `NULL` and counted as incorrect.

---

### Mode 2 — BEIR (retrieval-only, standardised corpus)

Downloads a pre-built benchmark corpus, embeds it with wikembed, queries it with wikibench's retriever, and scores against pre-labelled relevance judgements.

**What changes the score:** Embedding model and retrieval method only. LLM, prompt design, and ZIM chunking have no effect.

**Ground truth:** Human-labelled relevance judgements (qrels) provided by the BEIR benchmark — a mapping of which documents are relevant for each query, scored 0–3.

**Why the corpus is separate from your ZIM:** BEIR requires pre-labelled relevance pairs. Your ZIM has no such labels. BEIR provides its own labelled corpus as the measurement vehicle.

---

### Mode comparison

| | MedQA (ZIM) | BEIR |
|---|---|---|
| Corpus | Your ZIM-derived index | BEIR's own documents |
| Measures | Full pipeline accuracy | Retrieval quality only |
| LLM required | Yes (Ollama) | No |
| Ground truth | Correct answer letter | Pre-labelled doc relevance |
| Primary metric | Accuracy % | nDCG@10 |
| What changes the score | Chunking + embedding + retrieval + prompt + model | Embedding model + retrieval method only |

Use **BEIR** to tune and validate your retriever in isolation. Use **MedQA** to validate the full pipeline on your actual deployment corpus.

---

## Metrics defined

### MedQA metrics

**`accuracy`**

```
accuracy = n_correct / n_total
```

Where `n_correct` is the count of questions where the predicted answer letter exactly matches the correct letter. Questions where parsing returns `NULL` are counted as incorrect. Reported as a percentage.

**`retrieval_latency_ms`**

Time in milliseconds from the start of the retrieval call to the last chunk returned. Measured client-side. Includes FAISS search, FTS5 query, and chunk fetch from SQLite. Does not include embedding the query.

**`n_chunks_retrieved`**

Number of chunks actually returned by the retriever. Normally equals `top_k` but may be lower if fewer matching documents exist (common with `sparse` on rare queries).

**`ttft_ms`** — Time to First Token

Time in milliseconds from sending the HTTP POST to Ollama until the first non-empty content token arrives in the streaming response. Measured client-side. Includes network round-trip, prompt evaluation (prefill), and any model loading time if a cold start occurred.

**`tokens_per_sec`** — Generation throughput

```
tokens_per_sec = eval_count / (eval_duration_ns / 1_000_000_000)
```

`eval_count` is the number of tokens generated (from Ollama). `eval_duration_ns` is the total generation time in nanoseconds (from Ollama). Measures how fast the model generates output tokens once it has started.

**`prefill_rate`** — Prompt processing throughput

```
prefill_rate = prompt_eval_count / (prompt_eval_duration_ns / 1_000_000_000)
```

`prompt_eval_count` is the number of prompt tokens processed. Measures how fast the model processes the input context. Lower prefill rate means the context window is large relative to the model's capacity.

**`total_duration_ms`** — Full round-trip

Total time from the Ollama API call to the final response token, in milliseconds. Includes model load, prefill, and generation. Reported by Ollama.

**`load_duration_ms`** — Model load time

Time spent loading model weights into RAM (or confirming they are already loaded). Reported by Ollama. High on first query after Ollama starts; near-zero when the model is already in memory.

**`cold_start`**

Boolean. `True` if `load_duration_ns > 2_000_000_000` (2 seconds), indicating the model was not already loaded in memory at the time of the request. Cold starts inflate `ttft_ms` and `total_duration_ms` and should be excluded when comparing generation speed across runs.

**`ram_before_mb` / `ram_after_mb`**

System RAM in use (used bytes / 1024²) immediately before and after the Ollama streaming call. Measured via `psutil`. The delta (`ram_after - ram_before`) approximates the memory cost of one inference pass.

**`cpu_pct`**

Mean CPU utilisation across the duration of the Ollama streaming call, sampled every 0.5 seconds by a background thread. Reported as a percentage (0–100 per core, can exceed 100 on multi-core systems).

**`pi_temp_c`**

Raspberry Pi CPU temperature in degrees Celsius, read via `vcgencmd measure_temp` after each LLM call. `NULL` on non-Pi hardware where `vcgencmd` is not available.

---

### BEIR metrics

**nDCG@k** — Normalised Discounted Cumulative Gain

Measures ranking quality. Documents placed higher in the result list contribute more to the score; the contribution is discounted logarithmically by rank position. Normalised against the ideal ranking so that 1.0 is perfect and 0.0 is worst.

```
DCG@k  = Σ (relevance_i / log2(rank_i + 1))   for i in 1..k
nDCG@k = DCG@k / IDCG@k
```

The primary BEIR metric is **nDCG@10** — quality of the top 10 results. This is the number to compare against published baselines.

**MAP@k** — Mean Average Precision

Measures both precision and recall jointly. For each query, computes the average precision at each rank position where a relevant document appears, then averages across all queries. Sensitive to whether all relevant documents are retrieved, not just the top ones.

**Recall@k**

Fraction of all known relevant documents that appear in the top-k results:

```
Recall@k = |relevant ∩ top_k| / |relevant|
```

Recall@100 is commonly reported — it tells you how much of the relevant set the retriever can find if given 100 results. Useful for RAG pipelines where a reranker may follow the retriever.

---

## BEIR datasets

| Dataset | Domain | Corpus size | BM25 nDCG@10 | Start here |
|---------|--------|-------------|---------------|-----------|
| `nfcorpus` | Biomedical literature | 3.6k docs | 0.325 | ✓ smallest |
| `bioasq` | Biomedical QA | ~1M docs | 0.465 | |
| `scifact` | Scientific claims | 5k docs | 0.665 | |
| `trec-covid` | COVID-19 papers | 171k docs | 0.656 | |
| `fiqa` | Financial QA | 57k docs | 0.236 | |
| `msmarco` | Web passages | 8.8M docs | 0.228 | |

BM25 baseline = plain keyword search with no neural embedding. Beating it with `dense` or `hybrid_rrf` confirms your embedding model is adding value.

### Running BEIR

```bash
# Full run: downloads dataset, builds index, evaluates
wikibench beir nfcorpus --retrieval hybrid_rrf

# Compare all retrieval methods (reuse existing index)
wikibench beir nfcorpus --retrieval dense        --skip-download --skip-embed
wikibench beir nfcorpus --retrieval sparse       --skip-download --skip-embed
wikibench beir nfcorpus --retrieval hybrid_rrf   --skip-download --skip-embed

# Different dataset
wikibench beir scifact --retrieval hybrid_rrf
```

### Example BEIR output

```
────────────────────────────────────────────────────
  BEIR — nfcorpus   retrieval=hybrid_rrf
────────────────────────────────────────────────────
  Metric        @1       @3       @5       @10      @100
  ────────────  ───────  ───────  ───────  ───────  ───────
  NDCG          0.5231   0.4512   0.4108   0.3741   —
  MAP           0.0412   0.0743   0.0891   0.1023   0.1544
  RECALL        0.0312   0.0812   0.1124   0.1743   0.4821
────────────────────────────────────────────────────
```

---

## Retrieval methods

| Method | How it works | BEIR | ZIM / MedQA |
|--------|-------------|------|-------------|
| `dense` | FAISS nearest-neighbour on L2-normalised vectors (inner product = cosine similarity) | ✓ | ✓ |
| `sparse` | BM25 via SQLite FTS5 on chunk text | ✓ (scores below BM25 baseline — see Considerations) | ✓ |
| `hybrid_rrf` | Dense + sparse lists fused with Reciprocal Rank Fusion (k=60). Score = Σ 1/(k + rank) across both lists | ✓ | ✓ recommended |
| `structured` | Dense retrieval re-ranked by Wikipedia section depth: lead (depth 0) first, then infobox, then all other | not meaningful | ✓ ZIM only |

---

## Considerations

### Embedding model must match

The `embedding_model` in your config must be the same model wikembed used to build the index. Different models produce incompatible vector spaces even at the same dimension. Check which model was used:

```bash
cat /path/to/index/.done   # contains model name in the completion JSON
```

### Small models and RAG context

Research shows models below 7B parameters can be hurt by retrieved context — the model gets distracted by the retrieved passages and answers incorrectly on questions it would have answered correctly without context. This affects MedQA scores but has no bearing on BEIR (no LLM involved).

To diagnose: run the same config with `top_k: 0` (no retrieval). If that scores higher, context distraction is the problem. Try reducing `top_k` to 2–3 or shortening chunk size.

### Sparse retrieval vs published BM25 baselines

Published BEIR BM25 scores use a tuned full-document BM25 (rank-bm25 or Elasticsearch). wikibench `sparse` uses FTS5 BM25 over chunks. Expect wikibench sparse to score below the published baseline — this is expected, not a bug.

### `structured` on BEIR

BEIR corpora are flat documents with no section hierarchy. All chunks will have `depth=0`. `structured` behaves identically to `dense` on BEIR. Run it on ZIM-derived indexes only.

### BEIR results do not predict MedQA results

A retriever that scores well on NFCorpus does not guarantee good MedQA accuracy. BEIR uses flat text with no Wikipedia section structure. Your ZIM chunks have section depth, infobox content, and medical Wikipedia vocabulary. The two corpora are different enough that a model could rank well on one and poorly on the other.

BEIR tells you: is my retriever fundamentally sound?
MedQA tells you: does the full pipeline work on my actual deployment corpus?

---

## MedQA dataset format

Expects JSONL where each line is:

```json
{
  "question": "A 45-year-old presents with...",
  "options": {"A": "Pneumonia", "B": "Bronchitis", "C": "...", "D": "...", "E": "..."},
  "answer": "A"
}
```

Both 4-option and 5-option US MedQA formats are supported. The `answer` field can be a letter (`"A"`) or full option text (matched by exact string comparison).

---

## Config file reference

```yaml
model:            "llama3.2:3b"           # Ollama model name (required)
embedding_model:  "BAAI/bge-small-en-v1.5" # must match wikembed index (required)
retrieval:        "hybrid_rrf"             # dense|sparse|hybrid_rrf|structured (required)
index_dir:        "/path/to/index"         # wikembed output dir (required)
dataset_path:     "/path/to/test_us.jsonl" # MedQA JSONL (required)
n_questions:      100                      # omit or null for full dataset
top_k:            5                        # chunks retrieved per question
```

---

## CLI reference

```
wikibench [--db PATH] <command>

Global:
  --db PATH     SQLite results database (default: wikibench.db)

Commands:
  run <config>              Run a MedQA benchmark
    --ollama-url URL        Ollama base URL (default: http://localhost:11434)
    --no-warmup             Skip model warmup query
    --top-k INT             Chunks to retrieve (default: 5)

  beir <dataset>            Run BEIR retrieval evaluation
    --retrieval METHOD      dense|sparse|hybrid_rrf|structured (default: dense)
    --embedding-model ID    Must match wikembed index model
    --data-dir DIR          Dataset download directory (default: ./beir_data)
    --index-dir DIR         wikembed index dir (default: <data-dir>/<dataset>/index)
    --chunk-size INT        Chars per chunk (default: 500)
    --overlap INT           Chunk overlap in chars (default: 50)
    --k-values LIST         Comma-separated k values (default: 1,3,5,10,100)
    --skip-download         Skip corpus download
    --skip-embed            Skip wikembed indexing step

  list                      List all MedQA runs

  export <run_id>           Export a run to CSV/JSONL
    --out-dir DIR           Output directory (default: .)
    --fmt {csv,jsonl,both}  Export format (default: both)
```

---

## Output files

```
wikibench.db          ← SQLite: runs, questions, results, llm_metrics tables
exports/
├── run_1.csv         ← full join: questions + results + metrics, one row per question
└── run_1.jsonl       ← same data, one JSON object per line
beir_data/
└── nfcorpus/
    ├── corpus.jsonl  ← BEIR corpus
    ├── queries.jsonl ← BEIR queries
    ├── qrels/
    │   └── test.tsv  ← relevance judgements
    ├── chunks.jsonl  ← converted corpus (wikichunk format, input to wikembed)
    └── index/
        ├── faiss.index
        ├── chunks.db
        └── .done
```

---

## Raspberry Pi setup

wikibench is designed for a **Raspberry Pi 5 (8 GB)** with a local Ollama instance. No cloud API required.

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Storage | 8 GB free | 16 GB+ (index + results DB) |
| OS | Raspberry Pi OS 64-bit Bookworm | Same |

### Models

| Model | RAM | Notes |
|-------|-----|-------|
| `llama3.2:3b` | ~2.5 GB | Recommended starting point |
| `llama3.2:8b` | ~6 GB | Test RAM headroom before committing |
| `phi3:mini` | ~2 GB | Fast, smaller context window |
| `gemma2:2b` | ~2 GB | Good balance |

For long runs, use `tmux` to survive SSH disconnects:

```bash
tmux new -s bench
wikibench run configs/my_run.yaml
# Ctrl+B then D to detach; tmux attach -t bench to re-attach
```

Temperature is recorded per question automatically. Monitor in real time:

```bash
watch -n 5 vcgencmd measure_temp
```

If temperature consistently exceeds 80 °C, add a heatsink or fan.

---

## Dependencies

| Package | Purpose | Install |
|---------|---------|---------|
| `fastembed` | Query embedding | Core |
| `faiss-cpu` | Dense vector search | Core |
| `requests` | Ollama streaming API | Core |
| `psutil` | RAM and CPU monitoring | Core |
| `pyyaml` | Config parsing | Core |
| `numpy` | Vector operations | Core |
| `streamlit` | Dashboard UI | `pip install streamlit` |
| `datasets` | BEIR corpus download | `pip install -e ".[beir]"` |
| `pytrec-eval-terrier` | nDCG/MAP/Recall scoring | `pip install -e ".[beir]"` |
