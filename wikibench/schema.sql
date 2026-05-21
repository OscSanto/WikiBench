-- wikibench database schema
-- One run = one model + one retrieval method + one dataset config.
-- Results are written per-question so a crashed run is resumable.

CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    model           TEXT    NOT NULL,
    embedding_model TEXT    NOT NULL,
    retrieval       TEXT    NOT NULL,
    index_dir       TEXT    NOT NULL,
    dataset_path    TEXT    NOT NULL,
    n_questions     INTEGER,                -- NULL = full dataset
    accuracy        REAL,                   -- filled on completion
    started_at      TEXT    NOT NULL,
    completed_at    TEXT,
    config_hash     TEXT    NOT NULL,       -- sha256 of config YAML for resume detection
    config_json     TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS questions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          INTEGER NOT NULL REFERENCES runs(id),
    question_idx    INTEGER NOT NULL,       -- position in dataset (0-based)
    question_text   TEXT    NOT NULL,
    options_json    TEXT    NOT NULL,       -- {"A": "...", "B": "...", ...}
    correct_answer  TEXT    NOT NULL,       -- "A"–"E"
    UNIQUE(run_id, question_idx)
);

CREATE TABLE IF NOT EXISTS results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id         INTEGER NOT NULL REFERENCES questions(id),
    run_id              INTEGER NOT NULL REFERENCES runs(id),
    predicted_answer    TEXT,               -- "A"–"E", or NULL if parse failed
    is_correct          INTEGER NOT NULL DEFAULT 0,
    context_json        TEXT,               -- JSON array of retrieved chunk dicts
    n_chunks_retrieved  INTEGER,
    retrieval_latency_ms REAL,
    raw_response        TEXT
);

CREATE TABLE IF NOT EXISTS llm_metrics (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id               INTEGER NOT NULL REFERENCES results(id),
    run_id                  INTEGER NOT NULL REFERENCES runs(id),
    -- Ollama timing (all _ms fields converted from Ollama's nanoseconds)
    total_duration_ms       INTEGER,
    load_duration_ms        INTEGER,
    prompt_eval_count       INTEGER,        -- tokens in prompt (prefill tokens)
    prompt_eval_duration_ms INTEGER,
    eval_count              INTEGER,        -- tokens generated
    eval_duration_ms        INTEGER,
    -- Derived
    ttft_ms                 REAL,           -- measured client-side via streaming
    tokens_per_sec          REAL,           -- eval_count / eval_duration_s
    prefill_rate            REAL,           -- prompt_eval_count / prompt_eval_duration_s
    cold_start              INTEGER,        -- 1 if load_duration > 2s
    -- System
    ram_before_mb           REAL,
    ram_after_mb            REAL,
    cpu_pct                 REAL,
    pi_temp_c               REAL            -- NULL on non-Pi hardware
);

CREATE INDEX IF NOT EXISTS idx_results_run   ON results(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_run   ON llm_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_questions_run ON questions(run_id);
