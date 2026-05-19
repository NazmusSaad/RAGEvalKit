-- RAGEvalKit DuckDB schema

CREATE TABLE IF NOT EXISTS documents (
    doc_id      TEXT PRIMARY KEY,
    source_path TEXT NOT NULL,
    title       TEXT,
    num_chars   INTEGER,
    ingested_at TIMESTAMP DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id             TEXT PRIMARY KEY,
    doc_id               TEXT REFERENCES documents(doc_id),
    ordinal              INTEGER,
    text                 TEXT NOT NULL,
    num_tokens           INTEGER,
    chunking_config_hash TEXT
);

CREATE TABLE IF NOT EXISTS eval_sets (
    evalset_id   TEXT PRIMARY KEY,
    name         TEXT,
    created_at   TIMESTAMP DEFAULT now(),
    generated_by TEXT,
    config_json  JSON
);

CREATE TABLE IF NOT EXISTS eval_questions (
    question_id      TEXT PRIMARY KEY,
    evalset_id       TEXT REFERENCES eval_sets(evalset_id),
    question         TEXT NOT NULL,
    reference_answer TEXT,
    source_chunk_ids JSON,
    difficulty       TEXT,
    qtype            TEXT
);

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,
    name        TEXT,
    tag         TEXT,
    config_hash TEXT,
    config_json JSON,
    evalset_id  TEXT REFERENCES eval_sets(evalset_id),
    git_sha     TEXT,
    started_at  TIMESTAMP,
    finished_at TIMESTAMP,
    status      TEXT
);

CREATE TABLE IF NOT EXISTS run_items (
    item_id           TEXT PRIMARY KEY,
    run_id            TEXT REFERENCES runs(run_id),
    question_id       TEXT REFERENCES eval_questions(question_id),
    generated_answer  TEXT,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    total_cost_usd    DOUBLE,
    latency_ms        INTEGER,
    model             TEXT,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS retrieved_contexts (
    item_id    TEXT REFERENCES run_items(item_id),
    rank       INTEGER,
    chunk_id   TEXT,
    chunk_text TEXT,
    score      DOUBLE,
    PRIMARY KEY (item_id, rank)
);

CREATE TABLE IF NOT EXISTS metric_scores (
    item_id     TEXT REFERENCES run_items(item_id),
    metric      TEXT,
    score       DOUBLE,
    label       TEXT,
    reason      TEXT,
    judge_model TEXT,
    raw_json    JSON,
    PRIMARY KEY (item_id, metric)
);

CREATE TABLE IF NOT EXISTS claim_evaluations (
    item_id              TEXT REFERENCES run_items(item_id),
    claim_idx            INTEGER,
    claim_text           TEXT,
    verdict              TEXT,
    supporting_chunk_ids JSON,
    rationale            TEXT,
    PRIMARY KEY (item_id, claim_idx)
);

CREATE TABLE IF NOT EXISTS root_causes (
    item_id          TEXT REFERENCES run_items(item_id),
    primary_cause    TEXT,
    secondary_causes JSON,
    suggested_fix    TEXT,
    PRIMARY KEY (item_id)
);
