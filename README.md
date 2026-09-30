# Enterprise RAG & Reconciliation Agent

An agentic RAG service for sales data spread across five systems that don't agree with each other. It uses FastAPI with PostgreSQL + pgvector. It ingests every feed, detects **schema drift** as feeds change over time, normalizes everything onto one canonical schema, and reconciles **cross-source discrepancies**. It then answers questions through a Claude tool-use agent whose answers must cite the evidence they rely on.

An **Apache Airflow** pipeline rebuilds the data and index, then runs an **LLM-as-a-judge** benchmark. The run fails if grounding accuracy or answer accuracy drops below its threshold.

```mermaid
flowchart LR
    subgraph Sources
        CRM[CRM csv] & ERP[ERP json] & WH[Warehouse csv ×3]
    end
    Sources --> P[Profile + map columns<br/>heuristics → LLM resolver]
    P --> D{Diff vs last<br/>schema version}
    D -->|drift events| DB[(Postgres)]
    P --> N[Normalize] --> DB
    DB --> R[Reconcile<br/>match + tolerances] --> DB
    DB --> C[Chunk + embed] --> V[(pgvector HNSW<br/>+ tsvector GIN)]
    Q[/POST /query/] --> A[Claude agent<br/>tool-use loop]
    A <--> T[search · reconcile_order · get_discrepancies<br/>get_schema_drift · query_sales]
    T <--> V & DB
    A --> G{Grounding gate<br/>citations + numbers}
    G -->|pass| Ans[Cited answer]
    G -->|fail twice| Abs[Abstain]
    subgraph Airflow
        E[golden set → agent → Claude judge → metrics → quality gate]
    end
    E -.-> Q
```

## What it does

**Distributed, drifting sources.** A seeded generator writes the same orders into five systems. Each system has its own column names, ID formats (`SO-000123`, `123`, `SO000123`), money units (dollars vs. integer cents), date formats (ISO, US, EU), region names and status vocabularies. Four feeds change schema between their first and second batch:

| Source | Drift in batch 2 |
|---|---|
| `crm` | `customer_id` → `cust_id`; `order_date` ISO → `MM/DD/YYYY`; new `sales_rep` column |
| `erp` | `amountCents` (integer cents) → `amount` (decimal dollars) |
| `wh_emea` | `LINE_VALUE` → `NET_VALUE`; new `VAT_AMOUNT` column |
| `wh_apac` | `UNIT_PRICE` dropped |
| `wh_na` | none |

Discrepancies are injected at known rates, and everything injected is written to `ground_truth.json`: amount, quantity, currency and status mismatches, missing records, and duplicates.

**Schema drift detection** ([`ingest/schema_drift.py`](src/recon_rag/ingest/schema_drift.py)). Every column is profiled: is it an ID, a date and in which format, money and in which unit, a status or region vocabulary, a currency code? Each column is then scored against every canonical field on name similarity and on whether its values look right for that field. Confident matches are accepted automatically. Ambiguous ones go to a Claude resolver with structured output, or are held for review, never guessed. Consecutive schema versions are diffed into typed drift events: renamed, added, removed, format changed, unit changed.

**Reconciliation** ([`reconcile/engine.py`](src/recon_rag/reconcile/engine.py)). Records are matched on normalized order ID. Records whose ID matches nothing fall back to matching on customer + date + amount. The CRM is the reference, and every other source is compared to it with configurable tolerances. A quantity mismatch isn't also counted as an amount mismatch.

**Hybrid retrieval** ([`index/retriever.py`](src/recon_rag/index/retriever.py)). Indexed chunks are one per order (all source records side by side plus the reconciliation result), one per discrepancy, one per drift event and one per schema version. They're embedded locally with `bge-small-en-v1.5` via fastembed, so embedding needs no API key. Retrieval fuses pgvector cosine search, length-normalized Postgres full-text search and exact business-key lookup with Reciprocal Rank Fusion.

**The agent** ([`agent/`](src/recon_rag/agent)). It runs a Claude tool-use loop with five tools. `query_sales` is a *whitelisted* aggregate builder: the model picks a metric, grouping and filters from enums and never writes SQL. The final message is structured JSON (`answer`, `citations`, `abstained`). Before it is returned, a deterministic **grounding gate** checks three things:
- every cited ID was actually returned by a tool in this session;
- every number in the answer appears in the *cited* evidence (so no arithmetic the evidence doesn't show);
- every order, customer or SKU ID in the answer appears in the cited evidence.

A failing answer gets one revision. If it fails again, the agent abstains.

**Evaluation** ([`eval/`](src/recon_rag/eval)). A 150-case golden set is built from ground truth in six categories:

| Category | Cases |
|---|---|
| Order lookups | 30 |
| Discrepancy questions | 50 |
| Aggregates | ~23 |
| Schema drift | 12 |
| Multi-hop | 15 |
| Unanswerable (should abstain) | 20 |

Claude grades each answer as a judge. It splits the answer into atomic claims, marks each one supported or not against the retrieved evidence, and checks correctness against the reference answer. Metrics:

| Metric | Definition |
|---|---|
| **Grounding accuracy** | Answered cases where the judge found every claim supported *and* the deterministic gate passed |
| Hallucination rate | 1 − grounding accuracy |
| Answer accuracy | All cases; abstaining on an unanswerable question counts as correct |
| Abstention precision / recall | Against the cases that should be abstained on |
| Judge ↔ validator agreement | How often the LLM judge and the deterministic check agree |

Grounding alone can be gamed: an agent that just quotes evidence is perfectly grounded and useless. So the quality gate requires **both** grounding ≥ 95% **and** answer accuracy ≥ 80% (both configurable).

**Airflow** ([`airflow/dags/rag_eval_pipeline.py`](airflow/dags/rag_eval_pipeline.py)). The daily pipeline is `generate_data → ingest → detect_drift → reconcile → index → run_eval → wait_for_eval → publish_report → quality_gate`. The reconcile task fails if recall against ground truth drops below 99%. The eval runs asynchronously in the API and is awaited by a rescheduling sensor. Airflow only makes HTTP calls, so its image needs none of the project's dependencies.

## Results

Measured on the default dataset (2,000 orders, 5,950 source records, seed 42):

| Stage | Result |
|---|---|
| Schema drift detection | 8 / 8 injected drift events, 0 false positives |
| Reconciliation | 303 / 303 injected discrepancies, precision 1.00, recall 1.00 |
| Records rejected at normalization | 0 / 5,950 |

**Claude benchmark:** not yet recorded here. Run `docker compose up` with an `ANTHROPIC_API_KEY`, trigger the DAG (or `recon-rag eval`), and copy the table from `reports/eval_run_<id>.md`.

For reference, the offline extractive baseline (`LLM_PROVIDER=fake`) quotes the first line of the top search hit. It scores 100% grounding but only about 35–50% answer accuracy, and it **fails** the quality gate. That's the reason for the accuracy threshold.

## Quick start

```bash
cp .env.example .env          # add ANTHROPIC_API_KEY
docker compose up -d --build  # postgres+pgvector, api (:8000), airflow (:8080, admin/admin)
```

Then either unpause and trigger `rag_eval_pipeline` in Airflow at http://localhost:8080, or drive the API directly:

```bash
curl -X POST localhost:8000/datasets/generate -H 'content-type: application/json' -d '{"n_orders": 2000}'
curl -X POST localhost:8000/ingest
curl -X POST localhost:8000/reconcile
curl -X POST localhost:8000/index
curl -X POST localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "Do the source systems agree on order SO-000417? If not, what is wrong?"}'
curl -X POST localhost:8000/eval/runs -H 'content-type: application/json' -d '{"sample_size": 30}'
```

Interactive API docs are at http://localhost:8000/docs. The other endpoints are `/discrepancies`, `/drift`, `/schemas`, `/eval/runs/{id}` and `/eval/runs/{id}/report`.

### Local development

```bash
pip install -e ".[dev]"
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg://recon:recon@localhost:5432/recon
recon-rag pipeline --generate
recon-rag ask "Which source dropped a column, and which one?"
recon-rag eval --sample 30
```

Set `LLM_PROVIDER=fake` and `EMBEDDING_PROVIDER=hash` to run everything offline, with no API key or model download.

### Tests

```bash
pytest                              # unit tests, no database needed
DATABASE_URL=... pytest             # + integration: full pipeline, tools, API and an eval run on real pgvector
```

CI runs lint, format check and the full suite against a `pgvector/pgvector:pg16` service container.

## Configuration

Every setting in [`config.py`](src/recon_rag/config.py) can be overridden by an environment variable of the same name. The main ones:

| Variable | Default | |
|---|---|---|
| `AGENT_MODEL` / `AGENT_EFFORT` | `claude-opus-5-5` / `medium` | agent model and effort |
| `JUDGE_MODEL` / `JUDGE_EFFORT` | `claude-opus-5-5` / `high` | judge model and effort |
| `ANTHROPIC_FALLBACKS` | `default` | server-side refusal fallback; empty to disable |
| `GROUNDING_THRESHOLD` | `0.95` | quality gate |
| `ANSWER_ACCURACY_THRESHOLD` | `0.8` | quality gate |
| `EVAL_SAMPLE_SIZE` | all | stratified subset for cheaper runs |
| `DRIFT_AUTO_ACCEPT` | `0.75` | column-mapping confidence needed to skip the resolver |
| `AMOUNT_ABS_TOLERANCE` / `AMOUNT_REL_TOLERANCE` | `0.01` / `0.005` | reconciliation tolerances |

A full eval run makes roughly 150 agent conversations of 2–5 turns each, plus 150 judge calls. Use `EVAL_SAMPLE_SIZE` while iterating.

## Layout

```
src/recon_rag/
  datagen.py            seeded multi-source generator + ground truth
  ingest/               loaders, profiling, column mapping, drift diff, normalization
  reconcile/            matching + tolerance rules, ground-truth scoring
  index/                chunking, embeddings, hybrid retrieval (pgvector + FTS + RRF)
  agent/                tool definitions, tool-use loop, grounding gate
  llm/                  Anthropic client, scripted offline LLM, drift resolver
  eval/                 golden set, LLM judge, metrics, runner + reports
  api/main.py           FastAPI app
airflow/dags/           rag_eval_pipeline
tests/                  unit + integration
```

## Limitations

- The data is synthetic. The drift and discrepancy patterns are realistic, but a production feed will produce column shapes the heuristics haven't seen. That is the LLM resolver's job, and it has only been tested with a stub here.
- Using a Claude judge on a Claude agent risks shared blind spots. The deterministic grounding gate and the judge ↔ validator agreement metric are there to catch that. Spot-check the failures section of each report.
- The grounding gate checks numbers and identifiers, not free-text claims. The LLM judge covers those.
