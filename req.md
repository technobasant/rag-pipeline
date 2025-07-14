# Home Assignment for Senior Big Data Engineer – AI-Focused Data Infrastructure

**Role:** Senior Big Data Engineer – AI-Focused Data Infrastructure  
**Effort:** ≤ 16 working hours (spread over ≤ 2 calendar days)  

## Deliverables

1. ≤ 5-minute screen-capture video (link in README) showing the full pipeline running locally

## 1. Scenario & Goal

Build a mini end-to-end "Data-Scrape-to-RAG" lakehouse:

1. Scrape a small public website (50–100 pages) or open dataset from the internet at runtime.
   - Examples: Project Gutenberg texts, BBC Tech RSS, a docs sub-domain, or any site that allows crawling.

2. Land → Bronze → Silver → Gold transform flow (Delta or Parquet) in an object store.

3. Generate embeddings for the cleaned text, persist them in a vector index, and expose them via a simple RAG API.

4. Provide metadata, lineage, and data-quality checks so downstream AI agents can trust the data.

## 2. Functional Requirements

| Area | Must-Have (free/offline) | Nice-to-Have (bonus) |
|------|--------------------------|----------------------|
| **Scraping** | Python scraper (BeautifulSoup, Playwright, or Scrapy) that respects rate-limits & robots.txt. Saves raw HTML & extracted text to MinIO. | JS-rendering with Playwright; configurable site URL via env var. |
| **Storage** | Local MinIO bucket as "data lake." Folder layout: `/raw`, `/bronze`, `/silver`, `/gold`. | S3-switch via env var. |
| **Catalog** | DuckDB + Delta Lake (delta-python or delta-rs). | Iceberg or Hive-metastore adapter. |
| **Compute** | Spark (local in Docker) -or- DuckDB SQL for transforms. | Polars/Fugue fallback for small data. |
| **Orchestration** | Apache Airflow DAG: scrape → bronze → silver → gold → embed. | Unit tests for DAG with pytest-airflow. |
| **Vector DB** | Chroma or FAISS, persisted to local volume. | Switchable back-ends. |
| **Embeddings** | Local model via SentenceTransformers (all-MiniLM-L6) or Ollama encoder. | GPU-accelerated encoder option. |
| **RAG API** | FastAPI route `/ask?question=` returning `{answer, sources}` by retrieving from vector store & running local LLM (Ollama or llama-cpp-python). | Streaming responses; top-k parameter. |
| **Lineage / Governance** | Emit OpenLineage events; JSON logs stored in `/lineage`. | Marquez UI docker for lineage graph. |
| **Observability** | Airflow UI, pretty + JSON logs, optional `/metrics` endpoint. | Grafana dashboard JSON export. |
| **Local Run** | Single `docker compose up` spins MinIO, Airflow, Spark, Chroma, FastAPI, Ollama. | Make targets for scrape, etl, embed, serve. |
| **Deploy (optional)** | IaC script (Terraform/SAM) for AWS Free Tier. | GitHub Actions CI to lint/test. |

## 3. Non-Functional Requirements

- **Runtime:** Full DAG on 50–100 pages ≤ 5 minutes on 4-core laptop.
- **Security:** No secrets committed; `.env.example` with placeholders.
- **Docs:** Clear README with architecture diagram, quick-start, and future-work section.

## 4. Deliverable Checklist

```text
README.md ← setup, design, video link
architecture.png ← diagram
docker-compose.yaml
Makefile ← common dev targets
airflow/dags/ ← scrape + ETL + embed DAG
src/ ← scraper, helpers, API, tests
sample_config/ ← yaml/json with target site & selectors
tests/ ← pytest unit tests
```

## 5. Mandatory Demo Video (≤ 5 min)

1. Clone repo → `docker compose up`.
2. Trigger Airflow DAG or `make run_all`.
3. Show MinIO buckets (`raw/`, `gold/`) & Delta table in DuckDB.
4. Call `/ask` twice to demo RAG answers with source links.
