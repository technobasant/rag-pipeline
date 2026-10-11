# Data-Scrape-to-RAG Lakehouse

An end-to-end, locally runnable scrape-to-RAG lakehouse. It scrapes a public documentation site into MinIO, refines the pages through Bronze, Silver and Gold layers with Spark, catalogs them in DuckDB, embeds the Gold chunks into Chroma, and answers questions through a FastAPI RAG API backed by a local Ollama model (via LiteLLM when installed). Airflow orchestrates the pipeline, OpenLineage events feed a Marquez lineage graph, and Prometheus and Grafana provide observability.

## 🏗️ Architecture

![Architecture Diagram](architecture.png)

## Tech Stack

- **Orchestration:** Apache Airflow 2.8 (`scrape_etl_embed_dag`: scrape → bronze → silver → gold → embed)
- **Scraping:** requests + BeautifulSoup with robots.txt checks; optional Playwright engine for JavaScript-rendered pages (`SCRAPER_ENGINE`)
- **Object storage:** MinIO (S3-compatible), `datalake` bucket with `raw/`, `bronze/`, `silver/` and `gold/` prefixes
- **Transforms:** Apache Spark 3.5 (PySpark) writing Parquet for each layer
- **Catalog:** DuckDB views over the lake layers (`data/catalog.duckdb`, created on the first run)
- **Embeddings and vector store:** `all-MiniLM-L6-v2` embeddings served by Ollama, stored in Chroma
- **LLM and API:** FastAPI (`/ask`, `/ask/stream`, `/search`, `/health`, `/metrics`, `/chat` web UI) calling Ollama (`qwen3:0.6b` by default) through `src/common/llm_client.py`, which uses LiteLLM when it is installed and the Ollama Python client otherwise
- **Lineage:** OpenLineage events, Marquez API and Marquez Web UI
- **Observability:** Prometheus metrics, provisioned Grafana dashboards, structlog JSON logs
- **Tooling:** Docker Compose, Make, pytest

## 🚀 Quick Start

```bash
# 1. Create the local env files (.env.docker, env.docker, .env) from the tracked *.example templates
make setup

# 2. Build the images and start all services
make up

# 3. Run the full pipeline: scrape → Bronze/Silver/Gold ETL → embeddings into Chroma
make run_all

# 4. Ask the RAG API a question
curl "http://localhost:8001/ask?question=What%20is%20Python%3F"
```

`make run_all` runs the scrape, ETL and embedding steps directly in the Airflow scheduler container. To run the same steps as an Airflow DAG instead, open the Airflow UI, unpause `scrape_etl_embed_dag` (DAGs start paused) and trigger it.

## 📋 Prerequisites

- Docker & Docker Compose
- Make
- 4GB+ RAM available for containers
- Internet connection for scraping
- Spark/Hadoop S3 JARs in `jars/` for the ETL step (not committed because of their size; see [JARS_README.md](JARS_README.md))

## 🛠️ Development

### Available Commands

```bash
make help          # Show all available commands
make setup         # Initialize environment
make build         # Build Docker images
make up            # Start all services
make down          # Stop services
make clean         # Clean up data and volumes
make test          # Run test suite
make demo          # Quick demo workflow
```

### Service Endpoints

The credentials below are local development defaults. Change them before exposing any service beyond your own machine.

- **Airflow UI**: <http://localhost:8080> (airflow/airflow, local development default)
- **MinIO Console**: <http://localhost:9001> (minioadmin/minioadmin, local development default)
- **RAG API**: <http://localhost:8001/docs>
- **Chroma DB**: <http://localhost:8000>
- **Spark Master UI**: <http://localhost:8081>
- **Marquez Web UI**: <http://localhost:3001>

## 📊 Pipeline Overview

1. **Scrape**: Web scraper fetches pages from docs.python.org and saves raw HTML, text and metadata to MinIO
2. **Bronze**: Raw text and metadata joined and stored as Parquet
3. **Silver**: Cleaned and normalized text data
4. **Gold**: Chunked text ready for embedding
5. **Embed**: Generate `all-MiniLM-L6-v2` vectors with Ollama and load them into Chroma
6. **RAG**: Query interface with local LLM via Ollama

## 🔧 Configuration

The containers read `.env.docker`, which `make setup` creates from the tracked `.env.docker.example` template. `.env.docker` is git-ignored, so edit it freely. `.env` (from `.env.example`) is a variant with `localhost` hosts for running modules outside Docker. All credentials in the templates are local development defaults, and the Airflow Fernet key and webserver secret key must be set for any real deployment.

- `SCRAPE_BASE_URL`: Target website to scrape
- `SCRAPE_MAX_PAGES`: Maximum pages to process
- `OLLAMA_MODEL`: LLM model for RAG responses (default pulled automatically in `ollama-entrypoint.sh`)
- `LITELLM_EMBED_MODEL`: Embedding model served by Ollama (default `mahonzhan/all-MiniLM-L6-v2`, also pulled in `ollama-entrypoint.sh`)

## 📁 Project Structure

```md
├── airflow/dags/           # Airflow DAG definitions
├── src/
│   ├── common/            # Shared utilities
│   ├── scraper/           # Web scraping logic
│   ├── etl/               # Data transformation
│   ├── catalog/           # DuckDB catalog over the lake layers
│   ├── embeddings/        # Vector generation
│   └── api/               # FastAPI RAG service
├── sample_config/         # Scraping configuration
├── tests/                 # Test suite
├── data/                  # Data storage (mounted volumes)
└── lineage/               # OpenLineage events
```

## 🧪 Testing

```bash
# Run all tests
make test

# Test individual components
docker-compose exec airflow-scheduler python -m pytest tests/test_scraper.py -v
```

## 📈 Monitoring

- **Airflow UI**: Monitor DAG execution and logs
- **MinIO Console**: Browse data lake contents  
- **API Metrics**: `/metrics` endpoint for Prometheus
- **Lineage**: OpenLineage events in `/lineage` directory
- **Prometheus**: <http://localhost:9090> - Metrics collection
- **Grafana**: <http://localhost:3000> (admin/admin, local development default) - Dashboard visualization

### Grafana Dashboards

The project includes pre-configured Grafana dashboards:

- **Data Pipeline Overview**: Complete pipeline monitoring including scraping, ETL, embeddings, and API metrics
- **Service Health**: Real-time status of all services

To export/import dashboards:

```bash
# Export all dashboards
python monitoring/dashboard_manager.py export --all

# Import a dashboard
python monitoring/dashboard_manager.py import monitoring/grafana/dashboards/data-pipeline-overview.json
```

## 🎥 Demo Video

Watch a 5-minute demo showing the complete pipeline in action:

[![Demo Video](https://img.shields.io/badge/▶️_Watch_Demo-Google_Drive-blue?style=for-the-badge)](https://drive.google.com/file/d/1t4fgtmcU1nBDCjIeTbbeYqh0B7m_Pnv5/view?usp=drive_link)

The demo covers:
1. `docker compose up` - Starting all services
2. Trigger Airflow DAG - Running the data pipeline
3. Show MinIO buckets and Delta tables - Data lake exploration
4. Call `/ask` endpoint - Testing the RAG API

## 🔍 Troubleshooting

### Common Issues

**Services not starting:**

```bash
make down && make clean && make up
```

**Out of memory:**

- Reduce `SCRAPE_MAX_PAGES` in `.env.docker`
- Increase Docker memory limit

**Scraping blocked:**

- Check `robots.txt` compliance
- Adjust `SCRAPE_DELAY_SECONDS`

### Logs

```bash
# View all service logs
make logs

# Specific service logs
docker-compose logs -f airflow-scheduler
docker-compose logs -f rag-api
```

## 🚧 Implemented Nice-to-Have Features

- [x] JavaScript rendering with Playwright (configurable via `SCRAPER_ENGINE`)
- [x] Configurable site URL via environment variable (`SCRAPE_BASE_URL`)
- [x] S3 storage support (switch via `STORAGE_BACKEND`)
- [x] GPU-accelerated embeddings (auto-detected, set `EMBEDDING_DEVICE=cuda`)
- [x] Streaming API responses (`/ask/stream` endpoint)
- [x] pytest-airflow DAG tests
- [x] Marquez integration for lineage visualization (enable via `MARQUEZ_ENABLED`)
- [x] Grafana dashboards for monitoring

## 🔄 Linting and Tests

```bash
# Install development dependencies
pip install ruff black isort mypy pytest pytest-cov

# Run linting
ruff check src/ tests/
black --check src/ tests/
isort --check-only src/ tests/

# Run tests
pytest tests/ -v --cov=src
```

## 🚀 Future Enhancements

- [ ] Kubernetes Helm charts for production deployment
- [ ] Advanced RAG with reranking and hybrid search
- [ ] Real-time streaming data ingestion
- [ ] Multi-language support for scraping
- [ ] Distributed Spark processing on Kubernetes
- [ ] MLflow integration for model versioning

## 📄 License

Licensed under the [Apache License 2.0](LICENSE). Copyright 2025 Basant Bhattarai.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Add tests for new functionality
4. Submit pull request

## LLM Integration (LiteLLM)

This project uses [LiteLLM](https://github.com/BerriAI/litellm) as a unified client for both embeddings and text generation (completions). On container start the entrypoint script pulls the required models, so the API is ready on first request. You can switch to OpenAI or any other provider by adjusting environment variables.

### Example configuration for Ollama (local)

```text
LITELLM_API_BASE=http://localhost:11434
LITELLM_MODEL=qwen3:0.6b
LITELLM_EMBED_MODEL=mahonzhan/all-MiniLM-L6-v2
# LITELLM_API_KEY=  # Only needed for remote providers
```

- The embedding pipeline and RAG API both go through `src/common/llm_client.py` for all LLM operations.
- LiteLLM is optional: it is not in `requirements.txt`, and without it the client falls back to the Ollama Python client. Install `litellm` to use other providers.
- To switch providers, just change the environment variables.

See `sample_config/site.yaml` for a sample config.
