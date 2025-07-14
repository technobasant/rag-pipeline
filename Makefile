.PHONY: help setup build up down clean logs scrape etl embed serve run_all test test-unit test-integration test-coverage test-report test-scraper test-etl test-embeddings test-api test-dag demo

# Default target
help:
	@echo "Data-Scrape-to-RAG Lakehouse - Development Commands"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup     - Initialize environment and copy .env.example to .env"
	@echo "  build     - Build all Docker images"
	@echo "  up        - Start all services with docker-compose"
	@echo "  down      - Stop all services"
	@echo "  clean     - Clean up data, logs, and Docker volumes"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  scrape    - Run web scraper only"
	@echo "  etl       - Run ETL pipeline (bronze → silver → gold)"
	@echo "  embed     - Generate embeddings and load to Chroma"
	@echo "  serve     - Start RAG API server"
	@echo "  run_all   - Execute full pipeline end-to-end"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test           - Run all tests"
	@echo "  test-unit      - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  test-coverage  - Run tests with coverage report"
	@echo "  test-report    - Generate comprehensive test report"
	@echo "  test-scraper   - Run scraper module tests"
	@echo "  test-etl       - Run ETL pipeline tests"
	@echo "  test-embeddings - Run embeddings module tests"
	@echo "  test-api       - Run API tests"
	@echo "  test-dag       - Run Airflow DAG tests"
	@echo ""
	@echo "Development Commands:"
	@echo "  demo      - Quick demo setup and execution"
	@echo "  logs      - Show logs from all services"

# Setup and initialization
setup:
	@echo "Setting up development environment..."
	@if [ ! -f .env ]; then cp env.example .env; echo "Created .env from env.example"; fi
	@echo "Environment setup complete!"

# Docker operations
build:
	@echo "Building Docker images..."
	docker-compose build

up:
	@echo "Starting all services..."
	docker-compose up -d
	@echo "Services started. Access points:"
	@echo "  - Airflow UI: http://localhost:8080 (admin/admin)"
	@echo "  - Spark Master UI: http://localhost:8081"
	@echo "  - MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
	@echo "  - RAG API: http://localhost:8001/docs"
	@echo "  - Chroma: http://localhost:8000"

down:
	@echo "Stopping all services..."
	docker-compose down

clean:
	@echo "Cleaning up data, logs, and Docker volumes..."
	docker-compose down -v
	docker system prune -f
	rm -rf data/raw/* data/bronze/* data/silver/* data/gold/* lineage/*
	@echo "Cleanup complete!"

logs:
	@echo "Showing logs from all services..."
	docker-compose logs -f

# Pipeline execution
scrape:
	@echo "Running web scraper..."
	docker-compose exec airflow-scheduler python -m src.scraper.scraper

etl:
	@echo "Running ETL pipeline..."
	docker-compose exec airflow-scheduler python -m src.etl.bronze_silver_gold

embed:
	@echo "Generating embeddings..."
	docker-compose exec airflow-scheduler python -m src.embeddings.embedder

serve:
	@echo "Starting RAG API server..."
	@echo "API will be available at http://localhost:8001"
	@echo "API documentation at http://localhost:8001/docs"

run_all: up
	@echo "Executing full pipeline..."
	@sleep 30  # Wait for services to be ready
	@echo "Step 1: Scraping data..."
	@$(MAKE) scrape
	@echo "Step 2: Running ETL..."
	@$(MAKE) etl
	@echo "Step 3: Generating embeddings..."
	@$(MAKE) embed
	@echo "Step 4: RAG API is ready!"
	@echo "Pipeline complete! Test with: curl 'http://localhost:8001/ask?question=What is Python?'"

# Testing
test:
	@echo "Running all tests..."
	./scripts/run_tests.py all -v

test-unit:
	@echo "Running unit tests..."
	./scripts/run_tests.py unit -v

test-integration:
	@echo "Running integration tests..."
	./scripts/run_tests.py integration -v

test-coverage:
	@echo "Running tests with coverage..."
	./scripts/run_tests.py unit -v -c

test-report:
	@echo "Generating comprehensive test report..."
	./scripts/run_tests.py report

test-scraper:
	@echo "Running scraper module tests..."
	./scripts/run_tests.py scraper -v

test-etl:
	@echo "Running ETL pipeline tests..."
	./scripts/run_tests.py etl -v

test-embeddings:
	@echo "Running embeddings module tests..."
	./scripts/run_tests.py embeddings -v

test-api:
	@echo "Running API tests..."
	./scripts/run_tests.py api -v

test-dag:
	@echo "Running Airflow DAG tests..."
	./scripts/run_tests.py dag -v

# Testing in Docker (alternative approach)
test-docker:
	@echo "Running tests in Docker container..."
	docker-compose exec airflow-scheduler python -m pytest tests/ -v

test-docker-coverage:
	@echo "Running tests with coverage in Docker..."
	docker-compose exec airflow-scheduler python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Demo workflow
demo: clean setup build up
	@echo "Starting demo workflow..."
	@sleep 45  # Wait for all services to initialize
	@echo "Triggering Airflow DAG..."
	@curl -X POST "http://localhost:8080/api/v1/dags/scrape_etl_embed_dag/dagRuns" \
		-H "Content-Type: application/json" \
		-u admin:admin \
		-d '{"dag_run_id": "demo_run_$(shell date +%s)"}'
	@echo ""
	@echo "Demo started! Monitor progress at:"
	@echo "  - Airflow UI: http://localhost:8080"
	@echo "  - MinIO Console: http://localhost:9001"
	@echo ""
	@echo "Once DAG completes, test RAG API:"
	@echo "  curl 'http://localhost:8001/ask?question=What is Python?'"

# Development helpers
install-deps:
	@echo "Installing Python dependencies locally..."
	pip install -r requirements.txt

format:
	@echo "Formatting code with black..."
	black src/ tests/

lint:
	@echo "Linting code with flake8..."
	flake8 src/ tests/

# Monitoring and debugging
status:
	@echo "Service status:"
	docker-compose ps

health:
	@echo "Health check for all services:"
	@curl -s http://localhost:8080/health || echo "Airflow: DOWN"
	@curl -s http://localhost:8001/health || echo "RAG API: DOWN"
	@curl -s http://localhost:8000/api/v1/heartbeat || echo "Chroma: DOWN"
	@curl -s http://localhost:9000/minio/health/live || echo "MinIO: DOWN"

# Data inspection
inspect-data:
	@echo "Data inspection:"
	@echo "Raw files:"
	@ls -la data/raw/ 2>/dev/null || echo "No raw data found"
	@echo "Bronze/Silver/Gold tables:"
	@ls -la data/bronze/ data/silver/ data/gold/ 2>/dev/null || echo "No processed data found"

# Quick API test
test-api-quick:
	@echo "Testing RAG API..."
	@curl -s "http://localhost:8001/ask?question=What%20is%20Python%3F&include_meta=1&top_k=5" | jq . || echo "API not responding"