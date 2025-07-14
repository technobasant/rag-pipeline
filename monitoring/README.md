# Monitoring Setup

This directory contains the monitoring configuration for the Data-Scrape-to-RAG pipeline using Prometheus and Grafana.

## Architecture

- **Prometheus**: Collects metrics from all services
- **Grafana**: Visualizes metrics with pre-configured dashboards

## Quick Start

1. Start the monitoring stack:
```bash
docker-compose up -d prometheus grafana
```

2. Access services:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

## Dashboards

### Data Pipeline Overview
The main dashboard (`data-pipeline-overview.json`) includes:
- Web Scraping Activity: Pages scraped and errors over time
- ETL Pipeline Processing: Records processed in Bronze/Silver/Gold layers
- Embedding Generation: Vectors created and generation rate
- RAG API Performance: Request rate and response times
- Service Health Status: Real-time status of all services

## Exporting Dashboards

To export a dashboard from Grafana:
1. Navigate to the dashboard
2. Click the share icon
3. Select "Export" tab
4. Toggle "Export for sharing externally"
5. Save the JSON file to `monitoring/grafana/dashboards/`

## Importing Dashboards

Dashboards in `monitoring/grafana/dashboards/` are automatically imported on Grafana startup.

To manually import:
1. Go to Grafana → Dashboards → Import
2. Upload the JSON file or paste its content
3. Select the Prometheus datasource
4. Click Import

## Adding Custom Metrics

### In Your Application

Example using Prometheus Python client:
```python
from prometheus_client import Counter, Histogram, start_http_server

# Define metrics
pages_scraped = Counter('scraper_pages_scraped_total', 'Total pages scraped')
scrape_duration = Histogram('scraper_duration_seconds', 'Time spent scraping')

# Use in code
pages_scraped.inc()
with scrape_duration.time():
    # scraping logic
    pass

# Start metrics server
start_http_server(9091)
```

### Configure Prometheus

Add your service to `prometheus.yml`:
```yaml
- job_name: 'my-service'
  static_configs:
    - targets: ['my-service:9091']
```

## Alerting (Optional)

To add alerts, create `monitoring/prometheus/alerts.yml`:
```yaml
groups:
  - name: pipeline
    rules:
      - alert: HighErrorRate
        expr: rate(scraper_errors_total[5m]) > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: High error rate in scraper
```

## Best Practices

1. **Metric Naming**: Follow Prometheus conventions
   - Use `_total` suffix for counters
   - Use `_seconds` suffix for durations
   - Use lowercase with underscores

2. **Label Cardinality**: Keep label values bounded
   - Good: `status="success"` or `status="error"`
   - Bad: `user_id="12345"` (unbounded)

3. **Dashboard Organization**: Group related metrics
   - Use rows for logical sections
   - Set appropriate time ranges
   - Add descriptions to panels

## Troubleshooting

### Prometheus Not Scraping
- Check target status: http://localhost:9090/targets
- Verify network connectivity between containers
- Check service logs: `docker logs prometheus`

### Grafana Not Loading
- Ensure Prometheus is running first
- Check datasource configuration
- Verify dashboard JSON syntax

### Missing Metrics
- Confirm service is exposing `/metrics` endpoint
- Check Prometheus scrape configuration
- Wait for scrape interval (default: 15s) 