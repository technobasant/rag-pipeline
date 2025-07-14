"""
Data lineage tracking using OpenLineage and Marquez.
"""

import os
import json
import uuid
from typing import Dict, Any, Optional, List, Union
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager

import structlog
import httpx

# Optional OpenLineage imports with fallbacks
try:
    from openlineage.client import OpenLineageClient
    from openlineage.client.event_v2 import (
        RunEvent,
        RunState,
        Run,
        Job,
        Dataset,
        InputDataset,
        OutputDataset
    )
    from openlineage.client.uuid import generate_new_uuid
    from openlineage.client.facet_v2 import RunFacet
    OPENLINEAGE_AVAILABLE = True
except ImportError:
    # Mock classes for when OpenLineage is not available
    OPENLINEAGE_AVAILABLE = False
    
    class MockOpenLineageClient:
        def __init__(self, *args, **kwargs):
            pass
    
    class MockRunState:
        START = "START"
        COMPLETE = "COMPLETE"
        FAIL = "FAIL"
        RUNNING = "RUNNING"
        ABORT = "ABORT"
        OTHER = "OTHER"
        
        @property
        def value(self):
            return self
    
    class MockDataset:
        def __init__(self, namespace=None, name=None, facets=None):
            self.namespace = namespace
            self.name = name
            self.facets = facets or {}
    
    class MockRunFacet:
        def to_dict(self):
            return {}
    
    # Assign mock classes
    OpenLineageClient = MockOpenLineageClient
    RunState = MockRunState()
    Dataset = MockDataset
    InputDataset = MockDataset
    OutputDataset = MockDataset
    RunFacet = MockRunFacet
    
    def generate_new_uuid():
        return str(uuid.uuid4())
    
    # Mock other classes
    RunEvent = dict
    Run = dict
    Job = dict

from .config import config


class MarquezClient:
    """Client for sending lineage events to Marquez."""
    
    def __init__(self, url: str, namespace: str):
        self.url = url.rstrip('/')
        self.namespace = namespace
        self.logger = structlog.get_logger(__name__)
        
    async def send_event(self, event: Dict[str, Any]) -> bool:
        """Send an OpenLineage event to Marquez."""
        if not config.lineage.marquez_enabled:
            return True
            
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{self.url}/api/v1/lineage",
                    json=event,
                    headers={
                        "Content-Type": "application/json",
                        "X-Namespace": self.namespace
                    }
                )
                response.raise_for_status()
                self.logger.info("Sent lineage event to Marquez", 
                               event_type=event.get("eventType"),
                               job_name=event.get("job", {}).get("name"))
                return True
                
        except Exception as e:
            self.logger.error("Failed to send event to Marquez", error=str(e))
            return False
    
    def send_event_sync(self, event: Dict[str, Any]) -> bool:
        """Synchronous version of send_event."""
        if not config.lineage.marquez_enabled:
            return True
            
        try:
            response = httpx.post(
                f"{self.url}/api/v1/lineage",
                json=event,
                headers={
                    "Content-Type": "application/json",
                    "X-Namespace": self.namespace
                },
                timeout=10.0
            )
            response.raise_for_status()
            self.logger.info("Sent lineage event to Marquez", 
                           event_type=event.get("eventType"),
                           job_name=event.get("job", {}).get("name"))
            return True
            
        except Exception as e:
            self.logger.error("Failed to send event to Marquez", error=str(e))
            return False


class LineageTracker:
    """Track data lineage for the ETL pipeline."""
    
    def __init__(self, lineage_dir: str = "lineage"):
        self.lineage_dir = Path(lineage_dir)
        self.lineage_dir.mkdir(exist_ok=True)
        self.logger = structlog.get_logger(__name__)
        
        # Initialize Marquez client if enabled
        if config.lineage.marquez_enabled:
            self.marquez_client = MarquezClient(
                url=config.lineage.marquez_url,
                namespace=config.lineage.marquez_namespace
            )
        else:
            self.marquez_client = None
        
        # OpenLineage namespace
        self.namespace = config.lineage.marquez_namespace or "data-scrape-rag"
        self.producer = "https://github.com/yourusername/data-scrape-rag"
    
    def _save_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """Save lineage event to local file and send to Marquez."""
        timestamp = datetime.now(timezone.utc).isoformat()
        filename = f"{event_type}_{timestamp.replace(':', '-')}.json"
        filepath = self.lineage_dir / filename
        
        # Save locally
        with open(filepath, 'w') as f:
            json.dump(event_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved lineage event", 
                        event_type=event_type, 
                        filepath=str(filepath))
        
        # Send to Marquez if enabled
        if self.marquez_client:
            self.marquez_client.send_event_sync(event_data)
    
    def create_run_event(
        self,
        job_name: str,
        run_id: str,
        event_type: RunState,
        inputs: Optional[List[Dataset]] = None,
        outputs: Optional[List[Dataset]] = None,
        job_facets: Optional[Dict[str, Any]] = None,
        run_facets: Optional[Dict[str, RunFacet]] = None
    ) -> Dict[str, Any]:
        """Create an OpenLineage run event."""
        event_time = datetime.now(timezone.utc).isoformat()
        
        # Build job
        job_data = {
            "namespace": self.namespace,
            "name": job_name
        }
        if job_facets:
            job_data["facets"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                 for k, v in job_facets.items()}
        
        # Build run
        run_data = {
            "runId": run_id
        }
        if run_facets:
            run_data["facets"] = {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                 for k, v in run_facets.items()}
        
        # Build event
        event = {
            "eventType": event_type.value if hasattr(event_type, 'value') else event_type,
            "eventTime": event_time,
            "run": run_data,
            "job": job_data,
            "producer": self.producer,
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/RunEvent"
        }
        
        # Add datasets
        if inputs:
            event["inputs"] = [self._dataset_to_dict(d) for d in inputs]
        if outputs:
            event["outputs"] = [self._dataset_to_dict(d) for d in outputs]
        
        return event
    
    def _dataset_to_dict(self, dataset: Dataset) -> Dict[str, Any]:
        """Convert Dataset object to dictionary."""
        if isinstance(dataset, dict):
            return dataset
            
        result = {
            "namespace": dataset.namespace if hasattr(dataset, 'namespace') else self.namespace,
            "name": dataset.name if hasattr(dataset, 'name') else str(dataset)
        }
        
        if hasattr(dataset, 'facets') and dataset.facets:
            result["facets"] = {}
            for k, v in dataset.facets.items():
                if hasattr(v, 'to_dict'):
                    result["facets"][k] = v.to_dict()
                else:
                    result["facets"][k] = v
        
        return result
    
    def track_job_start(
        self,
        job_name: str,
        run_id: Optional[str] = None,
        inputs: Optional[List[Dataset]] = None,
        description: Optional[str] = None,
        source_code_location: Optional[str] = None
    ) -> str:
        """Track job start event."""
        run_id = run_id or str(generate_new_uuid())
        
        job_facets = {}
        if description:
            job_facets["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": description
            }
        
        if source_code_location:
            job_facets["sourceCodeLocation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SourceCodeLocationJobFacet.json",
                "type": "file",
                "url": f"file://{source_code_location}"
            }
        
        event = self.create_run_event(
            job_name=job_name,
            run_id=run_id,
            event_type="START",
            inputs=inputs,
            job_facets=job_facets if job_facets else None
        )
        
        self._save_event(f"{job_name}_start", event)
        return run_id
    
    def track_job_complete(
        self,
        job_name: str,
        run_id: str,
        outputs: Optional[List[Dataset]] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> None:
        """Track job completion event."""
        run_facets = {}
        
        if stats:
            run_facets["stats"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/StatsRunFacet.json",
                **stats
            }
        
        event = self.create_run_event(
            job_name=job_name,
            run_id=run_id,
            event_type="COMPLETE",
            outputs=outputs,
            run_facets=run_facets if run_facets else None
        )
        
        self._save_event(f"{job_name}_complete", event)
    
    def track_job_fail(
        self,
        job_name: str,
        run_id: str,
        error_message: str,
        stack_trace: Optional[str] = None
    ) -> None:
        """Track job failure event."""
        run_facets = {
            "errorMessage": {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/ErrorMessageRunFacet.json",
                "message": error_message,
                "programmingLanguage": "Python"
            }
        }
        
        if stack_trace:
            run_facets["errorMessage"]["stackTrace"] = stack_trace
        
        event = self.create_run_event(
            job_name=job_name,
            run_id=run_id,
            event_type="FAIL",
            run_facets=run_facets
        )
        
        self._save_event(f"{job_name}_fail", event)
    
    def create_dataset(
        self,
        name: str,
        namespace: Optional[str] = None,
        description: Optional[str] = None,
        schema: Optional[List[Dict[str, str]]] = None,
        stats: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Create a dataset with optional facets."""
        dataset = {
            "namespace": namespace or self.namespace,
            "name": name,
            "facets": {}
        }
        
        if description:
            dataset["facets"]["documentation"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationDatasetFacet.json",
                "description": description
            }
        
        if schema:
            dataset["facets"]["schema"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                "fields": schema
            }
        
        if stats:
            dataset["facets"]["dataQualityMetrics"] = {
                "_producer": self.producer,
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DataQualityMetricsInputDatasetFacet.json",
                "rowCount": stats.get("rowCount", 0),
                "bytes": stats.get("bytes", 0),
                "columnMetrics": stats.get("columnMetrics", {})
            }
        
        return dataset
    
    def create_minio_dataset(self, path: str, **kwargs) -> Dataset:
        """Create a MinIO/S3 dataset."""
        return self.create_dataset(
            name=path,
            namespace=f"s3://{config.minio.bucket_name}",
            **kwargs
        )
    
    def create_spark_dataset(self, table: str, **kwargs) -> Dataset:
        """Create a Spark/Delta table dataset."""
        return self.create_dataset(
            name=table,
            namespace=f"spark://delta/{config.minio.bucket_name}",
            **kwargs
        )
    
    def create_chroma_dataset(self, collection: str, **kwargs) -> Dataset:
        """Create a Chroma vector database dataset."""
        return self.create_dataset(
            name=collection,
            namespace=f"chroma://{config.chroma.host}:{config.chroma.port}",
            **kwargs
        )

    def create_input_dataset(self, name: str, namespace: Optional[str] = None, **kwargs) -> Dataset:
        """Convenience method to create an input dataset."""
        return self.create_dataset(name, namespace, **kwargs)
    
    def create_output_dataset(self, name: str, namespace: Optional[str] = None, **kwargs) -> Dataset:
        """Convenience method to create an output dataset."""
        return self.create_dataset(name, namespace, **kwargs)

    @contextmanager
    def track_operation(self, operation_type, inputs=None, outputs=None):
        print(f"[Lineage] Start {operation_type} | inputs={inputs} | outputs={outputs}")
        try:
            yield self
        finally:
            print(f"[Lineage] End {operation_type}")

    def add_metadata(self, metadata: dict):
        print(f"[Lineage] Metadata: {metadata}")


# Correct context manager usage: Do NOT decorate classes with @contextmanager.
# Only use @contextmanager for generator functions. This class implements __enter__ and __exit__.
class LineageContext:
    """Context manager for tracking lineage of a job execution."""
    
    def __init__(
        self,
        job_name: str,
        tracker: LineageTracker,
        description: Optional[str] = None,
        source_code_location: Optional[str] = None
    ):
        self.job_name = job_name
        self.tracker = tracker
        self.description = description
        self.source_code_location = source_code_location
        self.run_id = None
        self.inputs = []
        self.outputs = []
        self.stats = {}
    
    def add_input(self, dataset: Dataset) -> None:
        """Add an input dataset."""
        self.inputs.append(dataset)
    
    def add_output(self, dataset: Dataset) -> None:
        """Add an output dataset."""
        self.outputs.append(dataset)
    
    def add_stats(self, **kwargs) -> None:
        """Add statistics to the run."""
        self.stats.update(kwargs)
    
    def __enter__(self):
        """Start tracking the job."""
        self.run_id = self.tracker.track_job_start(
            job_name=self.job_name,
            description=self.description,
            source_code_location=self.source_code_location,
            inputs=self.inputs
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Complete or fail the job based on exceptions."""
        if exc_type is None:
            # Success
            self.tracker.track_job_complete(
                job_name=self.job_name,
                run_id=self.run_id,
                outputs=self.outputs,
                stats=self.stats
            )
        else:
            # Failure
            import traceback
            self.tracker.track_job_fail(
                job_name=self.job_name,
                run_id=self.run_id,
                error_message=str(exc_val),
                stack_trace=traceback.format_exc()
            )
        
        # Don't suppress exceptions
        return False


# Global lineage tracker instance
lineage_tracker = LineageTracker()