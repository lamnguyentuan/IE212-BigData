# Deployment & Orchestration

This module contains configurations for deploying the system across different environments.

## 1. Docker Compose (Local Dev)
The primary method for running the full stack locally.
- **File**: `docker-compose.yml` (in project root).
- **Services**: Kafka, Zookeeper, MinIO, MongoDB, Spark, Serving, Dashboard, Airflow.

## 2. Airflow (Orchestration)
DAGs located in `airflow/dags/` define the automation workflows:
- `ingest_tiktok_dag`: Daily crawl.
- `tikharm_to_silver_dag`: ETL pipeline.
- `training_pipeline_dag`: MLOps pipeline (Preprocess -> Train -> Finetune).

## 3. Kubernetes (Production)
Manifests in `k8s/` allow deploying key stateless services to a cluster.
- `minio.yaml`: Distributed Object Storage.
- `serving.yaml`: Scalable Model Inference.
