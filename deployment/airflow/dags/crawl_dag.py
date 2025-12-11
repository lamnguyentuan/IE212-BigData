
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator

default_args = {
    'owner': 'minhhieu',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'tiktok_crawl_pipeline',
    default_args=default_args,
    description='Crawl TikTok Data and Process',
    schedule_interval=timedelta(days=1),
)

# Task 1: Run Crawler Container
# Note: we use "auto_remove" to clean up, and "command" to override the default.
# But our crawler dockerfile might not have a default CMD that exits. 
# We should probably run the script specifically.
crawl_task = DockerOperator(
    task_id='run_crawler',
    image='ie212-bigdata_crawler:latest', # Will need to ensure image name matches
    api_version='auto',
    auto_remove=True,
    command='python data_pipeline/end_to_end_demo.py review',
    docker_url='unix://var/run/docker.sock',
    network_mode='ie212-bigdata_default', # Connect to same network
    environment={
        'MINIO_ENDPOINT': 'minio:9000',
        'MINIO_ACCESS_KEY': 'minioadmin',
        'MINIO_SECRET_KEY': 'minioadmin'
    },
    dag=dag,
)
