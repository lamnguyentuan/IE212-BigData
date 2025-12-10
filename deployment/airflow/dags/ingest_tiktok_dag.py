from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ingest_tiktok_dag',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    # Task 1: Crawl TikTok Data
    crawl_task = BashOperator(
        task_id='crawl_tiktok',
        bash_command='python /opt/airflow/dags/data-ingestion/tiktok_crawl/tiktok_scraper.py --mode crawl'
    )
    
    # Task 2: Upload to MinIO Bronze (Assuming scraper handles this or separate script)
    # Using the scraper's integrated upload
    
    crawl_task
