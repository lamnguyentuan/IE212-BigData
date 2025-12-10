from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

with DAG(
    'tikharm_etl_dag',
    default_args=default_args,
    schedule_interval=None, # Manual trigger
    catchup=False
) as dag:

    # 1. Upload TikHarm to Bronze
    upload_bronze = BashOperator(
        task_id='upload_tikharm_bronze',
        bash_command='python /opt/airflow/dags/data-ingestion/tikharm_upload/upload_tikharm_to_minio.py'
    )
    
    # 2. Bronze -> Silver
    bronze_to_silver = BashOperator(
        task_id='bronze_to_silver',
        bash_command='python /opt/airflow/dags/data-pipeline/spark-streaming/medallion/bronze_to_silver_tikharm.py'
    )
    
    # 3. Silver -> Gold (Training Sets)
    silver_to_gold = BashOperator(
        task_id='silver_to_gold',
        bash_command='python /opt/airflow/dags/data-pipeline/spark-streaming/medallion/silver_to_gold_training_sets.py'
    )
    
    upload_bronze >> bronze_to_silver >> silver_to_gold
