from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
}

with DAG(
    'training_pipeline_dag',
    default_args=default_args,
    schedule_interval='@weekly',
    catchup=False
) as dag:

    # 1. Preprocessing (Full)
    preprocess = BashOperator(
        task_id='preprocess_full',
        bash_command='python /opt/airflow/dags/offline_training/preprocessing/pipelines/preprocess_full_pipeline.py'
    )

    # 2. Pretrain TikHarm
    pretrain = BashOperator(
        task_id='pretrain_tikharm',
        bash_command='python /opt/airflow/dags/offline_training/pretrain/train_tikharm.py'
    )
    
    # 3. Finetune VN
    finetune = BashOperator(
        task_id='finetune_vn',
        bash_command='python /opt/airflow/dags/offline_training/finetune/train_tiktok_vn.py'
    )
    
    preprocess >> pretrain >> finetune
