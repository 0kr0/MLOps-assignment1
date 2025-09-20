from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from code.datasets.process_data import process_wine_data

default_args = {
    'owner': 'Anton',
    'start_date': datetime(2025, 9, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('Wine_data_engineering', default_args = default_args, schedule_interval = timedelta(minutes=5), catch_up = False)
process_task = PythonOperator(
    task_id = 'process_data',
    python_callable = process_wine_data,
    op_kwargs={'raw_path': "C:\Users\Антон\Documents\Innopolis\Practical Machine Learning and Deep Learning\MLOps-assignment1\data\raw\winequality-red.csv",
               'processed_dir': "C:\Users\Антон\Documents\Innopolis\Practical Machine Learning and Deep Learning\MLOps-assignment1\data\processed"},
    dag = dag,
)