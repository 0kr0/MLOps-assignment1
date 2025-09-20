from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from code.models.train_model_mlflow import train

default_args = {
    'owner': 'Anton',
    'start_date': datetime(2025, 9, 18),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG('Model engineering', default_args = default_args, schedule_interval = timedelta(minutes=5), catch_up = False)

# Similar to Step 2, add a PythonOperator for train_wine_model
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train,
    op_kwargs={
        'train_path': '/path/to/repo/data/processed/train.csv',
        'test_path': '/path/to/repo/data/processed/test.csv',
        'model_path': '/path/to/repo/models/wine_model.joblib',
        'scaler_path': '/path/to/repo/models/wine_scaler.joblib'  # New
    },
    dag=dag,
)
