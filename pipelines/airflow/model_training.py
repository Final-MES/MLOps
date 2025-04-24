from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Union
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
import numpy as np
import pickle
import os
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

default_args: Dict[str, Any] = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 19),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag: DAG = DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='ML 모델 학습 파이프라인',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

def fetch_training_data(**kwargs) -> str:
    """학습 데이터 가져오기"""
    # 실제 구현에서는 데이터베이스에서 데이터 가져오기
    # 여기서는 예시 데이터 생성
    np.random.seed(42)
    data_size: int = 1000
    
    data: Dict[str, np.ndarray] = {
        'temperature': np.random.normal(50, 15, data_size),
        'vibration': np.random.normal(2, 1, data_size),
        'pressure': np.random.normal(100, 20, data_size),
        'run_time': np.random.normal(5000, 1000, data_size),
        'target_quality': np.random.normal(85, 10, data_size)
    }
    
    # 약간의 상관관계 추가
    data['target_quality'] = data['target_quality'] - 0.2 * data['temperature'] + 0.3 * data['pressure'] - 0.1 * data['vibration']
    
    df: pd.DataFrame = pd.DataFrame(data)
    
    # 데이터 저장
    os.makedirs('/tmp/ml_data', exist_ok=True)
    file_path: str = '/tmp/ml_data/training_data.csv'
    df.to_csv(file_path, index=False)
    
    return file_path

def preprocess_data(**kwargs) -> str:
    """데이터 전처리"""
    ti = kwargs['ti']
    file_path: str = ti.xcom_pull(task_ids='fetch_training_data')
    
    # 데이터 로드
    df: pd.DataFrame = pd.read_csv(file_path)
    
    # 이상치 처리 (간단한 예시)
    for col in df.columns:
        mean: float = df[col].mean()
        std: float = df[col].std()
        df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    # 정규화
    for col in df.columns:
        if col != 'target_quality':  # 타겟 변수는 정규화하지 않음
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    
    # 전처리된 데이터 저장
    processed_file_path: str = file_path.replace('.csv', '_processed.csv')
    df.to_csv(processed_file_path, index=False)
    
    return processed_file_path

def train_model(**kwargs) -> Dict[str, Any]:
    """모델 학습"""
    ti = kwargs['ti']
    file_path: str = ti.xcom_pull(task_ids='preprocess_data')
    
    # 데이터 로드
    df: pd.DataFrame = pd.read_csv(file_path)
    
    # 특성과 타겟 분리
    X: pd.DataFrame = df.drop('target_quality', axis=1)
    y: pd.Series = df['target_quality']
    
    # 학습 및 테스트 데이터 분리
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow 실험 설정
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("quality_prediction")
    
    with mlflow.start_run() as run:
        # 모델 학습
        model: RandomForestRegressor = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # 모델 평가
        y_pred: np.ndarray = model.predict(X_test)
        mse: float = mean_squared_error(y_test, y_pred)
        r2: float = r2_score(y_test, y_pred)
        
        # 지표 로깅
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # 모델 저장
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # 모델을 로컬에도 저장
        os.makedirs('/tmp/models', exist_ok=True)
        model_path: str = '/tmp/models/quality_prediction_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"모델 학습 완료. MSE: {mse:.4f}, R²: {r2:.4f}")
        print(f"MLflow 실행 ID: {run.info.run_id}")
        
        return {
            'model_path': model_path,
            'mlflow_run_id': run.info.run_id,
            'metrics': {'mse': mse, 'r2': r2}
        }

def evaluate_model(**kwargs) -> str:
    """모델 평가 및 검증"""
    ti = kwargs['ti']
    result: Dict[str, Any] = ti.xcom_pull(task_ids='train_model')
    
    model_path: str = result['model_path']
    metrics: Dict[str, float] = result['metrics']
    
    # 모델 로드
    with open(model_path, 'rb') as f:
        model: RandomForestRegressor = pickle.load(f)
    
    # 여기서는 간단히 결과 출력
    print(f"모델 성능 평가: MSE = {metrics['mse']:.4f}, R² = {metrics['r2']:.4f}")
    
    # 기준 충족 여부 확인
    is_acceptable: bool = metrics['r2'] > 0.7  # R² > 0.7인 경우 허용
    
    if is_acceptable:
        return "모델이 성능 기준을 충족합니다. 배포 가능합니다."
    else:
        return "모델이 성능 기준을 충족하지 않습니다. 추가 개선이 필요합니다."

def register_model(**kwargs) -> Dict[str, Any]:
    """모델 등록"""
    ti = kwargs['ti']
    result: Dict[str, Any] = ti.xcom_pull(task_ids='train_model')
    evaluation: str = ti.xcom_pull(task_ids='evaluate_model')
    
    mlflow_run_id: str = result['mlflow_run_id']
    
    # MLflow 모델 레지스트리에 등록
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    model_uri: str = f"runs:/{mlflow_run_id}/random_forest_model"
    model_name: str = "quality_prediction_model"
    model_version = mlflow.register_model(model_uri, model_name)
    
    print(f"모델이 등록되었습니다: {model_name} 버전 {model_version.version}")
    
    # 모델을 최신 버전으로 설정
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production"
    )
    
    return {
        'model_name': model_name,
        'model_version': model_version.version,
        'status': 'production'
    }

# 태스크 정의
fetch_data_task: PythonOperator = PythonOperator(
    task_id='fetch_training_data',
    python_callable=fetch_training_data,
    dag=dag,
)

preprocess_task: PythonOperator = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task: PythonOperator = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

evaluate_task: PythonOperator = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

register_task: PythonOperator = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

trigger_deployment: TriggerDagRunOperator = TriggerDagRunOperator(
    task_id='trigger_model_deployment',
    trigger_dag_id='model_deployment_pipeline',
    conf={'model_info': '{{ ti.xcom_pull(task_ids="register_model") }}'},
    dag=dag,
)

# 태스크 의존성 설정
fetch_data_task >> preprocess_task >> train_task >> evaluate_task >> register_task >> trigger_deployment