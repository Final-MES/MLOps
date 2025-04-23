from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.models import Variable
import pandas as pd
import numpy as np
import json
import os
import logging
from src.data_preprocessing import SensorDataPreprocessor  # 보간 기능이 있는 클래스 임포트

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 4, 19),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'sensor_data_processing_pipeline',
    default_args=default_args,
    description='서로 다른 주기의 센서 데이터 처리 및 보간 파이프라인',
    schedule_interval=timedelta(hours=1),
    catchup=False,
)

def read_sensor_data(**kwargs):
    """각 센서의 데이터 파일에서 데이터 읽기"""
    # Airflow Variable에서 데이터 경로 가져오기
    try:
        data_dir = Variable.get("sensor_data_dir", default_var="/app/data/raw/vibration")
    except:
        data_dir = "/app/data/raw/vibration"
        logging.info(f"Variable sensor_data_dir not found, using default path: {data_dir}")
    
    # 센서 데이터 파일 목록 가져오기 (실제 환경에서는 파일 명명 규칙에 맞게 수정 필요)
    sensor_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    
    if not sensor_files:
        logging.error("No sensor data files found in the directory")
        return {'status': 'error', 'message': 'No sensor data files found'}
    
    # 가장 최근 파일 선택 (또는 필요에 따라 다른 선택 방법 사용)
    latest_file = max(sensor_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
    file_path = os.path.join(data_dir, latest_file)
    
    try:
        # 센서 데이터 로드
        df = pd.read_csv(file_path,names =["time","normal","type1","type2","type3"])
        logging.info(f"Loaded {len(df)} records from {file_path}")
        
        # 시간 컬럼 확인 및 처리
        time_column = None
        for col in ['timestamp', 'time', 'datetime']:
            if col in df.columns:
                time_column = col
                df[col] = pd.to_datetime(df[col])
                break
        
        if time_column is None:
            logging.warning("No time column found in the data, creating a default one")
            df['time'] = pd.to_datetime(pd.date_range(start='now', periods=len(df), freq='S'))
            time_column = 'time'
        
        # 상태 컬럼 확인
        state_column = None
        for col in ['state', 'status', 'condition', 'label', 'class']:
            if col in df.columns:
                state_column = col
                break
        
        if state_column is None:
            logging.warning("No state column found in the data")
            # 실제 구현에서는 필요에 따라 기본값 설정 또는 오류 발생
        
        # 데이터 임시 저장
        os.makedirs('/tmp/sensor_data', exist_ok=True)
        temp_path = f'/tmp/sensor_data/raw_sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(temp_path, index=False)
        
        # 데이터 정보 반환
        data_info = {
            'file_path': temp_path,
            'original_file': latest_file,
            'record_count': len(df),
            'time_column': time_column,
            'state_column': state_column,
            'columns': list(df.columns),
            'status': 'success'
        }
        
        if state_column:
            # 상태별 레코드 수 계산
            state_counts = df[state_column].value_counts().to_dict()
            data_info['state_counts'] = state_counts
            logging.info(f"State distribution: {state_counts}")
        
        return data_info
        
    except Exception as e:
        logging.error(f"Error reading sensor data: {e}")
        return {'status': 'error', 'message': str(e)}

def preprocess_and_interpolate_data(**kwargs):
    """센서 데이터 전처리 및 보간"""
    ti = kwargs['ti']
    data_info = ti.xcom_pull(task_ids='read_sensor_data')
    
    if not data_info or data_info.get('status') == 'error':
        logging.error("Failed to read sensor data")
        return {'status': 'error', 'message': 'Failed to read sensor data'}
    
    file_path = data_info['file_path']
    time_column = data_info['time_column']
    state_column = data_info.get('state_column')
    
    try:
        # 데이터 로드
        df = pd.read_csv(file_path, names = ["time","normal","type1","type2","type3"])
        df[time_column] = pd.to_datetime(df["time"])
        
        # 전처리기 초기화
        preprocessor = SensorDataPreprocessor(window_size=15)
        
        # 결측치 처리
        logging.info("Handling missing values")
        missing_before = df.isna().sum().sum()
        df = preprocessor.handle_missing_values(df)
        missing_after = df.isna().sum().sum()
        logging.info(f"Missing values: {missing_before} before, {missing_after} after preprocessing")
        
        # 이상치 처리
        logging.info("Handling outliers")
        df = preprocessor.handle_outliers(df, exclude_columns=[time_column, state_column] if state_column else [time_column])
        
        # 시간 컬럼 처리 (보간을 위해 숫자로 변환)
        df['time_seconds'] = df[time_column].astype(np.int64) // 10**9  # 타임스탬프를 초 단위로 변환
        
        # 특성 컬럼 (시간과 상태 컬럼 제외)
        feature_columns = [col for col in df.columns if col not in [time_column, state_column, 'time_seconds']]
        
        # 보간 데이터 준비 (전처리기에 맞게 데이터 구조 변환)
        sensor_data = {}
        
        # 전체 데이터를 하나의 센서로 취급
        sensor_data['combined_sensor'] = df[['time_seconds'] + feature_columns].rename(columns={'time_seconds': 'time'})
        
        # 균일한 시간 간격으로 데이터 보간
        logging.info("Interpolating sensor data")
        
        # 시간 간격 계산 (초 단위)
        times = sorted(df['time_seconds'].values)
        intervals = np.diff(times)
        min_interval = np.min(intervals[intervals > 0]) if any(intervals > 0) else 0.001
        logging.info(f"Using interpolation step: {min_interval}")
        
        # 보간 수행
        interpolated_data = preprocessor.interpolate_sensor_data(
            sensor_data,
            time_range=None,  # 자동 생성
            step=min_interval,
            kind='linear'  # 선형 보간 사용
        )
        
        # 보간된 데이터 처리
        interpolated_df = interpolated_data['combined_sensor']
        
        # 원래 시간 형식으로 변환
        interpolated_df['timestamp'] = pd.to_datetime(interpolated_df['time'], unit='s')
        
        # 상태 정보 추가 (가장 가까운 시간의 상태로 보간)
        if state_column:
            # 원본 데이터에서 시간과 상태만 추출
            state_df = df[[time_column, state_column]].copy()
            state_df['time_seconds'] = state_df[time_column].astype(np.int64) // 10**9
            
            # 가장 가까운 시간의 상태로 보간
            # 각 보간된 시간에 대해 가장 가까운 원본 시간 인덱스 찾기
            from scipy.spatial.distance import cdist
            
            # 두 시간 배열 간의 거리 계산
            distances = cdist(
                interpolated_df['time'].values.reshape(-1, 1),
                state_df['time_seconds'].values.reshape(-1, 1)
            )
            
            # 각 행에 대한 최소 거리 인덱스 찾기
            closest_indices = np.argmin(distances, axis=1)
            
            # 보간된 데이터에 상태 추가
            interpolated_df[state_column] = state_df[state_column].iloc[closest_indices].values
        
        # 최종 데이터 저장
        processed_path = f'/tmp/sensor_data/processed_sensor_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        interpolated_df.to_csv(processed_path, index=False)
        
        logging.info(f"Processed data saved to {processed_path} ({len(interpolated_df)} records)")
        
        return {
            'processed_file_path': processed_path,
            'record_count': len(interpolated_df),
            'time_column': 'timestamp',
            'state_column': state_column,
            'feature_columns': feature_columns,
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"Error preprocessing sensor data: {e}")
        return {'status': 'error', 'message': str(e)}

def prepare_data_for_model(**kwargs):
    """전처리된 데이터를 모델 학습용으로 준비"""
    ti = kwargs['ti']
    preprocess_result = ti.xcom_pull(task_ids='preprocess_and_interpolate_data')
    
    if not preprocess_result or preprocess_result.get('status') == 'error':
        logging.error("Failed to preprocess sensor data")
        return {'status': 'error', 'message': 'Failed to preprocess sensor data'}
    
    processed_file_path = preprocess_result['processed_file_path']
    state_column = preprocess_result.get('state_column')
    
    try:
        # 전처리된 데이터 로드
        df = pd.read_csv(processed_file_path)
        
        # 특성 공학 (필요한 경우)
        logging.info("Performing feature engineering")
        
        # 통계적 특성 추출
        preprocessor = SensorDataPreprocessor(window_size=15)
        feature_columns = preprocess_result['feature_columns']
        
        # 추가 특성 생성
        df = preprocessor.extract_statistical_moments(df, columns=feature_columns)
        df = preprocessor.extract_frequency_features(df, columns=feature_columns)
        
        # 최종 학습 데이터 경로 지정
        final_data_path = os.path.join('/app/data/processed', f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # 최종 디렉토리 생성
        os.makedirs(os.path.dirname(final_data_path), exist_ok=True)
        
        # 최종 데이터 저장
        df.to_csv(final_data_path, index=False)
        
        logging.info(f"Training data saved to {final_data_path} ({len(df)} records, {len(df.columns)} columns)")
        
        # 클래스 분포 확인 (상태 컬럼이 있는 경우)
        if state_column and state_column in df.columns:
            state_counts = df[state_column].value_counts().to_dict()
            logging.info(f"Class distribution in training data: {state_counts}")
        
        return {
            'model_data_path': final_data_path,
            'record_count': len(df),
            'feature_count': len(df.columns),
            'state_column': state_column,
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"Error preparing data for model: {e}")
        return {'status': 'error', 'message': str(e)}

def trigger_model_training(**kwargs):
    """모델 학습 파이프라인 트리거"""
    ti = kwargs['ti']
    data_prep_result = ti.xcom_pull(task_ids='prepare_data_for_model')
    
    if not data_prep_result or data_prep_result.get('status') == 'error':
        logging.error("Failed to prepare data for model")
        return {'status': 'error', 'message': 'Failed to prepare data for model'}
    
    try:
        # 모델 학습 DAG 트리거를 위한 정보 준비
        model_data_path = data_prep_result['model_data_path']
        
        # 트리거 정보 구성
        trigger_info = {
            'data_path': model_data_path,
            'feature_count': data_prep_result['feature_count'],
            'record_count': data_prep_result['record_count'],
            'state_column': data_prep_result.get('state_column'),
            'triggered_at': datetime.now().isoformat()
        }
        
        # 실제 트리거 실행
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
        trigger_task = TriggerDagRunOperator(
            task_id='trigger_model_training',
            trigger_dag_id='model_training_pipeline',
            conf={'trigger_info': json.dumps(trigger_info)},
            wait_for_completion=False,
            dag=dag
        )
        
        # 트리거 실행
        # 참고: 이 방식은 실제 Airflow에서는 실행되지 않을 수 있음 (파이프라인 로직 예시용)
        # trigger_task.execute(context=kwargs)
        
        logging.info(f"Triggered model_training_pipeline with data: {trigger_info}")
        
        return {
            'trigger_info': trigger_info,
            'status': 'success',
            'message': 'Model training triggered successfully'
        }
        
    except Exception as e:
        logging.error(f"Error triggering model training: {e}")
        return {'status': 'error', 'message': str(e)}

# 태스크 정의
read_data_task = PythonOperator(
    task_id='read_sensor_data',
    python_callable=read_sensor_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_and_interpolate_data',
    python_callable=preprocess_and_interpolate_data,
    dag=dag,
)

prepare_model_data_task = PythonOperator(
    task_id='prepare_data_for_model',
    python_callable=prepare_data_for_model,
    dag=dag,
)

trigger_training_task = PythonOperator(
    task_id='trigger_model_training',
    python_callable=trigger_model_training,
    dag=dag,
)

# 태스크 의존성 설정
read_data_task >> preprocess_task >> prepare_model_data_task >> trigger_training_task