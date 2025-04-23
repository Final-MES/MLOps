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
        data_dir = Variable.get("sensor_data_dir", default_var="/app/data/raw")
    except:
        data_dir = "/app/data/raw"
        logging.info(f"Variable sensor_data_dir not found, using default path: {data_dir}")
    
    # 센서 ID 목록
    sensor_ids = ['sensor1', 'sensor2', 'sensor3', 'sensor4']
    
    # 각 센서별 데이터 로드
    sensor_data = {}
    
    for sensor_id in sensor_ids:
        try:
            # 가장 최근 파일 찾기 (실제 환경에서는 파일 명명 규칙에 맞게 수정 필요)
            sensor_files = [f for f in os.listdir(data_dir) if f.startswith(f"{sensor_id}_") and f.endswith(".csv")]
            if not sensor_files:
                logging.warning(f"No data files found for {sensor_id}")
                continue
                
            latest_file = max(sensor_files, key=lambda x: os.path.getmtime(os.path.join(data_dir, x)))
            file_path = os.path.join(data_dir, latest_file)
            
            # 데이터 로드
            df = pd.read_csv(file_path)
            
            # 시간 컬럼 확인 및 변환
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            elif 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'])
                df = df.drop('time', axis=1)
            else:
                logging.warning(f"No time column found in {sensor_id} data")
                continue
            
            sensor_data[sensor_id] = df
            logging.info(f"Loaded {len(df)} records from {file_path}")
            
        except Exception as e:
            logging.error(f"Error reading {sensor_id} data: {e}")
    
    # 센서 데이터 임시 저장
    os.makedirs('/tmp/sensor_data', exist_ok=True)
    sensor_data_info = {}
    
    for sensor_id, df in sensor_data.items():
        temp_path = f'/tmp/sensor_data/raw_{sensor_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(temp_path, index=False)
        sensor_data_info[sensor_id] = {
            'file_path': temp_path,
            'record_count': len(df),
            'start_time': df['timestamp'].min().isoformat(),
            'end_time': df['timestamp'].max().isoformat(),
            'columns': list(df.columns)
        }
    
    return {
        'sensor_data_info': sensor_data_info,
        'sensor_ids': list(sensor_data.keys())
    }

def interpolate_and_synchronize_data(**kwargs):
    """서로 다른 주기의 센서 데이터를 보간하고 동기화"""
    ti = kwargs['ti']
    data_info = ti.xcom_pull(task_ids='read_sensor_data')
    sensor_data_info = data_info['sensor_data_info']
    sensor_ids = data_info['sensor_ids']
    
    if not sensor_ids:
        logging.error("No sensor data available for interpolation")
        return {'status': 'error', 'message': 'No sensor data available'}
    
    try:
        # 각 센서 데이터 로드
        sensor_dataframes = {}
        for sensor_id in sensor_ids:
            file_path = sensor_data_info[sensor_id]['file_path']
            df = pd.read_csv(file_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            sensor_dataframes[sensor_id] = df
        
        # 센서 데이터 전처리기 초기화
        preprocessor = SensorDataPreprocessor(window_size=15)
        
        # 데이터 구조를 SensorDataPreprocessor.interpolate_sensor_data 메서드에 맞게 변환
        # 이 메서드는 'time' 컬럼을 기대하므로 'timestamp'를 'time'으로 변환
        for sensor_id, df in sensor_dataframes.items():
            df['time'] = df['timestamp'].astype(np.int64) // 10**9  # 타임스탬프를 초 단위로 변환
            sensor_dataframes[sensor_id] = df
        
        # 균일한 시간 간격으로 데이터 보간
        # 가장 세밀한 시간 간격 찾기 (기본값: 0.001초)
        min_intervals = []
        for sensor_id, df in sensor_dataframes.items():
            if len(df) > 1:
                times = df['time'].sort_values().values
                intervals = np.diff(times)
                min_interval = np.min(intervals[intervals > 0]) if any(intervals > 0) else 0.001
                min_intervals.append(min_interval)
        
        step = min(min_intervals) if min_intervals else 0.001
        logging.info(f"Using interpolation step: {step}")
        
        # 보간 수행
        interpolated_data = preprocessor.interpolate_sensor_data(
            sensor_dataframes,
            time_range=None,  # 자동 생성
            step=step,
            kind='linear'  # 선형 보간 사용
        )
        
        # 결과 저장
        interpolated_data_paths = {}
        for sensor_id, df in interpolated_data.items():
            output_path = f'/tmp/sensor_data/interpolated_{sensor_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(output_path, index=False)
            interpolated_data_paths[sensor_id] = output_path
        
        # 모든 센서 데이터를 통합하여 하나의 다변량 시계열 데이터셋 생성
        # 공통 타임스탬프(time 컬럼)를 기준으로 결합
        combined_df = None
        
        for sensor_id, df in interpolated_data.items():
            if combined_df is None:
                combined_df = df.copy()
                # 센서 ID를 컬럼명에 추가
                combined_df.columns = [f"{sensor_id}_{col}" if col != 'time' else col for col in combined_df.columns]
            else:
                # 'time' 컬럼을 제외한 나머지 컬럼만 추가
                temp_df = df.copy()
                temp_df.columns = [f"{sensor_id}_{col}" if col != 'time' else col for col in temp_df.columns]
                # 'time' 컬럼을 기준으로 결합
                combined_df = pd.merge(combined_df, temp_df, on='time', how='outer')
        
        # 결합된 데이터 저장
        combined_path = f'/tmp/sensor_data/combined_sensors_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        combined_df.to_csv(combined_path, index=False)
        
        logging.info(f"Successfully interpolated and combined data from {len(sensor_ids)} sensors")
        
        return {
            'interpolated_data_paths': interpolated_data_paths,
            'combined_data_path': combined_path,
            'record_count': len(combined_df),
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"Error in data interpolation: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def prepare_data_for_model(**kwargs):
    """보간된 데이터를 모델 학습용으로 준비"""
    ti = kwargs['ti']
    interp_result = ti.xcom_pull(task_ids='interpolate_and_synchronize_data')
    
    if interp_result['status'] != 'success':
        logging.error("Interpolation failed, cannot prepare data for model")
        return {'status': 'error', 'message': 'Previous step failed'}
    
    combined_data_path = interp_result['combined_data_path']
    
    try:
        # 결합된 데이터 로드
        df = pd.read_csv(combined_data_path)
        
        # 데이터에 라벨 추가 (실제 구현에서는 라벨 소스에 따라 달라질 수 있음)
        # 이 예제에서는 가정: 센서 데이터 파일명에 상태 정보가 포함되어 있다고 가정
        # 실제 구현에서는 라벨 소스(DB, 별도 파일 등)에 맞게 수정 필요
        
        # 데이터 상태 라벨 결정 (예시 - 실제로는 데이터 출처에 따라 수정 필요)
        if 'error_type' in df.columns:
            # 이미 라벨이 있는 경우
            pass
        else:
            # 파일 이름에서 상태 추출 (예시)
            file_name = os.path.basename(combined_data_path)
            if 'normal' in file_name.lower():
                df['state'] = 'normal'
            elif 'type1' in file_name.lower():
                df['state'] = 'type1'
            elif 'type2' in file_name.lower():
                df['state'] = 'type2'
            elif 'type3' in file_name.lower():
                df['state'] = 'type3'
            else:
                # 기본값 또는 다른 방법으로 라벨 부여
                df['state'] = 'unknown'
        
        # 모델 학습용 데이터 저장
        model_data_path = f'/tmp/sensor_data/model_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(model_data_path, index=False)
        
        # 최종 학습 데이터 경로 지정
        final_data_path = os.path.join('/app/data/processed', f'training_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # 최종 디렉토리 생성
        os.makedirs(os.path.dirname(final_data_path), exist_ok=True)
        
        # 최종 위치로 파일 복사
        import shutil
        shutil.copy2(model_data_path, final_data_path)
        
        logging.info(f"Prepared model data saved to {final_data_path}")
        
        return {
            'model_data_path': final_data_path,
            'record_count': len(df),
            'features': [col for col in df.columns if col != 'state' and col != 'time'],
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"Error preparing data for model: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

def trigger_model_training(**kwargs):
    """모델 학습 파이프라인 트리거"""
    ti = kwargs['ti']
    data_prep_result = ti.xcom_pull(task_ids='prepare_data_for_model')
    
    if data_prep_result['status'] != 'success':
        logging.error("Data preparation failed, cannot trigger model training")
        return {'status': 'error', 'message': 'Previous step failed'}
    
    try:
        from airflow.operators.trigger_dagrun import TriggerDagRunOperator
        
        # 모델 학습 DAG 트리거를 위한 정보 준비
        model_data_path = data_prep_result['model_data_path']
        
        # 트리거 정보 구성
        trigger_info = {
            'data_path': model_data_path,
            'feature_count': len(data_prep_result['features']),
            'record_count': data_prep_result['record_count'],
            'triggered_at': datetime.now().isoformat()
        }
        
        # 실제 환경에서는 여기서 TriggerDagRunOperator 사용
        # 이 예제에서는 로깅만 수행
        logging.info(f"Would trigger model_training_pipeline with data: {trigger_info}")
        
        return {
            'trigger_info': trigger_info,
            'status': 'success',
            'message': 'Model training triggered successfully'
        }
        
    except Exception as e:
        logging.error(f"Error triggering model training: {e}")
        return {
            'status': 'error',
            'message': str(e)
        }

# 태스크 정의
read_data_task = PythonOperator(
    task_id='read_sensor_data',
    python_callable=read_sensor_data,
    dag=dag,
)

interpolate_task = PythonOperator(
    task_id='interpolate_and_synchronize_data',
    python_callable=interpolate_and_synchronize_data,
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
read_data_task >> interpolate_task >> prepare_model_data_task >> trigger_training_task