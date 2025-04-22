from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.mysql.operators.mysql import MySqlOperator
from airflow.models import Variable
import pandas as pd
import json
import os
import logging

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
    'data_collection_pipeline',
    default_args=default_args,
    description='공장 센서 데이터 수집 및 처리 파이프라인',
    schedule_interval=timedelta(hours=1),
    catchup=False,
)

def read_sensor_data(**kwargs):
    """실제 센서 데이터 파일에서 데이터 읽기"""
    # Airflow Variable에서 데이터 경로 가져오기 (없으면 기본값 사용)
    # 추후 실행 시 Airflow UI에서 Variable을 설정할 수 있음
    try:
        data_path = Variable.get("sensor_data_path", default_var="/path/to/sensor_data.csv")
    except:
        data_path = "/path/to/sensor_data.csv"
        logging.info(f"Variable sensor_data_path not found, using default path: {data_path}")
    
    logging.info(f"Reading sensor data from: {data_path}")
    
    try:
        # 실제 파일 확장자에 따라 적절한 판다스 함수 사용
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx') or data_path.endswith('.xls'):
            df = pd.read_excel(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        logging.info(f"Successfully read {len(df)} records from {data_path}")
        
        # 임시 저장 경로
        os.makedirs('/tmp/sensor_data', exist_ok=True)
        temp_path = f'/tmp/sensor_data/processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(temp_path, index=False)
        
        return {
            'file_path': temp_path,
            'record_count': len(df),
            'columns': list(df.columns)
        }
        
    except Exception as e:
        logging.error(f"Error reading sensor data: {e}")
        raise

def validate_sensor_data(**kwargs):
    """센서 데이터 유효성 검증"""
    ti = kwargs['ti']
    data_info = ti.xcom_pull(task_ids='read_sensor_data')
    file_path = data_info['file_path']
    
    try:
        df = pd.read_csv(file_path)
        
        # 필수 컬럼 확인 (실제 데이터에 맞게 수정 필요)
        required_columns = ['timestamp', 'sensor_id', 'value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logging.warning(f"Missing required columns: {missing_columns}")
            # 필요에 따라 더미 컬럼 추가 또는 오류 처리
        
        # 데이터 타입 변환 및 정리
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # 타임스탬프가 없는 행 제거
            invalid_timestamp = df['timestamp'].isna().sum()
            if invalid_timestamp > 0:
                logging.warning(f"Removed {invalid_timestamp} rows with invalid timestamps")
                df = df.dropna(subset=['timestamp'])
        
        # 숫자형 컬럼의 이상치 확인
        numeric_columns = df.select_dtypes(include=['number']).columns
        outliers_count = 0
        
        for col in numeric_columns:
            if col != 'sensor_id':  # ID 컬럼은 제외
                # 간단한 IQR 기반 이상치 감지
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outliers_count += len(outliers)
                
                if len(outliers) > 0:
                    logging.info(f"Found {len(outliers)} outliers in column {col}")
        
        # 검증 결과 저장
        validated_path = file_path.replace('.csv', '_validated.csv')
        df.to_csv(validated_path, index=False)
        
        validation_info = {
            'file_path': validated_path,
            'original_count': data_info['record_count'],
            'validated_count': len(df),
            'outliers_count': outliers_count
        }
        
        logging.info(f"Data validation complete: {validation_info}")
        return validation_info
        
    except Exception as e:
        logging.error(f"Error validating sensor data: {e}")
        raise

def transform_sensor_data(**kwargs):
    """센서 데이터 변환 및 가공"""
    ti = kwargs['ti']
    validation_info = ti.xcom_pull(task_ids='validate_sensor_data')
    file_path = validation_info['file_path']
    
    try:
        df = pd.read_csv(file_path)
        
        # 타임스탬프가 있는 경우 시간대별 집계
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['date'] = df['timestamp'].dt.date
            
            # 시간대별 집계 (센서별, 일자별 평균)
            if 'sensor_id' in df.columns and 'value' in df.columns:
                agg_df = df.groupby(['date', 'sensor_id', 'hour']).agg({
                    'value': ['mean', 'min', 'max', 'std', 'count']
                }).reset_index()
                
                # 다중 인덱스 컬럼 이름 평평하게 만들기
                agg_df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in agg_df.columns.values]
                
                # 집계 데이터 저장
                agg_path = file_path.replace('_validated.csv', '_aggregated.csv')
                agg_df.to_csv(agg_path, index=False)
                
                logging.info(f"Created aggregated data with {len(agg_df)} records")
        
        # 변환된 원본 데이터 저장
        transformed_path = file_path.replace('_validated.csv', '_transformed.csv')
        df.to_csv(transformed_path, index=False)
        
        return {
            'transformed_path': transformed_path,
            'aggregated_path': agg_path if 'agg_path' in locals() else None,
            'record_count': len(df)
        }
        
    except Exception as e:
        logging.error(f"Error transforming sensor data: {e}")
        raise

def load_data_to_database(**kwargs):
    """처리된 데이터를 데이터베이스에 로드"""
    ti = kwargs['ti']
    transform_info = ti.xcom_pull(task_ids='transform_sensor_data')
    transformed_path = transform_info['transformed_path']
    
    try:
        # 데이터베이스 연결 정보 (MySQL 예시)
        # 실제 구현 시 Airflow Connections에서 가져오는 것이 좋음
        conn_config = {
            'host': 'mysql',
            'port': 3306,
            'user': 'mlops_user',
            'password': 'mlops_password',
            'database': 'mlops'
        }
        
        # 이 부분은 실제 DB 연결 시 구현
        # 여기서는 로깅만 수행
        logging.info(f"Would load data from {transformed_path} to database")
        logging.info(f"Connection config: {conn_config}")
        
        # 실제 MySQL 로드 예시 (주석 처리)
        """
        import mysql.connector
        
        conn = mysql.connector.connect(**conn_config)
        cursor = conn.cursor()
        
        # 테이블 생성 (없는 경우)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_data (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp DATETIME,
                sensor_id VARCHAR(50),
                value FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 데이터 로드
        df = pd.read_csv(transformed_path)
        
        # 레코드 삽입
        for _, row in df.iterrows():
            cursor.execute('''
                INSERT INTO sensor_data (timestamp, sensor_id, value)
                VALUES (%s, %s, %s)
            ''', (
                row['timestamp'],
                row['sensor_id'],
                row['value']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        """
        
        return {
            'status': 'success',
            'message': f"Successfully simulated loading {transform_info['record_count']} records to database",
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error loading data to database: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }

# 태스크 정의
read_data_task = PythonOperator(
    task_id='read_sensor_data',
    python_callable=read_sensor_data,
    dag=dag,
)

validate_data_task = PythonOperator(
    task_id='validate_sensor_data',
    python_callable=validate_sensor_data,
    dag=dag,
)

transform_data_task = PythonOperator(
    task_id='transform_sensor_data',
    python_callable=transform_sensor_data,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data_to_database',
    python_callable=load_data_to_database,
    dag=dag,
)

# 태스크 의존성 설정
read_data_task >> validate_data_task >> transform_data_task >> load_data_task