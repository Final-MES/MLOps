from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.mysql_hook import MySqlHook
import pandas as pd
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
    'sql_to_csv_pipeline',
    default_args=default_args,
    description='MySQL 데이터를 CSV로 추출하는 파이프라인',
    schedule_interval=timedelta(days=1),  # 매일 실행
    catchup=False,
)

def extract_data_from_mysql(**kwargs):
    """MySQL 데이터베이스에서 데이터 추출"""
    try:
        # 추출할 테이블 이름
        table_name = kwargs.get('table_name', 'sensor_data')
        
        # MySQL 연결 (Airflow UI에서 connection 설정 필요)
        mysql_hook = MySqlHook(mysql_conn_id='mysql_connection')
        
        # 쿼리 실행
        query = f"SELECT * FROM {table_name}"
        
        # 데이터를 DataFrame으로 로드
        df = mysql_hook.get_pandas_df(query)
        
        logging.info(f"추출된 데이터 행 수: {len(df)}")
        
        # 저장 디렉토리 생성
        output_dir = '/tmp/extracted_data'
        os.makedirs(output_dir, exist_ok=True)
        
        # 현재 날짜/시간을 파일명에 포함
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file_path = f"{output_dir}/{table_name}_{timestamp}.csv"
        
        # CSV로 저장
        df.to_csv(csv_file_path, index=False)
        
        logging.info(f"데이터가 CSV 파일로 저장되었습니다: {csv_file_path}")
        
        return {
            'file_path': csv_file_path,
            'row_count': len(df),
            'columns': list(df.columns)
        }
        
    except Exception as e:
        logging.error(f"데이터 추출 중 오류 발생: {e}")
        raise

def process_extracted_csv(**kwargs):
    """추출된 CSV 파일 처리 (필요에 따라 추가 처리)"""
    ti = kwargs['ti']
    extract_result = ti.xcom_pull(task_ids='extract_data_from_mysql')
    
    file_path = extract_result['file_path']
    
    try:
        # CSV 파일 읽기
        df = pd.read_csv(file_path)
        
        # 데이터 기본 정보 출력
        logging.info(f"CSV 데이터 정보:")
        logging.info(f"- 행 수: {len(df)}")
        logging.info(f"- 열: {', '.join(df.columns)}")
        
        # 데이터 요약 통계
        # (대용량 데이터의 경우 비용이 많이 들 수 있으므로 필요한 경우 주석 해제)
        # logging.info(f"데이터 요약 통계:\n{df.describe()}")
        
        # 최종 CSV 파일 경로 (처리 후 파일을 별도로 저장하려면 여기서 수정)
        final_csv_path = file_path
        
        return {
            'processed_file_path': final_csv_path,
            'status': 'success'
        }
        
    except Exception as e:
        logging.error(f"CSV 처리 중 오류 발생: {e}")
        raise

# 다양한 테이블에 대해 작업을 생성할 수 있도록 함수 정의
def create_extract_task(table_name):
    return PythonOperator(
        task_id=f'extract_{table_name}_from_mysql',
        python_callable=extract_data_from_mysql,
        op_kwargs={'table_name': table_name},
        dag=dag,
    )

# 각 테이블에 대한 추출 태스크 정의
# 여러 테이블을 처리하려면 이 목록을 수정
tables_to_extract = ['sensor_data', 'equipment_data', 'quality_data']

extract_tasks = {}
process_tasks = {}

for table in tables_to_extract:
    # 추출 태스크
    extract_tasks[table] = PythonOperator(
        task_id=f'extract_{table}_from_mysql',
        python_callable=extract_data_from_mysql,
        op_kwargs={'table_name': table},
        dag=dag,
    )
    
    # 처리 태스크
    process_tasks[table] = PythonOperator(
        task_id=f'process_{table}_csv',
        python_callable=process_extracted_csv,
        dag=dag,
    )
    
    # 태스크 의존성 설정
    extract_tasks[table] >> process_tasks[table]