from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from airflow import DAG
from airflow.operators.python import PythonOperator
import json
import requests
import time
import mlflow
import mlflow.pyfunc
import os
import pickle

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
    'model_deployment_pipeline',
    default_args=default_args,
    description='ML 모델 배포 파이프라인',
    schedule_interval=None,  # 수동 트리거 또는 다른 DAG에 의해 트리거됨
    catchup=False,
)

def get_production_model(**kwargs) -> Dict[str, Any]:
    """MLflow에서 프로덕션 모델 가져오기"""
    # DAG 트리거 시 전달된 모델 정보 가져오기
    model_name: str = "quality_prediction_model"
    model_version: Optional[str] = None
    
    try:
        dag_run_conf = kwargs.get('dag_run').conf
        if dag_run_conf and 'model_info' in dag_run_conf:
            model_info: Dict[str, Any] = json.loads(dag_run_conf['model_info'].replace("'", '"'))
            model_name = model_info['model_name']
            model_version = model_info['model_version']
        else:
            # 기본값 설정
            model_name = "quality_prediction_model"
            model_version = None  # 최신 프로덕션 버전 사용
    except Exception as e:
        print(f"모델 정보를 가져오는 중 오류 발생: {e}")
    
    # MLflow 설정
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    # 모델 가져오기
    if model_version:
        model_uri: str = f"models:/{model_name}/{model_version}"
    else:
        model_uri: str = f"models:/{model_name}/Production"
    
    # 모델 다운로드 경로
    os.makedirs('/tmp/deployed_models', exist_ok=True)
    local_path: str = f'/tmp/deployed_models/{model_name}_v{model_version if model_version else "latest"}'
    
    # 모델 다운로드
    model: Any = mlflow.pyfunc.load_model(model_uri)
    
    # 모델 저장
    with open(f'{local_path}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"모델을 다운로드했습니다: {model_name} 버전 {model_version if model_version else '최신'}")
    
    return {
        'model_name': model_name,
        'model_version': model_version,
        'local_path': f'{local_path}.pkl'
    }

def prepare_serving_environment(**kwargs) -> Dict[str, Any]:
    """모델 서빙 환경 준비"""
    ti = kwargs['ti']
    model_info: Dict[str, Any] = ti.xcom_pull(task_ids='get_production_model')
    
    # 실제 구현에서는 모델 서빙 환경 구성 (FastAPI, Flask 등)
    print(f"모델 서빙 환경을 준비 중입니다: {model_info['model_name']}")
    
    # 예시 설정 파일 생성
    config: Dict[str, Any] = {
        'model_path': model_info['local_path'],
        'model_name': model_info['model_name'],
        'model_version': model_info['model_version'],
        'api_port': 8000,
        'log_level': 'info',
        'batch_size': 32
    }
    
    config_path: str = '/tmp/deployed_models/serving_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"서빙 설정을 저장했습니다: {config_path}")
    
    return {
        'config_path': config_path,
        'api_port': 8000
    }

def deploy_model_service(**kwargs) -> Dict[str, Any]:
    """모델 서비스 배포"""
    ti = kwargs['ti']
    model_info: Dict[str, Any] = ti.xcom_pull(task_ids='get_production_model')
    serving_info: Dict[str, Any] = ti.xcom_pull(task_ids='prepare_serving_environment')
    
    # 실제 구현에서는 컨테이너 오케스트레이션 시스템에 배포 요청
    print(f"모델 서비스를 배포 중입니다: {model_info['model_name']}")
    
    # 배포 정보 및 상태 반환
    deployment_info: Dict[str, Any] = {
        'model_name': model_info['model_name'],
        'model_version': model_info['model_version'],
        'deployment_id': f"deployment-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        'status': 'in_progress',
        'api_endpoint': f"http://model-service:8000/predict",
        'health_endpoint': f"http://model-service:8000/health"
    }
    
    # 배포 정보 저장
    deployment_path: str = '/tmp/deployed_models/deployment_info.json'
    with open(deployment_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    # 실제 배포 대신 성공적인 배포 시뮬레이션
    time.sleep(5)  # 배포 시간 시뮬레이션
    
    # 배포 상태 업데이트
    deployment_info['status'] = 'completed'
    with open(deployment_path, 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    print(f"모델 서비스가 성공적으로 배포되었습니다: {deployment_info['deployment_id']}")
    
    return deployment_info

def validate_deployment(**kwargs) -> Dict[str, Any]:
    """배포 검증"""
    ti = kwargs['ti']
    deployment_info: Dict[str, Any] = ti.xcom_pull(task_ids='deploy_model_service')
    
    # 실제 구현에서는 API 엔드포인트 호출 및 응답 검증
    print(f"배포를 검증 중입니다: {deployment_info['deployment_id']}")
    
    # 테스트 데이터
    test_data: Dict[str, List[float]] = {
        'temperature': [45.0, 55.0],
        'vibration': [2.1, 1.5],
        'pressure': [95.0, 105.0],
        'run_time': [4800, 5200]
    }
    
    # 예측 API 호출 시뮬레이션
    # 실제 환경에서는 아래 주석을 해제하고 실제 API 호출
    """
    try:
        response: requests.Response = requests.post(
            deployment_info['api_endpoint'],
            json=test_data,
            timeout=10
        )
        response.raise_for_status()
        predictions: Dict[str, Any] = response.json()
        print(f"API 응답: {predictions}")
        validation_status: str = 'success'
    except Exception as e:
        print(f"API 호출 중 오류 발생: {e}")
        validation_status: str = 'failed'
    """
    
    # 시뮬레이션된 성공 응답
    validation_status: str = 'success'
    
    # 검증 결과 반환
    validation_result: Dict[str, Any] = {
        'deployment_id': deployment_info['deployment_id'],
        'validation_status': validation_status,
        'timestamp': datetime.now().isoformat()
    }
    
    print(f"검증 결과: {validation_status}")
    
    return validation_result

def configure_monitoring(**kwargs) -> Dict[str, str]:
    """모델 모니터링 설정"""
    ti = kwargs['ti']
    deployment_info: Dict[str, Any] = ti.xcom_pull(task_ids='deploy_model_service')
    validation_result: Dict[str, Any] = ti.xcom_pull(task_ids='validate_deployment')
    
    # 모니터링 설정 생성
    monitoring_config: Dict[str, Any] = {
        'deployment_id': deployment_info['deployment_id'],
        'model_name': deployment_info['model_name'],
        'model_version': deployment_info['model_version'],
        'metrics': [
            {'name': 'prediction_drift', 'threshold': 0.1, 'window_size': '1d'},
            {'name': 'data_drift', 'threshold': 0.2, 'window_size': '1d'},
            {'name': 'response_time', 'threshold': 200, 'window_size': '1h'}
        ],
        'alerts': {
            'slack_channel': '#model-alerts',
            'email': 'ml-team@example.com',
            'severity_levels': ['warning', 'critical']
        },
        'dashboard_url': 'http://grafana:3000/d/model-monitoring'
    }
    
    # 설정 저장
    config_path: str = '/tmp/deployed_models/monitoring_config.json'
    with open(config_path, 'w') as f:
        json.dump(monitoring_config, f, indent=2)
    
    print(f"모니터링 설정을 저장했습니다: {config_path}")
    
    # Prometheus 설정 업데이트 (실제 구현에서는 Prometheus 설정 파일 업데이트)
    print("Prometheus 대상을 업데이트했습니다.")
    
    # Grafana 대시보드 프로비저닝 (실제 구현에서는 Grafana API 호출)
    print("Grafana 대시보드를 프로비저닝했습니다.")
    
    return {
        'monitoring_config': config_path,
        'dashboard_url': monitoring_config['dashboard_url']
    }

def notify_deployment_status(**kwargs) -> str:
    """배포 상태 알림"""
    ti = kwargs['ti']
    deployment_info: Dict[str, Any] = ti.xcom_pull(task_ids='deploy_model_service')
    validation_result: Dict[str, Any] = ti.xcom_pull(task_ids='validate_deployment')
    monitoring_info: Dict[str, str] = ti.xcom_pull(task_ids='configure_monitoring')
    
    # 배포 요약 메시지 생성
    message: str = f"""
    모델 배포 완료
    
    모델: {deployment_info['model_name']} (버전 {deployment_info['model_version']})
    배포 ID: {deployment_info['deployment_id']}
    상태: {validation_result['validation_status']}
    API 엔드포인트: {deployment_info['api_endpoint']}
    모니터링 대시보드: {monitoring_info['dashboard_url']}
    
    배포 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    # 실제 구현에서는 알림 서비스 호출 (Slack, 이메일 등)
    print("배포 알림을 전송했습니다:")
    print(message)
    
    return "알림 전송 완료"

# 태스크 정의
get_model_task: PythonOperator = PythonOperator(
    task_id='get_production_model',
    python_callable=get_production_model,
    dag=dag,
)

prepare_env_task: PythonOperator = PythonOperator(
    task_id='prepare_serving_environment',
    python_callable=prepare_serving_environment,
    dag=dag,
)

deploy_task: PythonOperator = PythonOperator(
    task_id='deploy_model_service',
    python_callable=deploy_model_service,
    dag=dag,
)

validate_task: PythonOperator = PythonOperator(
    task_id='validate_deployment',
    python_callable=validate_deployment,
    dag=dag,
)

monitoring_task: PythonOperator = PythonOperator(
    task_id='configure_monitoring',
    python_callable=configure_monitoring,
    dag=dag,
)

notify_task: PythonOperator = PythonOperator(
    task_id='notify_deployment_status',
    python_callable=notify_deployment_status,
    dag=dag,
)

# 태스크 의존성 설정
get_model_task >> prepare_env_task >> deploy_task >> validate_task >> monitoring_task >> notify_task