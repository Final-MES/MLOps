from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
import pandas as pd
import numpy as np
import json
import pickle
import os
import mlflow
import requests
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io

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
    'model_monitoring_pipeline',
    default_args=default_args,
    description='ML 모델 모니터링 파이프라인',
    schedule_interval=timedelta(hours=6),
    catchup=False,
)

def get_deployment_info(**kwargs):
    """현재 배포된 모델 정보 가져오기"""
    # 실제 구현에서는 모델 레지스트리나 배포 시스템에서 정보 가져오기
    
    # 여기def get_deployment_info(**kwargs):
    """현재 배포된 모델 정보 가져오기"""
    # 실제 구현에서는 모델 레지스트리나 배포 시스템에서 정보 가져오기
    
    # 임시 배포 정보 파일에서 정보 읽기
    deployment_path = '/tmp/deployed_models/deployment_info.json'
    
    try:
        with open(deployment_path, 'r') as f:
            deployment_info = json.load(f)
    except FileNotFoundError:
        # 파일이 없는 경우 기본값 설정
        deployment_info = {
            'model_name': 'quality_prediction_model',
            'model_version': 'latest',
            'deployment_id': 'default-deployment',
            'api_endpoint': 'http://model-service:8000/predict',
            'health_endpoint': 'http://model-service:8000/health'
        }
    
    print(f"배포 정보를 가져왔습니다: {deployment_info['deployment_id']}")
    
    return deployment_info

def collect_production_data(**kwargs):
    """프로덕션 환경에서 새로운 데이터 수집"""
    # 실제 구현에서는 데이터베이스에서 최근 데이터 수집
    
    # 예시 데이터 생성
    np.random.seed(int(datetime.now().timestamp()) % 100000)
    data_size = 200
    
    # 조금 다른 분포를 가진 데이터 생성 (드리프트 시뮬레이션)
    data = {
        'temperature': np.random.normal(55, 16, data_size),  # 평균값이 약간 높아짐
        'vibration': np.random.normal(2.2, 1.2, data_size),  # 분산이 약간 커짐
        'pressure': np.random.normal(100, 20, data_size),
        'run_time': np.random.normal(5100, 950, data_size),
    }
    
    # 타겟 추가 (실제 프로덕션 환경에서는 지연됨)
    # 약간 달라진 상관관계 적용
    data['target_quality'] = 85 - 0.18 * data['temperature'] + 0.32 * data['pressure'] - 0.13 * data['vibration'] + np.random.normal(0, 3, data_size)
    
    df = pd.DataFrame(data)
    
    # 데이터 저장
    os.makedirs('/tmp/monitoring_data', exist_ok=True)
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_path = f'/tmp/monitoring_data/production_data_{current_time}.csv'
    df.to_csv(file_path, index=False)
    
    print(f"프로덕션 데이터를 수집했습니다: {file_path} ({len(df)} 레코드)")
    
    return file_path

def detect_data_drift(**kwargs):
    """데이터 드리프트 감지"""
    ti = kwargs['ti']
    prod_data_path = ti.xcom_pull(task_ids='collect_production_data')
    
    # 새 데이터 로드
    new_data = pd.read_csv(prod_data_path)
    
    # 참조 데이터 로드 (학습 데이터)
    # 실제 구현에서는 저장된 참조 데이터 통계 사용
    try:
        reference_stats_path = '/tmp/ml_data/reference_data_stats.json'
        with open(reference_stats_path, 'r') as f:
            reference_stats = json.load(f)
    except FileNotFoundError:
        # 참조 통계가 없으면 기본값 설정
        reference_stats = {
            'temperature': {'mean': 50, 'std': 15},
            'vibration': {'mean': 2, 'std': 1},
            'pressure': {'mean': 100, 'std': 20},
            'run_time': {'mean': 5000, 'std': 1000}
        }
    
    # 각 특성에 대한 드리프트 계산
    drift_metrics = {}
    for feature in ['temperature', 'vibration', 'pressure', 'run_time']:
        new_mean = new_data[feature].mean()
        new_std = new_data[feature].std()
        
        # 평균의 상대적 변화
        mean_drift = abs((new_mean - reference_stats[feature]['mean']) / reference_stats[feature]['mean'])
        # 표준편차의 상대적 변화
        std_drift = abs((new_std - reference_stats[feature]['std']) / reference_stats[feature]['std'])
        
        drift_metrics[feature] = {
            'mean_drift': mean_drift,
            'std_drift': std_drift,
            'is_significant': mean_drift > 0.1 or std_drift > 0.15  # 임계값 설정
        }
    
    # 전체 드리프트 상태 판단
    has_drift = any(m['is_significant'] for m in drift_metrics.values())
    
    # 시각화 (실제 Airflow에서는 작동하지 않을 수 있음 - 참고용)
    # 실제 구현에서는 이미지를 파일로 저장하거나 다른 방식 사용
    """
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(['temperature', 'vibration', 'pressure', 'run_time']):
        plt.subplot(2, 2, i+1)
        plt.hist(new_data[feature], bins=30, alpha=0.7)
        plt.axvline(reference_stats[feature]['mean'], color='red', linestyle='--', 
                   label=f"Reference mean: {reference_stats[feature]['mean']:.2f}")
        plt.axvline(new_data[feature].mean(), color='green', linestyle='-', 
                   label=f"New mean: {new_data[feature].mean():.2f}")
        plt.title(f"{feature} Distribution (Drift: {drift_metrics[feature]['mean_drift']:.2%})")
        plt.legend()
    
    plt.tight_layout()
    
    # 이미지를 바이트로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    
    # 이미지 저장
    drift_plot_path = '/tmp/monitoring_data/drift_plot.png'
    with open(drift_plot_path, 'wb') as f:
        f.write(buf.getvalue())
    plt.close()
    """
    
    # 드리프트 메트릭 저장
    drift_metrics_path = f'/tmp/monitoring_data/drift_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(drift_metrics_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'metrics': drift_metrics,
            'has_significant_drift': has_drift
        }, f, indent=2)
    
    print(f"데이터 드리프트 분석 완료: {'드리프트 감지됨' if has_drift else '드리프트 없음'}")
    
    return {
        'drift_metrics_path': drift_metrics_path,
        'has_drift': has_drift,
        'drift_features': [f for f, m in drift_metrics.items() if m['is_significant']]
    }

def evaluate_model_performance(**kwargs):
    """모델 성능 평가"""
    ti = kwargs['ti']
    prod_data_path = ti.xcom_pull(task_ids='collect_production_data')
    deployment_info = ti.xcom_pull(task_ids='get_deployment_info')
    
    # 모델 로드
    model_path = f"/tmp/deployed_models/{deployment_info['model_name']}_v{deployment_info['model_version'] if deployment_info.get('model_version') else 'latest'}.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print(f"모델 파일을 찾을 수 없습니다: {model_path}")
        # MLflow에서 다시 로드 시도
        mlflow.set_tracking_uri("http://mlflow:5000")
        try:
            model = mlflow.pyfunc.load_model(f"models:/{deployment_info['model_name']}/Production")
        except Exception as e:
            print(f"MLflow에서 모델을 로드하는 데 실패했습니다: {e}")
            return {
                'status': 'failed',
                'reason': f"모델을 로드할 수 없습니다: {e}"
            }
    
    # 데이터 로드
    data = pd.read_csv(prod_data_path)
    
    # 특성과 타겟 분리
    X = data.drop('target_quality', axis=1)
    y = data['target_quality']
    
    # 모델 예측
    y_pred = model.predict(X)
    
    # 성능 지표 계산
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    # 성능 임계값 설정
    performance_threshold = 0.7  # R² > 0.7
    is_performance_acceptable = r2 > performance_threshold
    
    # 결과 저장
    performance_path = f'/tmp/monitoring_data/model_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(performance_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_name': deployment_info['model_name'],
            'model_version': deployment_info.get('model_version', 'latest'),
            'metrics': {
                'mse': mse,
                'r2': r2
            },
            'threshold': performance_threshold,
            'is_acceptable': is_performance_acceptable
        }, f, indent=2)
    
    print(f"모델 성능 평가 완료: MSE = {mse:.4f}, R² = {r2:.4f}")
    print(f"성능 상태: {'허용 가능' if is_performance_acceptable else '허용 불가'}")
    
    return {
        'performance_path': performance_path,
        'metrics': {'mse': mse, 'r2': r2},
        'is_acceptable': is_performance_acceptable
    }

def send_monitoring_alert(**kwargs):
    """모니터링 알림 전송"""
    ti = kwargs['ti']
    drift_result = ti.xcom_pull(task_ids='detect_data_drift')
    performance_result = ti.xcom_pull(task_ids='evaluate_model_performance')
    
    # 알림이 필요한지 확인
    needs_alert = drift_result.get('has_drift', False) or not performance_result.get('is_acceptable', True)
    
    if not needs_alert:
        print("알림이 필요하지 않습니다. 모든 지표가 정상 범위 내에 있습니다.")
        return "알림 필요 없음"
    
    # 알림 메시지 생성
    alert_message = {
        'timestamp': datetime.now().isoformat(),
        'severity': 'high' if not performance_result.get('is_acceptable', True) else 'medium',
        'title': '모델 모니터링 알림',
        'message': []
    }
    
    if drift_result.get('has_drift', False):
        drift_features = ', '.join(drift_result.get('drift_features', []))
        alert_message['message'].append(f"데이터 드리프트가 감지되었습니다. 영향받은 특성: {drift_features}")
    
    if not performance_result.get('is_acceptable', True):
        metrics = performance_result.get('metrics', {})
        alert_message['message'].append(
            f"모델 성능이 임계값 아래로 떨어졌습니다. R² = {metrics.get('r2', 0):.4f}, MSE = {metrics.get('mse', 0):.4f}"
        )
    
    alert_message['message'] = '\n'.join(alert_message['message'])
    
    # 알림 저장
    alert_path = f'/tmp/monitoring_data/alert_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(alert_path, 'w') as f:
        json.dump(alert_message, f, indent=2)
    
    # 실제 알림 전송 (여기서는 로그로만 출력)
    print("======= 모니터링 알림 =======")
    print(f"심각도: {alert_message['severity']}")
    print(f"제목: {alert_message['title']}")
    print(f"메시지: {alert_message['message']}")
    print("=============================")
    
    # 알림 정보 반환
    return {
        'alert_path': alert_path,
        'severity': alert_message['severity'],
        'needs_retraining': not performance_result.get('is_acceptable', True) or drift_result.get('has_drift', False)
    }

def trigger_retraining(**kwargs):
    """필요한 경우 재학습 트리거"""
    ti = kwargs['ti']
    alert_result = ti.xcom_pull(task_ids='send_monitoring_alert')
    
    # 재학습이 필요한지 확인
    needs_retraining = alert_result.get('needs_retraining', False)
    
    if not needs_retraining:
        print("재학습이 필요하지 않습니다.")
        return "재학습 필요 없음"
    
    print("모델 재학습이 필요합니다. 학습 파이프라인을 트리거합니다.")
    
    # 실제 구현에서는 여기서 모델 학습 DAG 트리거
    # 여기서는 시뮬레이션만 수행
    retraining_request = {
        'timestamp': datetime.now().isoformat(),
        'reason': 'Performance degradation or data drift detected',
        'alert_severity': alert_result.get('severity', 'medium'),
        'requested_by': 'monitoring_pipeline'
    }
    
    # 재학습 요청 저장
    retraining_path = f'/tmp/monitoring_data/retraining_request_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(retraining_path, 'w') as f:
        json.dump(retraining_request, f, indent=2)
    
    return {
        'retraining_path': retraining_path,
        'status': 'requested'
    }

# 태스크 정의
get_deployment_task = PythonOperator(
    task_id='get_deployment_info',
    python_callable=get_deployment_info,
    dag=dag,
)

collect_data_task = PythonOperator(
    task_id='collect_production_data',
    python_callable=collect_production_data,
    dag=dag,
)

drift_task = PythonOperator(
    task_id='detect_data_drift',
    python_callable=detect_data_drift,
    dag=dag,
)

performance_task = PythonOperator(
    task_id='evaluate_model_performance',
    python_callable=evaluate_model_performance,
    dag=dag,
)

alert_task = PythonOperator(
    task_id='send_monitoring_alert',
    python_callable=send_monitoring_alert,
    dag=dag,
)

retraining_task = PythonOperator(
    task_id='trigger_retraining',
    python_callable=trigger_retraining,
    dag=dag,
)

trigger_training_dag = TriggerDagRunOperator(
    task_id='trigger_model_training_dag',
    trigger_dag_id='model_training_pipeline',
    conf={'triggered_by': 'monitoring_pipeline'},
    dag=dag,
)

# 태스크 의존성 설정
get_deployment_task >> collect_data_task
collect_data_task >> [drift_task, performance_task]
[drift_task, performance_task] >> alert_task
alert_task >> retraining_task
retraining_task >> trigger_training_dag