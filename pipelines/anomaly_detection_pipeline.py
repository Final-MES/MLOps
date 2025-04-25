import logging
import argparse
import pandas as pd
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
import os

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# paths.py 유틸리티 임포트
from src.utils.paths import get_project_paths, get_data_path, get_model_path, ensure_dir

# 프로젝트 모듈 임포트
from src.data.preprocessor import SensorDataPreprocessor  # 데이터 전처리용 클래스
from src.models.sensor.multivariate_model import prepare_multivariate_data, train_multivariate_model, MultivariateLSTMClassifier  # 모델 학습 관련 함수와 클래스
from src.models.evaluation import evaluate_multivariate_model, analyze_misclassifications, feature_importance_analysis  # 모델 평가 관련 함수

def main():
    """다변량 시계열 분류 모델을 사용한 이상 감지 파이프라인 실행 예제"""
    
    parser = argparse.ArgumentParser(description='다변량 시계열 분류 모델을 사용한 이상 감지')
    parser.add_argument('--mode', type=str, choices=['preprocess', 'train', 'evaluate', 'infer'], 
                      default='train', help='실행 모드')
    parser.add_argument('--data_dir', type=str, default='raw', help='원시 데이터 서브디렉토리')
    parser.add_argument('--processed_dir', type=str, default='processed', help='처리된 데이터 서브디렉토리')
    parser.add_argument('--sequence_length', type=int, default=50, help='시계열 시퀀스 길이')
    args = parser.parse_args()
    
    # 프로젝트 경로 설정
    project_paths = get_project_paths()
    
    # 필요한 디렉토리 생성 (Path 객체 사용)
    data_raw_path = ensure_dir(get_data_path(args.data_dir))
    data_processed_path = ensure_dir(get_data_path(args.processed_dir))
    model_path = ensure_dir(project_paths["models"])
    plot_path = ensure_dir(project_paths["root"] / "plots")
    
    # 경로 정보 업데이트
    args.data_dir = data_raw_path
    args.processed_dir = data_processed_path
    args.model_dir = model_path
    args.plot_dir = plot_path
    
    if args.mode == 'preprocess':
        preprocess_sensor_data(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'evaluate':
        evaluate_model(args)
    elif args.mode == 'infer':
        perform_inference(args)

def preprocess_sensor_data(args):
    """센서 데이터 전처리 및 통합"""
    logger.info("센서 데이터 전처리 시작")
    
    # 통합 데이터 파일 찾기
    import glob
    data_files = glob.glob(os.path.join(args.data_dir, "*.csv"))
    
    if not data_files:
        logger.error(f"데이터 디렉토리{args.data_dir}에 CSV 파일이 없습니다.")
        return None, None
    
    # 가장 최근 파일 사용 (또는 다른 선택 기준 사용 가능)
    latest_file = max(data_files, key=os.path.getctime)
    logger.info(f"처리할 데이터 파일: {latest_file}")
    
    try:
        # 데이터 로드
        df = pd.read_csv(latest_file, names =["time","normal","type1","type2","type3"])
        logger.info(f"데이터 로드 완료: {len(df)} 행, {len(df.columns)} 열")
        
        # 시간 컬럼 확인 및 처리
        time_column = None
        for col in ['timestamp', 'time', 'datetime']:
            if col in df.columns:
                time_column = col
                # 시간 컬럼 형식이 문자열인 경우 datetime으로 변환
                if df[col].dtype == object:
                    df[col] = pd.to_datetime(df[col])
                break
        
        # 시간 컬럼이 없으면 인덱스 기반 시간 생성
        if time_column is None:
            logger.warning("시간 컬럼을 찾을 수 없어 인덱스 기반 시간을 생성합니다.")
            df['time'] = pd.to_datetime(pd.date_range(start='now', periods=len(df), freq='S'))
            time_column = 'time'
        
        # 상태 컬럼 확인
        state_column = None
        for col in ['state', 'status', 'condition', 'label', 'class']:
            if col in df.columns:
                state_column = col
                break
        
        if state_column is None:
            logger.warning("상태 컬럼을 찾을 수 없습니다.")
            return None, None
        
        # 상태 분포 확인
        state_counts = df[state_column].value_counts()
        logger.info(f"상태 분포: {state_counts.to_dict()}")
        
        # 전처리기 초기화
        preprocessor = SensorDataPreprocessor(window_size=15)
        
        # 결측치 처리
        logger.info("결측치 처리 중...")
        df_cleaned = preprocessor.handle_missing_values(df)
        
        # 이상치 처리
        logger.info("이상치 처리 중...")
        # 상태 및 시간 컬럼 제외
        exclude_cols = [col for col in [time_column, state_column] if col is not None]
        df_cleaned = preprocessor.handle_outliers(df_cleaned, exclude_columns=exclude_cols)
        
        # 시간 컬럼 초 단위 변환 (보간을 위해)
        df_cleaned['time_seconds'] = df_cleaned[time_column].astype(np.int64) // 10**9
        
        # 특성 컬럼 (시간과 상태 컬럼 제외)
        feature_cols = [col for col in df_cleaned.columns if col not in [time_column, state_column, 'time_seconds']]
        
        # 각 상태별로 나누지 않고 전체 데이터를 한 번에 보간
        sensor_data = {
            'combined_sensor': df_cleaned[['time_seconds'] + feature_cols].rename(columns={'time_seconds': 'time'})
        }
        
        # 보간 간격 계산 (초 단위)
        times = sorted(df_cleaned['time_seconds'].values)
        intervals = np.diff(times)
        min_interval = np.min(intervals[intervals > 0]) if any(intervals > 0) else 0.001
        logger.info(f"보간 간격: {min_interval}초")
        
        # 전체 데이터 보간
        logger.info("데이터 보간 중...")
        interpolated_data = preprocessor.interpolate_sensor_data(
            sensor_data,
            time_range=None,  # 자동 생성
            step=min_interval,
            kind='linear'  # 선형 보간 사용
        )
        
        # 보간된 데이터 가져오기
        interpolated_df = interpolated_data['combined_sensor']
        
        # 원래 시간 형식으로 변환
        interpolated_df['timestamp'] = pd.to_datetime(interpolated_df['time'], unit='s')
        
        # 상태 정보 추가 (가장 가까운 시간의 상태로 보간)
        state_df = df_cleaned[[time_column, state_column]].copy()
        state_df['time_seconds'] = state_df[time_column].astype(np.int64) // 10**9
        
        # 가장 가까운 시간의 상태 찾기
        from scipy.spatial.distance import cdist
        
        # 두 시간 배열 간의 거리 계산
        distances = cdist(
            interpolated_df['time'].values.reshape(-1, 1),
            state_df['time_seconds'].values.reshape(-1, 1)
        )
        
        # o 행에 대한 최소 거리 인덱스 찾기
        closest_indices = np.argmin(distances, axis=1)
        
        # 보간된 데이터에 상태 추가
        interpolated_df[state_column] = state_df[state_column].iloc[closest_indices].values
        
        # 처리된 데이터 저장
        os.makedirs(args.processed_dir, exist_ok=True)
        output_path = os.path.join(args.processed_dir, f"processed_sensor_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        interpolated_df.to_csv(output_path, index=False)
        logger.info(f"처리된 데이터 저장 완료: {output_path} (총 {len(interpolated_df)} 행)")
        
        # 추가 특성 추출
        logger.info("추가 특성 추출 중...")
        feature_df = preprocessor.extract_statistical_moments(interpolated_df, columns=feature_cols)
        feature_df = preprocessor.extract_frequency_features(feature_df, columns=feature_cols)
        
        # 특성이 추가된 데이터 저장
        feature_output_path = os.path.join(args.processed_dir, f"processed_sensor_data_with_features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        feature_df.to_csv(feature_output_path, index=False)
        logger.info(f"특성이 추가된 데이터 저장 완료: {feature_output_path} (총 {len(feature_df)} 행, {len(feature_df.columns)} 열)")
        
        return output_path, feature_output_path
        
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None
    
def train_model(args):
    """다변량 시계열 분류 모델 학습"""
    # 처리된 데이터 파일 찾기
    processed_files = list(args.processed_dir.glob("multivariate_sensor_data_with_features_*.csv"))
    
    if not processed_files:
        logger.error("처리된 데이터 파일을 찾을 수 없습니다. 먼저 preprocess 모드를 실행하세요.")
        return
    
    # 가장 최근 파일 사용
    data_path = max(processed_files, key=lambda p: p.stat().st_ctime)
    logger.info(f"학습에 사용할 데이터 파일: {data_path}")
    
    # 데이터 준비
    train_loader, val_loader, test_loader, data_info = prepare_multivariate_data(
        data_path=str(data_path),  # API가 문자열을 요구할 경우 변환
        state_column='state',
        time_column='time',
        feature_cols=None,  # 자동으로 모든 특성 사용
        sequence_length=args.sequence_length,
        test_size=0.2,
        val_size=0.2
    )
    
    # 데이터 통계 출력
    logger.info(f"데이터 정보:")
    logger.info(f"- 특성 수: {data_info['input_size']}")
    logger.info(f"- 클래스 수: {data_info['num_classes']}")
    logger.info(f"- 시퀀스 길이: {data_info['sequence_length']}")
    logger.info(f"- 학습 데이터: {data_info['train_size']} 샘플")
    logger.info(f"- 검증 데이터: {data_info['val_size']} 샘플")
    logger.info(f"- 테스트 데이터: {data_info['test_size']} 샘플")
    
    # 모델 파일 경로 생성
    model_filename = "multivariate_lstm_classifier"
    model_file_path = get_model_path(model_filename)
    
    # 모델 학습
    model, history = train_multivariate_model(
        train_loader=train_loader,
        val_loader=val_loader,
        data_info=data_info,
        hidden_size=128,
        num_layers=2,
        learning_rate=0.001,
        epochs=100,
        patience=10,
        model_dir=str(args.model_dir),  # 필요시 문자열로 변환
        model_name=model_filename
    )
    
    # 학습 이력 시각화
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plot_file = args.plot_dir / 'training_history.png'
    plt.savefig(plot_file)
    logger.info(f"학습 이력 시각화 저장: {plot_file}")
    
    # 모델 평가
    evaluation_result = evaluate_multivariate_model(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        plot=True,
        save_plot=True,
        plot_dir=str(args.plot_dir)  # 필요시 문자열로 변환
    )
    
    # 평가 결과 저장
    eval_path = args.model_dir / 'evaluation_results.json'
    with open(eval_path, 'w') as f:
        json.dump(evaluation_result, f, indent=2)
    
    logger.info(f"평가 결과 저장: {eval_path}")
    logger.info(f"테스트 정확도: {evaluation_result['accuracy']:.4f}")
    
    return model, data_info

def evaluate_model(args):
    """저장된 모델을 평가하고 분석"""
    # 모델 파일 찾기
    model_filename = "multivariate_lstm_classifier"
    model_path = get_model_path(model_filename)
    model_info_path = args.model_dir / f"{model_filename}_info.json"
    
    if not model_path.exists() or not model_info_path.exists():
        logger.error("모델 파일을 찾을 수 없습니다. 먼저 train 모드를 실행하세요.")
        return
    
    # 모델 정보 로드
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # 처리된 데이터 파일 찾기
    processed_files = list(args.processed_dir.glob("multivariate_sensor_data_with_features_*.csv"))
    
    if not processed_files:
        logger.error("처리된 데이터 파일을 찾을 수 없습니다.")
        return
    
    # 가장 최근 파일 사용
    data_path = max(processed_files, key=lambda p: p.stat().st_ctime)
    logger.info(f"평가에 사용할 데이터 파일: {data_path}")
    
    # 데이터 준비
    _, _, test_loader, data_info = prepare_multivariate_data(
        data_path=str(data_path),  # 문자열로 변환
        state_column='state',
        time_column='time',
        feature_cols=None,  # 자동으로 모든 특성 사용
        sequence_length=model_info['sequence_length'],
        test_size=0.2,
        val_size=0.2
    )
    
    # 모델 초기화
    model = MultivariateLSTMClassifier(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        num_classes=model_info['num_classes']
    )
    
    # 모델 가중치 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 모델 평가
    evaluation_result = evaluate_multivariate_model(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        plot=True,
        save_plot=True,
        plot_dir=str(args.plot_dir)  # 문자열로 변환
    )
    
    # 오분류 분석
    misclassification_result = analyze_misclassifications(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        max_samples=5,
        plot_dir=str(args.plot_dir)  # 문자열로 변환
    )
    
    # 특성 중요도 분석
    feature_importance_result = feature_importance_analysis(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        feature_names=data_info['feature_cols'][:20],  # 처음 20개 특성만 이름 사용
        plot=True,
        plot_dir=str(args.plot_dir)  # 문자열로 변환
    )
    
    # 종합 분석 결과 저장
    analysis_result = {
        'evaluation': evaluation_result,
        'feature_importance': feature_importance_result
    }
    
    analysis_path = args.model_dir / 'model_analysis_results.json'
    with open(analysis_path, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    logger.info(f"종합 분석 결과 저장: {analysis_path}")
    
    return analysis_result

def perform_inference(args):
    """실시간 데이터에 대한 추론 수행 예시"""
    # 모델 파일 찾기
    model_filename = "multivariate_lstm_classifier"
    model_path = get_model_path(model_filename)
    model_info_path = args.model_dir / f"{model_filename}_info.json"
    
    if not model_path.exists() or not model_info_path.exists():
        logger.error("모델 파일을 찾을 수 없습니다. 먼저 train 모드를 실행하세요.")
        return
    
    # 모델 정보 로드
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # 모델 초기화
    model = MultivariateLSTMClassifier(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        num_classes=model_info['num_classes']
    )
    
    # 모델 가중치 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    logger.info(f"모델 로드 완료: {model_path}")
    
    # 예제 데이터 생성 (실제 구현에서는 실시간 센서 데이터로 대체)
    logger.info("예제 데이터로 추론 수행")
    
    # 테스트 데이터 준비
    processed_files = list(args.processed_dir.glob("multivariate_sensor_data_with_features_*.csv"))
    latest_data_path = max(processed_files, key=lambda p: p.stat().st_ctime)
    
    _, _, test_loader, data_info = prepare_multivariate_data(
        data_path=str(latest_data_path),  # 문자열로 변환
        state_column='state',
        time_column='time',
        feature_cols=None,
        sequence_length=model_info['sequence_length'],
        test_size=0.1,
        val_size=0.1
    )
    
    # 배치에서 하나의 샘플 선택
    for inputs, _ in test_loader:
        sample_input = inputs[0:1]  # 첫 번째 샘플만 선택
        break
    
    # 추론 수행
    with torch.no_grad():
        outputs = model(sample_input)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        
        # 예측 결과 해석
        predicted_class = predicted.item()
        predicted_label = data_info['inverse_class_mapping'][predicted_class]
        confidence = probs[0][predicted_class].item()
        
        logger.info(f"추론 결과:")
        logger.info(f"- 예측 상태: {predicted_label}")
        logger.info(f"- 신뢰도: {confidence:.4f}")
        
        # 각 클래스별 확률
        for i in range(len(probs[0])):
            class_label = data_info['inverse_class_mapping'][i]
            class_prob = probs[0][i].item()
            logger.info(f"- {class_label} 확률: {class_prob:.4f}")
    
    logger.info("추론 데모 완료")

if __name__ == "__main__":
    main()