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
from src.models.multivariate_model import prepare_multivariate_data, train_multivariate_model, MultivariateLSTMClassifier  # 모델 학습 관련 함수와 클래스
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
    
    # 각 센서 데이터 파일 목록 가져오기
    sensor_files = {}
    for sensor_id in ['g1_sensor1', 'g1_sensor2', 'g1_sensor3', 'g1_sensor4']:
        sensor_files[sensor_id] = []
        for state in ['normal', 'type1', 'type2', 'type3']:
            pattern = f"{sensor_id}_{state}*.csv"
            matching_files = list(args.data_dir.glob(pattern))
            if matching_files:
                sensor_files[sensor_id].extend(matching_files)
    
    # 상태별로 센서 데이터 처리
    all_processed_data = []
    
    # 전처리기 초기화
    preprocessor = SensorDataPreprocessor(window_size=15)
    
    for state in ['normal', 'type1', 'type2', 'type3']:
        logger.info(f"{state} 상태 데이터 처리")
        
        # 각 센서별 현재 상태의 데이터 로드
        state_sensor_data = {}
        for sensor_id, files in sensor_files.items():
            state_files = [f for f in files if f"_{state}" in str(f)]
            if not state_files:
                logger.warning(f"{sensor_id}의 {state} 상태 데이터 파일을 찾을 수 없음")
                continue
                
            # 현재 상태 파일 중 첫 번째 파일 사용 (또는 병합 가능)
            df = pd.read_csv(state_files[0])
            
            # 시간 컬럼 처리
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
            elif 'time' not in df.columns:
                df['time'] = np.arange(len(df))
                
            state_sensor_data[sensor_id] = df
            logger.info(f"{sensor_id} {state} 데이터 로드: {len(df)} 행")
        
        # 동일한 시간 간격으로 보간
        if state_sensor_data:
            interpolated_data = preprocessor.interpolate_sensor_data(
                state_sensor_data,
                time_range=None,
                step=0.001,
                kind='linear'
            )
            
            # 각 센서 데이터를 하나의 데이터프레임으로 결합
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
            
            if combined_df is not None:
                # 상태 레이블 추가
                combined_df['state'] = state
                
                # 결과 저장
                all_processed_data.append(combined_df)
                logger.info(f"{state} 상태 결합 데이터: {len(combined_df)} 행")
    
    # 모든 상태 데이터 통합
    if all_processed_data:
        final_df = pd.concat(all_processed_data, ignore_index=True)
        
        # 결측치 처리
        final_df = final_df.fillna(method='ffill').fillna(method='bfill')
        
        # 처리된 데이터 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = args.processed_dir / f"multivariate_sensor_data_{timestamp}.csv"
        final_df.to_csv(output_path, index=False)
        logger.info(f"통합 데이터 저장 완료: {output_path} (총 {len(final_df)} 행)")
        
        # 추가적인 특성 추출
        logger.info("추가 특성 추출 진행 중...")
        feature_df = preprocessor.extract_statistical_moments(final_df, columns=[col for col in final_df.columns if col != 'time' and col != 'state'])
        feature_df = preprocessor.extract_frequency_features(feature_df, columns=[col for col in final_df.columns if col != 'time' and col != 'state'])
        
        # 특성이 추가된 데이터 저장
        feature_output_path = args.processed_dir / f"multivariate_sensor_data_with_features_{timestamp}.csv"
        feature_df.to_csv(feature_output_path, index=False)
        logger.info(f"특성이 추가된 데이터 저장 완료: {feature_output_path} (총 {len(feature_df)} 행, {len(feature_df.columns)} 열)")
        
        return output_path, feature_output_path
    else:
        logger.error("처리할 수 있는 센서 데이터가 없습니다.")
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