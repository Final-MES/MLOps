#!/usr/bin/env python
"""
다중 센서 데이터 분류 파이프라인

이 스크립트는 다중 센서 데이터를 사용하여 상태 분류 모델을 학습하고 평가하는 메인 파이프라인을 구현합니다.
"""

import os
import sys
import logging
import argparse
import torch
import numpy as np
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# 프로젝트 루트 경로를 파이썬 경로에 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# 모듈 임포트
from src.data.sensor.sensor_processor import SensorDataProcessor, prepare_sequence_data
from src.models.sensor.lstm_classifier import MultiSensorLSTMClassifier
from src.utils.training import (
    prepare_dataloaders, train_model, evaluate_model,
    save_model_info, save_evaluation_result
)
from src.utils.visualization import (
    plot_training_history, plot_confusion_matrix, plot_class_distribution,
    plot_sensor_data, plot_attention_weights, plot_feature_importance
)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'sensor_classification.log'))
    ]
)
logger = logging.getLogger(__name__)

def ensure_dir(directory: str) -> str:
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(directory, exist_ok=True)
    return directory

def preprocess_sensor_data(args: argparse.Namespace) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    센서 데이터 전처리
    
    Args:
        args: 명령줄 인자
        
    Returns:
        tuple: (학습 데이터, 검증 데이터, 테스트 데이터) 또는 오류 시 (None, None, None)
    """
    logger.info("센서 데이터 전처리 시작")
    
    # 데이터 디렉토리 확인
    if not os.path.isdir(args.data_dir):
        logger.error(f"데이터 디렉토리 {args.data_dir}이(가) 존재하지 않습니다.")
        return None, None, None

    try:
        # 데이터 처리기 초기화
        processor = SensorDataProcessor(interpolation_step=args.interp_step, window_size=args.window_size)
    
        # 센서 데이터 로드 및 보간
        logger.info("센서 데이터 로드 및 보간 중...")
        interpolated_data = processor.load_and_interpolate_sensor_data(args.data_dir, prefix=args.file_prefix)
        
        if not interpolated_data:
            logger.error("센서 데이터를 로드할 수 없습니다.")
            return None, None, None
    
        # 센서 데이터 결합 및 전처리
        logger.info("센서 데이터 결합 및 전처리 중...")
        processed_data = processor.combine_and_preprocess_sensor_data(interpolated_data)
    
        # 데이터 분할 및 결합
        logger.info("데이터 분할 및 결합 중...")
        train_data, valid_data, test_data = processor.split_and_combine_data(
            processed_data,
            train_ratio=args.train_ratio,
            valid_ratio=args.val_ratio,
            test_ratio=args.test_ratio
        )
    
        logger.info(f"학습 데이터 형태: {train_data.shape}")
        logger.info(f"검증 데이터 형태: {valid_data.shape}")
        logger.info(f"테스트 데이터 형태: {test_data.shape}")
    
        return train_data, valid_data, test_data
    
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def run_training_pipeline(args: argparse.Namespace) -> bool:
    """
    학습 파이프라인 실행
    
    Args:
        args: 명령줄 인자
        
    Returns:
        bool: 성공 여부
    """
    # 필요한 디렉토리 생성
    ensure_dir(args.output_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.plot_dir)
    
    # 데이터 전처리
    logger.info("===== 센서 데이터 전처리 =====")
    train_data, valid_data, test_data = preprocess_sensor_data(args)
    
    if train_data is None or valid_data is None or test_data is None:
        logger.error("데이터 전처리에 실패했습니다.")
        return False
    
    # 처리된 데이터 저장 (선택적)
    if args.save_data:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_path = os.path.join(args.output_dir, f"train_data_{current_time}.npy")
        valid_path = os.path.join(args.output_dir, f"valid_data_{current_time}.npy")
        test_path = os.path.join(args.output_dir, f"test_data_{current_time}.npy")
        
        np.save(train_path, train_data)
        np.save(valid_path, valid_data)
        np.save(test_path, test_data)
        
        logger.info(f"처리된 데이터 저장 완료:")
        logger.info(f"- 학습 데이터: {train_path}")
        logger.info(f"- 검증 데이터: {valid_path}")
        logger.info(f"- 테스트 데이터: {test_path}")
    
    # 시퀀스 데이터 준비
    logger.info("시퀀스 데이터 준비 중...")
    X_train, y_train = prepare_sequence_data(train_data, sequence_length=args.sequence_length)
    X_valid, y_valid = prepare_sequence_data(valid_data, sequence_length=args.sequence_length)
    X_test, y_test = prepare_sequence_data(test_data, sequence_length=args.sequence_length)
    
    # 클래스 분포 시각화
    plot_class_distribution(y_train, args.plot_dir, filename='train_class_distribution.png')
    plot_class_distribution(y_test, args.plot_dir, filename='test_class_distribution.png')
    
    # 센서 데이터 시각화 (샘플)
    plot_sensor_data(
        train_data[:1000],
        window_start=0,
        window_size=500,
        plot_dir=args.plot_dir,
        filename='sample_sensor_data.png'
    )
    
    # PyTorch 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_gpu else "cpu")
    logger.info(f"학습에 사용할 장치: {device}")
    
    # 데이터 로더 준비
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_valid, y_valid, device, batch_size=args.batch_size
    )
    
    # 테스트 데이터 로더
    test_loader, _ = prepare_dataloaders(
        X_test, y_test, X_test[:1], y_test[:1], device, batch_size=args.batch_size
    )
    
    # 모델 초기화
    input_size = X_train.shape[2]  # 특성 수
    num_classes = len(np.unique(y_train))  # 클래스 수
    
    model = MultiSensorLSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=num_classes,
        dropout_rate=args.dropout_rate
    ).to(device)
    
    # 모델 훈련
    logger.info("===== 모델 학습 시작 =====")
    model, history = train_model(
        train_loader=train_loader,
        valid_loader=val_loader,
        model=model,
        device=device,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.patience,
        model_dir=args.model_dir
    )
    
    # 학습 이력 시각화
    plot_training_history(history, args.plot_dir)
    
    # 모델 평가
    logger.info("===== 모델 평가 =====")
    evaluation_result = evaluate_model(model, test_loader, device)
    
    # 결과 출력
    logger.info(f"테스트 데이터 평가 결과:")
    logger.info(f"- 정확도: {evaluation_result['accuracy']:.4f}")
    
    # 혼동 행렬 시각화
    test_predictions = []
    test_labels = []
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    plot_confusion_matrix(test_labels, test_predictions, args.plot_dir)
    
    # 특성 중요도 시각화 (샘플 데이터 사용)
    sample_inputs = torch.from_numpy(X_test[:100]).to(device)
    sample_labels = torch.from_numpy(y_test[:100]).to(device)
    
    plot_feature_importance(
        model=model,
        data=sample_inputs,
        labels=sample_labels,
        plot_dir=args.plot_dir
    )
    
    # 어텐션 가중치 시각화 (샘플 데이터 사용)
    plot_attention_weights(
        model=model,
        data=sample_inputs[:10],
        labels=sample_labels[:10],
        plot_dir=args.plot_dir
    )
    
    # 모델 저장
    model_path = os.path.join(args.model_dir, f"sensor_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"최종 모델 저장 완료: {model_path}")
    
    # 모델 정보 저장
    save_model_info(model, args.model_dir, sequence_length=args.sequence_length)
    
    # 평가 결과 저장
    save_evaluation_result(evaluation_result, args.output_dir)
    
    logger.info("===== 훈련 파이프라인 완료 =====")
    
    return True

def main() -> None:
    """메인 함수: 명령줄 인자 파싱 및 파이프라인 실행"""
    parser = argparse.ArgumentParser(description='다중 센서 데이터를 이용한 상태 분류 모델')
    
    # 기본 디렉토리 설정
    parser.add_argument('--data_dir', type=str, default=os.path.join(project_root, 'data', 'raw'),
                      help='원시 데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default=os.path.join(project_root, 'data', 'processed'),
                      help='처리된 데이터 저장 디렉토리')
    parser.add_argument('--model_dir', type=str, default=os.path.join(project_root, 'models'),
                      help='모델 저장 디렉토리')
    parser.add_argument('--plot_dir', type=str, default=os.path.join(project_root, 'plots'),
                      help='결과 시각화 저장 디렉토리')
    
    # 데이터 전처리 인자
    parser.add_argument('--file_prefix', type=str, default='g1',
                      help='센서 데이터 파일 접두사')
    parser.add_argument('--interp_step', type=float, default=0.001,
                      help='보간 간격 (초 단위)')
    parser.add_argument('--window_size', type=int, default=15,
                      help='이동 평균 윈도우 크기')
    parser.add_argument('--save_data', action='store_true',
                      help='처리된 데이터 저장 여부')
    parser.add_argument('--train_ratio', type=float, default=0.6,
                      help='학습 데이터 비율')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                      help='검증 데이터 비율')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                      help='테스트 데이터 비율')
    
    # 모델 인자
    parser.add_argument('--sequence_length', type=int, default=50,
                      help='시퀀스 길이')
    parser.add_argument('--hidden_size', type=int, default=64,
                      help='LSTM 은닉층 크기')
    parser.add_argument('--num_layers', type=int, default=2,
                      help='LSTM 레이어 수')
    parser.add_argument('--dropout_rate', type=float, default=0.3,
                      help='드롭아웃 비율')
    
    # 학습 인자
    parser.add_argument('--epochs', type=int, default=100,
                      help='학습 에폭 수')
    parser.add_argument('--batch_size', type=int, default=32,
                      help='배치 크기')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                      help='학습률')
    parser.add_argument('--patience', type=int, default=10,
                      help='조기 종료 인내 횟수')
    parser.add_argument('--use_gpu', action='store_true',
                      help='GPU 사용 여부')
    
    args = parser.parse_args()
    
    try:
        # 파이프라인 실행
        success = run_training_pipeline(args)
        
        if success:
            logger.info("프로그램이 성공적으로 완료되었습니다.")
            sys.exit(0)
        else:
            logger.error("프로그램이 오류와 함께 종료되었습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"예상치 못한 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()