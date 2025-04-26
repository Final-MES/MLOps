#!/usr/bin/env python
"""
다중 센서 데이터 분류 시스템 CLI 인터페이스

이 스크립트는 센서 데이터 전처리, 모델 학습, 평가, 배포를 위한
대화형 명령줄 인터페이스를 제공합니다.
"""

import os
import sys
import time
import logging
import argparse
import torch
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import shutil

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
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
        logging.FileHandler(os.path.join(project_root, 'logs', 'cli_system.log'))
    ]
)
logger = logging.getLogger(__name__)

# 기본 디렉토리 설정
DEFAULT_DIRS = {
    'data_dir': os.path.join(project_root, 'data', 'raw'),
    'output_dir': os.path.join(project_root, 'data', 'processed'),
    'model_dir': os.path.join(project_root, 'models'),
    'plot_dir': os.path.join(project_root, 'plots'),
    'deploy_dir': os.path.join(project_root, 'deployment')
}

# 전역 변수로 상태 저장
STATE = {
    'preprocessed_data': None,
    'model': None,
    'training_history': None,
    'evaluation_result': None,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'model_params': {
        'sequence_length': 50,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout_rate': 0.3
    },
    'training_params': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'patience': 10
    },
    'preprocessing_params': {
        'file_prefix': 'g1',
        'interp_step': 0.001,
        'window_size': 15,
        'train_ratio': 0.6,
        'val_ratio': 0.2,
        'test_ratio': 0.2
    },
    'current_model_path': None
}

def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title: str):
    """헤더 출력"""
    clear_screen()
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print()

def print_status():
    """현재 상태 출력"""
    print("\n현재 상태:")
    print("-" * 40)
    
    # 데이터 상태
    if STATE['preprocessed_data'] is not None:
        train_data, valid_data, test_data = STATE['preprocessed_data']
        print(f"✅ 전처리된 데이터: 학습 {train_data.shape}, 검증 {valid_data.shape}, 테스트 {test_data.shape}")
    else:
        print("❌ 전처리된 데이터: 없음")
    # 모델 파라미터
    print(f"모델 : (설정된 파라미터: 은닉층 {STATE['model_params']['hidden_size']}, "
          f"레이어 {STATE['model_params']['num_layers']}, 드롭아웃 {STATE['model_params']['dropout_rate']})")
    
    # 학습 상태
    if STATE['training_history'] is not None:
        val_acc = STATE['training_history']['valid_accuracy'][-1]
        print(f"✅ 학습 완료: 검증 정확도 {val_acc:.4f}")
    else:
        print("❌ 학습: 미완료")
    
    # 평가 상태
    if STATE['evaluation_result'] is not None:
        acc = STATE['evaluation_result']['accuracy']
        print(f"✅ 평가 완료: 테스트 정확도 {acc:.4f}")
    else:
        print("❌ 평가: 미완료")
    
    # 배포 상태
    if STATE['current_model_path'] is not None:
        print(f"✅ 배포 준비 완료: {os.path.basename(STATE['current_model_path'])}")
    else:
        print("❌ 배포: 미준비")
    
    print("-" * 40)

def get_input(prompt: str, default: Any = None) -> str:
    """사용자 입력 받기 (기본값 지원)"""
    if default is not None:
        result = input(f"{prompt} [{default}]: ")
        return result if result.strip() else str(default)
    else:
        return input(f"{prompt}: ")

def get_numeric_input(prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
    """숫자 입력 받기 (범위 검사 포함)"""
    while True:
        try:
            result = input(f"{prompt} [{default}]: ")
            if not result.strip():
                result = default
            else:
                result = float(result)
                
            # 정수형인 경우 변환
            if result == int(result):
                result = int(result)
            
            # 범위 검사
            if min_val is not None and result < min_val:
                print(f"값이 너무 작습니다. 최소값: {min_val}")
                continue
            if max_val is not None and result > max_val:
                print(f"값이 너무 큽니다. 최대값: {max_val}")
                continue
                
            return result
        except ValueError:
            print("유효한 숫자를 입력해주세요.")

def get_yes_no_input(prompt: str, default: bool = True) -> bool:
    """예/아니오 입력 받기"""
    default_str = "Y/n" if default else "y/N"
    while True:
        result = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not result:
            return default
        elif result in ['y', 'yes']:
            return True
        elif result in ['n', 'no']:
            return False
        else:
            print("'y' 또는 'n'을 입력해주세요.")

def ensure_dirs() -> None:
    """필요한 디렉토리 생성"""
    for dir_path in DEFAULT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)

def preprocess_data_menu() -> None:
    """데이터 전처리 메뉴"""
    print_header("데이터 전처리")
    
    print("센서 데이터를 로드하고 전처리합니다.")
    print("이 단계에서는 데이터 파일 로드, 보간, 이동 평균 필터링을 수행합니다.\n")
    
    # 데이터 디렉토리 설정
    data_dir = get_input("데이터 디렉토리 경로", DEFAULT_DIRS['data_dir'])
    if not os.path.exists(data_dir):
        print(f"⚠️ 경고: 디렉토리 '{data_dir}'이(가) 존재하지 않습니다.")
        create_dir = get_yes_no_input("디렉토리를 생성하시겠습니까?")
        if create_dir:
            os.makedirs(data_dir, exist_ok=True)
            print(f"✅ 디렉토리 '{data_dir}'이(가) 생성되었습니다.")
        else:
            print("❌ 전처리를 취소합니다.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
    
    # 전처리 매개변수 설정
    print("\n전처리 매개변수 설정:")
    STATE['preprocessing_params']['file_prefix'] = get_input(
        "데이터 파일 접두사", STATE['preprocessing_params']['file_prefix']
    )
    STATE['preprocessing_params']['interp_step'] = get_numeric_input(
        "보간 간격 (초)", STATE['preprocessing_params']['interp_step'], min_val=0.0001, max_val=1.0
    )
    STATE['preprocessing_params']['window_size'] = get_numeric_input(
        "이동 평균 윈도우 크기", STATE['preprocessing_params']['window_size'], min_val=1, max_val=100
    )
    
    # 데이터 분할 비율 설정
    print("\n데이터 분할 비율 설정:")
    while True:
        train_ratio = get_numeric_input(
            "학습 데이터 비율", STATE['preprocessing_params']['train_ratio'], min_val=0.1, max_val=0.9
        )
        val_ratio = get_numeric_input(
            "검증 데이터 비율", STATE['preprocessing_params']['val_ratio'], min_val=0.05, max_val=0.5
        )
        test_ratio = get_numeric_input(
            "테스트 데이터 비율", STATE['preprocessing_params']['test_ratio'], min_val=0.05, max_val=0.5
        )
        
        # 합이 1이 되는지 확인
        total_ratio = train_ratio + val_ratio + test_ratio
        if abs(total_ratio - 1.0) < 0.001:  # 부동소수점 오차 허용
            STATE['preprocessing_params']['train_ratio'] = train_ratio
            STATE['preprocessing_params']['val_ratio'] = val_ratio
            STATE['preprocessing_params']['test_ratio'] = test_ratio
            break
        else:
            print(f"⚠️ 비율의 합이 1이 되어야 합니다. 현재 합: {total_ratio:.2f}")
    
    # 전처리 시작 확인
    print("\n전처리 설정이 완료되었습니다. 전처리를 시작합니다...")
    
    try:
        # 데이터 처리기 초기화
        processor = SensorDataProcessor(
            interpolation_step=STATE['preprocessing_params']['interp_step'],
            window_size=STATE['preprocessing_params']['window_size']
        )
        
        # 전처리 진행 표시
        print("\n[1/3] 센서 데이터 로드 및 보간 중...")
        interpolated_data = processor.load_and_interpolate_sensor_data(
            data_dir, prefix=STATE['preprocessing_params']['file_prefix']
        )
        
        if not interpolated_data:
            print(f"❌ 오류: 센서 데이터를 로드할 수 없습니다. 파일 접두사 '{STATE['preprocessing_params']['file_prefix']}'을(를) 확인하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
        
        print("\n[2/3] 센서 데이터 결합 및 전처리 중...")
        processed_data = processor.combine_and_preprocess_sensor_data(interpolated_data)
        
        print("\n[3/3] 데이터 분할 및 결합 중...")
        train_data, valid_data, test_data = processor.split_and_combine_data(
            processed_data,
            train_ratio=STATE['preprocessing_params']['train_ratio'],
            valid_ratio=STATE['preprocessing_params']['val_ratio'],
            test_ratio=STATE['preprocessing_params']['test_ratio']
        )
        
        # 결과 저장
        STATE['preprocessed_data'] = (train_data, valid_data, test_data)
        
        # 시퀀스 길이 설정 (전체 데이터 길이의 10% 정도로 제안)
        suggested_seq_len = min(int(train_data.shape[0] * 0.1), 100)
        suggested_seq_len = max(suggested_seq_len, 10)  # 최소값 확보
        # 5의 배수로 조정
        suggested_seq_len = (suggested_seq_len // 5) * 5
        if suggested_seq_len != STATE['model_params']['sequence_length']:
            STATE['model_params']['sequence_length'] = suggested_seq_len
        
        # 전처리 결과 출력
        print("\n✅ 전처리가 완료되었습니다.")
        print(f"- 학습 데이터: {train_data.shape} 샘플")
        print(f"- 검증 데이터: {valid_data.shape} 샘플")
        print(f"- 테스트 데이터: {test_data.shape} 샘플")
        print(f"- 제안된 시퀀스 길이: {STATE['model_params']['sequence_length']}")
        
        # 전처리 데이터 저장 여부 확인
        save_data = get_yes_no_input("\n전처리된 데이터를 저장하시겠습니까?")
        if save_data:
            output_dir = DEFAULT_DIRS['output_dir']
            os.makedirs(output_dir, exist_ok=True)
            
            current_time = time.strftime("%Y%m%d_%H%M%S")
            train_path = os.path.join(output_dir, f"train_data_{current_time}.npy")
            valid_path = os.path.join(output_dir, f"valid_data_{current_time}.npy")
            test_path = os.path.join(output_dir, f"test_data_{current_time}.npy")
            
            np.save(train_path, train_data)
            np.save(valid_path, valid_data)
            np.save(test_path, test_data)
            
            print("\n✅ 전처리된 데이터가 저장되었습니다:")
            print(f"- 학습 데이터: {train_path}")
            print(f"- 검증 데이터: {valid_path}")
            print(f"- 테스트 데이터: {test_path}")
        
    except Exception as e:
        print(f"\n❌ 오류: 전처리 중 예외가 발생했습니다: {str(e)}")
        logger.exception("전처리 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def model_params_menu() -> None:
    """모델 파라미터 설정 메뉴"""
    print_header("모델 파라미터 설정")
    
    print("LSTM 분류 모델의 파라미터를 설정합니다.\n")
    
    # 현재 설정 표시
    print("현재 설정:")
    print(f"- 시퀀스 길이: {STATE['model_params']['sequence_length']}")
    print(f"- 은닉층 크기: {STATE['model_params']['hidden_size']}")
    print(f"- LSTM 레이어 수: {STATE['model_params']['num_layers']}")
    print(f"- 드롭아웃 비율: {STATE['model_params']['dropout_rate']}")
    
    # 파라미터 설정
    print("\n새 파라미터 설정:")
    
    # 시퀀스 길이
    STATE['model_params']['sequence_length'] = get_numeric_input(
        "시퀀스 길이", STATE['model_params']['sequence_length'], min_val=5, max_val=200
    )
    
    # 은닉층 크기
    STATE['model_params']['hidden_size'] = get_numeric_input(
        "은닉층 크기", STATE['model_params']['hidden_size'], min_val=16, max_val=512
    )
    
    # LSTM 레이어 수
    STATE['model_params']['num_layers'] = get_numeric_input(
        "LSTM 레이어 수", STATE['model_params']['num_layers'], min_val=1, max_val=5
    )
    
    # 드롭아웃 비율
    STATE['model_params']['dropout_rate'] = get_numeric_input(
        "드롭아웃 비율", STATE['model_params']['dropout_rate'], min_val=0.0, max_val=0.9
    )
    
    # 학습 파라미터 설정
    print("\n학습 파라미터 설정:")
    
    # 배치 크기
    STATE['training_params']['batch_size'] = get_numeric_input(
        "배치 크기", STATE['training_params']['batch_size'], min_val=8, max_val=256
    )
    
    # 학습률
    STATE['training_params']['learning_rate'] = get_numeric_input(
        "학습률", STATE['training_params']['learning_rate'], min_val=0.0001, max_val=0.1
    )
    
    # 에폭 수
    STATE['training_params']['epochs'] = get_numeric_input(
        "최대 에폭 수", STATE['training_params']['epochs'], min_val=10, max_val=1000
    )
    
    # 조기 종료 인내 횟수
    STATE['training_params']['patience'] = get_numeric_input(
        "조기 종료 인내 횟수", STATE['training_params']['patience'], min_val=3, max_val=50
    )
    
    print("\n✅ 모델 파라미터 설정이 완료되었습니다.")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def train_model_menu() -> None:
    """모델 학습 메뉴"""
    print_header("모델 학습")
    
    # 데이터 확인
    if STATE['preprocessed_data'] is None:
        print("❌ 오류: 전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    train_data, valid_data, test_data = STATE['preprocessed_data']
    
    # 학습 파라미터 확인
    print("학습에 사용할 설정:")
    print(f"- 장치: {STATE['device']}")
    print(f"- 시퀀스 길이: {STATE['model_params']['sequence_length']}")
    print(f"- 은닉층 크기: {STATE['model_params']['hidden_size']}")
    print(f"- LSTM 레이어 수: {STATE['model_params']['num_layers']}")
    print(f"- 드롭아웃 비율: {STATE['model_params']['dropout_rate']}")
    print(f"- 배치 크기: {STATE['training_params']['batch_size']}")
    print(f"- 학습률: {STATE['training_params']['learning_rate']}")
    print(f"- 최대 에폭 수: {STATE['training_params']['epochs']}")
    print(f"- 조기 종료 인내 횟수: {STATE['training_params']['patience']}")
    
    # 학습 시작 확인
    start_training = get_yes_no_input("\n위 설정으로 학습을 시작하시겠습니까?")
    if not start_training:
        print("학습을 취소합니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    try:
        # 시퀀스 데이터 준비
        print("\n[1/6] 시퀀스 데이터 준비 중...")
        X_train, y_train = prepare_sequence_data(train_data, sequence_length=STATE['model_params']['sequence_length'])
        X_valid, y_valid = prepare_sequence_data(valid_data, sequence_length=STATE['model_params']['sequence_length'])
        X_test, y_test = prepare_sequence_data(test_data, sequence_length=STATE['model_params']['sequence_length'])
        
        print(f"시퀀스 데이터 준비 완료:")
        print(f"- 학습 데이터: {X_train.shape}, 레이블: {y_train.shape}")
        print(f"- 검증 데이터: {X_valid.shape}, 레이블: {y_valid.shape}")
        print(f"- 테스트 데이터: {X_test.shape}, 레이블: {y_test.shape}")
        
        # 클래스 분포 시각화를 위한 임시 디렉토리
        plot_dir = DEFAULT_DIRS['plot_dir']
        os.makedirs(plot_dir, exist_ok=True)
        
        print("\n[2/6] 데이터 분포 분석 중...")
        plot_class_distribution(y_train, plot_dir, filename='train_class_distribution.png')
        print(f"학습 데이터 클래스 분포 시각화 저장: {os.path.join(plot_dir, 'train_class_distribution.png')}")
        
        # 데이터 로더 준비
        print("\n[3/6] 데이터 로더 준비 중...")
        train_loader, val_loader = prepare_dataloaders(
            X_train, y_train, X_valid, y_valid, STATE['device'],
            batch_size=STATE['training_params']['batch_size']
        )
        
        # 모델 초기화
        print("\n[4/6] 모델 초기화 중...")
        input_size = X_train.shape[2]  # 특성 수
        num_classes = len(np.unique(y_train))  # 클래스 수
        
        model = MultiSensorLSTMClassifier(
            input_size=input_size,
            hidden_size=STATE['model_params']['hidden_size'],
            num_layers=STATE['model_params']['num_layers'],
            num_classes=num_classes,
            dropout_rate=STATE['model_params']['dropout_rate']
        ).to(STATE['device'])
        
        # 모델 정보 출력
        model_info = model.get_model_info()
        print(f"모델 초기화 완료:")
        print(f"- 모델 유형: {model_info['model_type']}")
        print(f"- 입력 크기: {model_info['input_size']}")
        print(f"- 은닉층 크기: {model_info['hidden_size']}")
        print(f"- LSTM 레이어 수: {model_info['num_layers']}")
        print(f"- 출력 클래스 수: {model_info['num_classes']}")
        print(f"- 파라미터 수: {model_info['parameter_count']:,}")
        
        # 모델 학습
        print("\n[5/6] 모델 학습 중...")
        model, history = train_model(
            train_loader=train_loader,
            valid_loader=val_loader,
            model=model,
            device=STATE['device'],
            num_epochs=STATE['training_params']['epochs'],
            learning_rate=STATE['training_params']['learning_rate'],
            early_stopping_patience=STATE['training_params']['patience'],
            model_dir=DEFAULT_DIRS['model_dir']
        )
        
        # 상태 업데이트
        STATE['model'] = model
        STATE['training_history'] = history
        
        # 시험 데이터 로더
        test_loader, _ = prepare_dataloaders(
            X_test, y_test, X_test[:1], y_test[:1], STATE['device'],
            batch_size=STATE['training_params']['batch_size']
        )
        
        # 학습 이력 시각화
        print("\n[6/6] 학습 결과 시각화 중...")
        history_path = plot_training_history(history, plot_dir)
        print(f"학습 이력 시각화 저장: {history_path}")
        
        # 학습 결과 출력
        print("\n✅ 모델 학습이 완료되었습니다.")
        
        # 최종 성능 출력
        val_acc = history['valid_accuracy'][-1]
        val_loss = history['valid_loss'][-1]
        print(f"\n최종 검증 성능:")
        print(f"- 검증 정확도: {val_acc:.4f}")
        print(f"- 검증 손실: {val_loss:.4f}")
        
        # 모델 저장 경로
        model_save_path = os.path.join(
            DEFAULT_DIRS['model_dir'],
            f"sensor_classifier_{time.strftime('%Y%m%d_%H%M%S')}.pth"
        )
        
        # 모델 저장
        torch.save(model.state_dict(), model_save_path)
        STATE['current_model_path'] = model_save_path
        print(f"\n모델이 저장되었습니다: {model_save_path}")
        
        # 모델 정보 저장
        model_info_path = save_model_info(model, DEFAULT_DIRS['model_dir'], STATE['model_params']['sequence_length'])
        print(f"모델 정보가 저장되었습니다: {model_info_path}")
        
        # 시각화 결과 확인 안내
        print(f"\n학습 이력 시각화는 '{history_path}'에서 확인할 수 있습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류: 모델 학습 중 예외가 발생했습니다: {str(e)}")
        logger.exception("모델 학습 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def evaluate_model_menu() -> None:
    """모델 평가 메뉴"""
    print_header("모델 평가")
    
    # 모델 확인
    if STATE['model'] is None:
        print("❌ 오류: 학습된 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 데이터 확인
    if STATE['preprocessed_data'] is None:
        print("❌ 오류: 전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    train_data, valid_data, test_data = STATE['preprocessed_data']
    
    try:
        # 테스트 데이터 준비
        print("\n[1/4] 테스트 데이터 준비 중...")
        X_test, y_test = prepare_sequence_data(test_data, sequence_length=STATE['model_params']['sequence_length'])
        print(f"테스트 데이터 준비 완료: {X_test.shape}, 레이블: {y_test.shape}")
        
        # 테스트 데이터 로더 준비
        print("\n[2/4] 테스트 데이터 로더 준비 중...")
        test_loader, _ = prepare_dataloaders(
            X_test, y_test, X_test[:1], y_test[:1], STATE['device'],
            batch_size=STATE['training_params']['batch_size']
        )
        
        # 모델 평가
        print("\n[3/4] 모델 평가 중...")
        evaluation_result = evaluate_model(STATE['model'], test_loader, STATE['device'])
        
        # 상태 업데이트
        STATE['evaluation_result'] = evaluation_result
        
        # 평가 결과 출력
        print("\n✅ 모델 평가가 완료되었습니다.")
        print(f"테스트 정확도: {evaluation_result['accuracy']:.4f}")
        
        # 클래스별 성능 출력
        print("\n클래스별 성능:")
        class_report = evaluation_result['classification_report']
        for class_name, metrics in class_report.items():
            if isinstance(metrics, dict):  # 클래스별 지표만 출력
                print(f"- {class_name}: 정밀도={metrics['precision']:.4f}, 재현율={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        # 평가 결과 시각화
        print("\n[4/4] 평가 결과 시각화 중...")
        
        # 혼동 행렬 계산을 위한 예측값 수집
        test_predictions = []
        test_labels = []
        
        model = STATE['model']
        model.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # 혼동 행렬 시각화
        cm_path = plot_confusion_matrix(test_labels, test_predictions, DEFAULT_DIRS['plot_dir'])
        print(f"혼동 행렬 시각화 저장: {cm_path}")
        
        # 어텐션 가중치 시각화
        sample_inputs = torch.from_numpy(X_test[:10]).to(STATE['device'])
        sample_labels = torch.from_numpy(y_test[:10]).to(STATE['device'])
        
        attn_path = plot_attention_weights(
            model=model,
            data=sample_inputs,
            labels=sample_labels,
            plot_dir=DEFAULT_DIRS['plot_dir']
        )
        print(f"어텐션 가중치 시각화 저장: {attn_path}")
        
        # 특성 중요도 시각화
        feature_path = plot_feature_importance(
            model=model,
            data=sample_inputs,
            labels=sample_labels,
            plot_dir=DEFAULT_DIRS['plot_dir']
        )
        print(f"특성 중요도 시각화 저장: {feature_path}")
        
        # 평가 결과 저장
        eval_path = save_evaluation_result(evaluation_result, DEFAULT_DIRS['output_dir'])
        print(f"\n평가 결과가 저장되었습니다: {eval_path}")
        
    except Exception as e:
        print(f"\n❌ 오류: 모델 평가 중 예외가 발생했습니다: {str(e)}")
        logger.exception("모델 평가 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def deploy_model_menu() -> None:
    """모델 배포 메뉴"""
    print_header("모델 배포")
    
    # 모델 확인
    if STATE['model'] is None or STATE['current_model_path'] is None:
        print("❌ 오류: 배포할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    print("모델 배포는 학습된 모델을 배포 디렉토리에 복사하고")
    print("추론을 위한 필요한 파일들을 준비하는 단계입니다.\n")
    
    # 배포 디렉토리 설정
    deploy_dir = get_input("배포 디렉토리 경로", DEFAULT_DIRS['deploy_dir'])
    
    try:
        # 배포 디렉토리 생성
        os.makedirs(deploy_dir, exist_ok=True)
        
        # 타임스탬프로 하위 디렉토리 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        deploy_subdir = os.path.join(deploy_dir, f"deployment_{timestamp}")
        os.makedirs(deploy_subdir, exist_ok=True)
        
        print("\n[1/5] 모델 파일 복사 중...")
        model_filename = os.path.basename(STATE['current_model_path'])
        deploy_model_path = os.path.join(deploy_subdir, model_filename)
        shutil.copy2(STATE['current_model_path'], deploy_model_path)
        print(f"모델 파일 복사 완료: {deploy_model_path}")
        
        # 모델 정보 파일 복사
        print("\n[2/5] 모델 정보 파일 복사 중...")
        model_info_src = os.path.join(DEFAULT_DIRS['model_dir'], 'model_info.json')
        model_info_dst = os.path.join(deploy_subdir, 'model_info.json')
        if os.path.exists(model_info_src):
            shutil.copy2(model_info_src, model_info_dst)
            print(f"모델 정보 파일 복사 완료: {model_info_dst}")
        else:
            # 모델 정보 파일이 없으면 새로 생성
            model_info = STATE['model'].get_model_info()
            model_info.update({
                "sequence_length": STATE['model_params']['sequence_length'],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            with open(model_info_dst, 'w') as f:
                json.dump(model_info, f, indent=4)
            print(f"모델 정보 파일 생성 완료: {model_info_dst}")
        
        # 전처리 설정 저장
        print("\n[3/5] 전처리 설정 저장 중...")
        preprocess_config = {
            "file_prefix": STATE['preprocessing_params']['file_prefix'],
            "interp_step": STATE['preprocessing_params']['interp_step'],
            "window_size": STATE['preprocessing_params']['window_size'],
            "sequence_length": STATE['model_params']['sequence_length']
        }
        preprocess_config_path = os.path.join(deploy_subdir, 'preprocess_config.json')
        with open(preprocess_config_path, 'w') as f:
            json.dump(preprocess_config, f, indent=4)
        print(f"전처리 설정 저장 완료: {preprocess_config_path}")
        
        # 평가 결과 복사 (있는 경우)
        print("\n[4/5] 평가 결과 복사 중...")
        if STATE['evaluation_result'] is not None:
            eval_result_path = os.path.join(deploy_subdir, 'evaluation_result.json')
            with open(eval_result_path, 'w') as f:
                # NumPy 배열을 일반 리스트로 변환
                eval_result = STATE['evaluation_result'].copy()
                eval_result_path = os.path.join(deploy_subdir, 'evaluation_result.json')
                with open(eval_result_path, 'w') as f:
                    json.dump(eval_result, f, indent=4)
                print(f"평가 결과 저장 완료: {eval_result_path}")
        else:
            print("평가 결과가 없어 복사를 건너뜁니다.")
        
        # 시각화 결과 복사
        print("\n[5/5] 시각화 결과 복사 중...")
        plot_src_dir = DEFAULT_DIRS['plot_dir']
        plot_dst_dir = os.path.join(deploy_subdir, 'plots')
        os.makedirs(plot_dst_dir, exist_ok=True)
        
        # 주요 시각화 파일 복사
        plot_files = [
            'training_history.png',
            'confusion_matrix.png',
            'attention_weights.png',
            'feature_importance.png'
        ]
        
        copied_files = []
        for plot_file in plot_files:
            src_path = os.path.join(plot_src_dir, plot_file)
            if os.path.exists(src_path):
                dst_path = os.path.join(plot_dst_dir, plot_file)
                shutil.copy2(src_path, dst_path)
                copied_files.append(plot_file)
        
        if copied_files:
            print(f"시각화 파일 복사 완료: {', '.join(copied_files)}")
        else:
            print("복사할 시각화 파일이 없습니다.")
        
        # 추론 스크립트 생성
        inference_script = """#!/usr/bin/env python
\"\"\"
센서 데이터 분류 모델 추론 스크립트

이 스크립트는 학습된 모델을 사용하여 새로운 센서 데이터를 분류합니다.
\"\"\"

import os
import sys
import torch
import numpy as np
import json
import argparse
from typing import Dict, List, Any

# 모델 클래스 정의
class MultiSensorLSTMClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(MultiSensorLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 어텐션 메커니즘
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )
        
        # 분류 레이어
        self.fc = torch.nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # LSTM 출력
        lstm_out, _ = self.lstm(x)
        
        # 어텐션 가중치 계산
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 어텐션 가중치를 사용하여 컨텍스트 벡터 계산
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # 최종 분류 결과
        out = self.fc(context)
        return out

def load_model(model_path: str, model_info_path: str) -> torch.nn.Module:
    \"\"\"모델 로드\"\"\"
    # 모델 정보 로드
    with open(model_info_path, 'r') as f:
        model_info = json.load(f)
    
    # 모델 초기화
    model = MultiSensorLSTMClassifier(
        input_size=model_info['input_size'],
        hidden_size=model_info['hidden_size'],
        num_layers=model_info['num_layers'],
        num_classes=model_info['num_classes']
    )
    
    # 모델 가중치 로드
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

def predict(model: torch.nn.Module, data: np.ndarray, sequence_length: int) -> Dict[str, Any]:
    \"\"\"예측 수행\"\"\"
    # 데이터를 시퀀스로 변환
    if len(data) < sequence_length:
        raise ValueError(f"데이터 길이가 너무 짧습니다. 최소 {sequence_length}개 이상 필요합니다.")
    
    # 마지막 시퀀스 선택
    sequence = data[-sequence_length:]
    
    # 텐서 변환
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # (1, sequence_length, input_size)
    
    # 예측
    with torch.no_grad():
        output = model(x)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # 클래스 매핑
    state_mapping = {0: 'normal', 1: 'type1', 2: 'type2', 3: 'type3'}
    predicted_state = state_mapping[predicted_class]
    
    return {
        'predicted_class': predicted_class,
        'predicted_state': predicted_state,
        'confidence': confidence,
        'probabilities': probabilities[0].tolist()
    }

def main():
    parser = argparse.ArgumentParser(description='센서 데이터 분류 추론')
    parser.add_argument('--data', required=True, help='센서 데이터 파일 경로 (CSV 또는 NumPy)')
    parser.add_argument('--model', default='model.pth', help='모델 파일 경로')
    parser.add_argument('--model-info', default='model_info.json', help='모델 정보 파일 경로')
    args = parser.parse_args()
    
    try:
        # 모델 로드
        print(f"모델 로드 중: {args.model}")
        model = load_model(args.model, args.model_info)
        
        # 모델 정보 로드
        with open(args.model_info, 'r') as f:
            model_info = json.load(f)
        
        sequence_length = model_info.get('sequence_length', 50)
        
        # 데이터 로드
        print(f"데이터 로드 중: {args.data}")
        if args.data.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(args.data)
            data = df.values
        elif args.data.endswith('.npy'):
            data = np.load(args.data)
        else:
            raise ValueError("지원되지 않는 파일 형식입니다. CSV 또는 NumPy 파일을 사용하세요.")
        
        # 예측 수행
        print("예측 수행 중...")
        result = predict(model, data, sequence_length)
        
        # 결과 출력
        print("\\n예측 결과:")
        print(f"- 예측 상태: {result['predicted_state']}")
        print(f"- 신뢰도: {result['confidence']:.4f}")
        
        print("\\n클래스별 확률:")
        state_mapping = {0: 'normal', 1: 'type1', 2: 'type2', 3: 'type3'}
        for i, prob in enumerate(result['probabilities']):
            print(f"- {state_mapping[i]}: {prob:.4f}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
"""
        
        # 추론 스크립트 저장
        inference_script_path = os.path.join(deploy_subdir, 'inference.py')
        with open(inference_script_path, 'w') as f:
            f.write(inference_script)
        
        # 실행 권한 부여
        os.chmod(inference_script_path, 0o755)
        print(f"\n추론 스크립트 생성 완료: {inference_script_path}")
        
        # README 생성
        readme_content = f"""# 센서 데이터 분류 모델 배포

## 배포 정보
- 배포 날짜: {time.strftime("%Y-%m-%d %H:%M:%S")}
- 모델 파일: {model_filename}

## 사용 방법

### 필요 조건
- Python 3.7 이상
- PyTorch
- NumPy

### 추론 실행
```bash
python inference.py --data 당신의_데이터.csv --model {model_filename} --model-info model_info.json
```

## 모델 정보
- 모델 유형: {STATE['model'].get_model_info()['model_type']}
- 입력 크기: {STATE['model'].get_model_info()['input_size']}
- 은닉층 크기: {STATE['model'].get_model_info()['hidden_size']}
- LSTM 레이어 수: {STATE['model'].get_model_info()['num_layers']}
- 시퀀스 길이: {STATE['model_params']['sequence_length']}
"""
        
        # README 저장
        readme_path = os.path.join(deploy_subdir, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"README 생성 완료: {readme_path}")
        
        print(f"\n✅ 모델 배포가 완료되었습니다: {deploy_subdir}")
        print("\n배포 파일 목록:")
        for root, dirs, files in os.walk(deploy_subdir):
            level = root.replace(deploy_subdir, '').count(os.sep)
            indent = ' ' * 4 * level
            subdir = os.path.basename(root)
            if level > 0:
                print(f"{indent}{subdir}/")
            for file in files:
                print(f"{indent}{'    ' if level > 0 else ''}{file}")
        
    except Exception as e:
        print(f"\n❌ 오류: 모델 배포 중 예외가 발생했습니다: {str(e)}")
        logger.exception("모델 배포 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def load_data_menu() -> None:
    """저장된 데이터 로드 메뉴"""
    print_header("저장된 데이터 로드")
    
    print("이전에 저장한 전처리 데이터를 로드합니다.\n")
    
    # 저장된 데이터 파일 목록 표시
    output_dir = DEFAULT_DIRS['output_dir']
    npy_files = [f for f in os.listdir(output_dir) if f.endswith('.npy')]
    
    if not npy_files:
        print(f"❌ 오류: '{output_dir}' 디렉토리에 저장된 데이터 파일이 없습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 파일 목록 출력
    print("저장된 데이터 파일:")
    train_files = [f for f in npy_files if f.startswith('train_data_')]
    valid_files = [f for f in npy_files if f.startswith('valid_data_')]
    test_files = [f for f in npy_files if f.startswith('test_data_')]
    
    # 시간별로 정렬
    train_files.sort(reverse=True)
    valid_files.sort(reverse=True)
    test_files.sort(reverse=True)
    
    # 동일한 시간대의 파일 그룹화
    data_sets = {}
    for train_file in train_files:
        timestamp = train_file[11:-4]  # 'train_data_' 제외, '.npy' 제외
        data_sets[timestamp] = {'train': train_file}
    
    for valid_file in valid_files:
        timestamp = valid_file[11:-4]
        if timestamp in data_sets:
            data_sets[timestamp]['valid'] = valid_file
    
    for test_file in test_files:
        timestamp = test_file[10:-4]
        if timestamp in data_sets:
            data_sets[timestamp]['test'] = test_file
    
    # 완전한 세트만 필터링
    complete_sets = {ts: files for ts, files in data_sets.items() 
                   if len(files) == 3 and 'train' in files and 'valid' in files and 'test' in files}
    
    if not complete_sets:
        print(f"❌ 오류: 완전한 데이터 세트(학습/검증/테스트)가 없습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 데이터 세트 출력
    print("\n사용 가능한 데이터 세트:")
    timestamps = list(complete_sets.keys())
    timestamps.sort(reverse=True)
    
    for i, ts in enumerate(timestamps, 1):
        files = complete_sets[ts]
        train_size = os.path.getsize(os.path.join(output_dir, files['train'])) // (1024 * 1024)
        print(f"{i}. 데이터 세트 {ts} (약 {train_size}MB)")
    
    # 사용자 선택
    while True:
        choice = get_input("\n로드할 데이터 세트 번호를 입력하세요", "1")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(timestamps):
                selected_ts = timestamps[choice_idx]
                selected_files = complete_sets[selected_ts]
                break
            else:
                print(f"유효한 번호를 입력하세요 (1-{len(timestamps)})")
        except ValueError:
            print("숫자를 입력하세요")
    
    try:
        print(f"\n데이터 세트 '{selected_ts}' 로드 중...")
        
        # 데이터 로드
        train_path = os.path.join(output_dir, selected_files['train'])
        valid_path = os.path.join(output_dir, selected_files['valid'])
        test_path = os.path.join(output_dir, selected_files['test'])
        
        train_data = np.load(train_path)
        valid_data = np.load(valid_path)
        test_data = np.load(test_path)
        
        # 상태 업데이트
        STATE['preprocessed_data'] = (train_data, valid_data, test_data)
        
        # 시퀀스 길이 제안
        suggested_seq_len = min(int(train_data.shape[0] * 0.1), 100)
        suggested_seq_len = max(suggested_seq_len, 10)  # 최소값 확보
        # 5의 배수로 조정
        suggested_seq_len = (suggested_seq_len // 5) * 5
        STATE['model_params']['sequence_length'] = suggested_seq_len
        
        print("\n✅ 데이터 로드가 완료되었습니다.")
        print(f"- 학습 데이터: {train_data.shape} 샘플")
        print(f"- 검증 데이터: {valid_data.shape} 샘플")
        print(f"- 테스트 데이터: {test_data.shape} 샘플")
        print(f"- 제안된 시퀀스 길이: {STATE['model_params']['sequence_length']}")
        
    except Exception as e:
        print(f"\n❌ 오류: 데이터 로드 중 예외가 발생했습니다: {str(e)}")
        logger.exception("데이터 로드 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")
def load_model_menu() -> None:
    """저장된 모델 로드 메뉴"""
    print_header("저장된 모델 로드")
    
    print("이전에 저장한 모델을 로드합니다.\n")
    
    # 저장된 모델 파일 목록 표시
    model_dir = DEFAULT_DIRS['model_dir']
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    
    if not model_files:
        print(f"❌ 오류: '{model_dir}' 디렉토리에 저장된 모델 파일이 없습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 파일 수정 시간 기준으로 정렬 (최신 순)
    model_files.sort(key=lambda f: os.path.getmtime(os.path.join(model_dir, f)), reverse=True)
    
    # 모델 목록 출력
    print("저장된 모델 파일:")
    for i, model_file in enumerate(model_files, 1):
        file_path = os.path.join(model_dir, model_file)
        file_size = os.path.getsize(file_path) // 1024  # KB 단위
        file_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_path)))
        print(f"{i}. {model_file} ({file_size:,}KB, {file_time})")
    
    # 사용자 선택
    while True:
        choice = get_input("\n로드할 모델 번호를 입력하세요", "1")
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(model_files):
                selected_model = model_files[choice_idx]
                break
            else:
                print(f"유효한 번호를 입력하세요 (1-{len(model_files)})")
        except ValueError:
            print("숫자를 입력하세요")
    
    # 모델 정보 파일 확인
    model_info_path = os.path.join(model_dir, 'model_info.json')
    if not os.path.exists(model_info_path):
        print(f"⚠️ 경고: 모델 정보 파일 '{model_info_path}'을(를) 찾을 수 없습니다.")
        print("모델 정보를 수동으로 입력해야 합니다.")
        
        # 모델 구성 정보 입력
        input_size = get_numeric_input("입력 특성 수", 4, min_val=1)
        hidden_size = get_numeric_input("은닉층 크기", STATE['model_params']['hidden_size'], min_val=8)
        num_layers = get_numeric_input("LSTM 레이어 수", STATE['model_params']['num_layers'], min_val=1, max_val=5)
        num_classes = get_numeric_input("출력 클래스 수", 4, min_val=2)
        sequence_length = get_numeric_input("시퀀스 길이", STATE['model_params']['sequence_length'], min_val=5)
        
        # 모델 파라미터 업데이트
        STATE['model_params']['hidden_size'] = hidden_size
        STATE['model_params']['num_layers'] = num_layers
        STATE['model_params']['sequence_length'] = sequence_length
        
        model_info = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_classes': num_classes,
            'sequence_length': sequence_length
        }
    else:
        # 모델 정보 파일에서 로드
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
        
        # 모델 파라미터 업데이트
        STATE['model_params']['hidden_size'] = model_info.get('hidden_size', STATE['model_params']['hidden_size'])
        STATE['model_params']['num_layers'] = model_info.get('num_layers', STATE['model_params']['num_layers'])
        STATE['model_params']['sequence_length'] = model_info.get('sequence_length', STATE['model_params']['sequence_length'])
    try:
        print(f"\n모델 '{selected_model}' 로드 중...")
        
        # 모델 초기화
        model = MultiSensorLSTMClassifier(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes'],
            dropout_rate=STATE['model_params']['dropout_rate']
        ).to(STATE['device'])
        
        # 모델 가중치 로드
        model_path = os.path.join(model_dir, selected_model)
        model.load_state_dict(torch.load(model_path, map_location=STATE['device']))
        model.eval()
        
        # 상태 업데이트
        STATE['model'] = model
        STATE['current_model_path'] = model_path
        
        print("\n✅ 모델 로드가 완료되었습니다.")
        print(f"- 모델 파일: {selected_model}")
        print(f"- 입력 크기: {model_info['input_size']}")
        print(f"- 은닉층 크기: {model_info['hidden_size']}")
        print(f"- LSTM 레이어 수: {model_info['num_layers']}")
        print(f"- 출력 클래스 수: {model_info['num_classes']}")
        print(f"- 시퀀스 길이: {STATE['model_params']['sequence_length']}")
        
    except Exception as e:
        print(f"\n❌ 오류: 모델 로드 중 예외가 발생했습니다: {str(e)}")
        logger.exception("모델 로드 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def system_config_menu() -> None:
    """시스템 설정 메뉴"""
    print_header("시스템 설정")
    
    print("시스템 설정을 변경합니다.\n")
    
    # 디렉토리 설정
    print("디렉토리 설정:")
    for dir_name, dir_path in DEFAULT_DIRS.items():
        new_path = get_input(f"{dir_name} 디렉토리", dir_path)
        DEFAULT_DIRS[dir_name] = new_path
        os.makedirs(new_path, exist_ok=True)
    
    # 장치 설정
    if torch.cuda.is_available():
        use_gpu = get_yes_no_input("\nGPU를 사용하시겠습니까?", default=True)
        STATE['device'] = torch.device("cuda" if use_gpu else "cpu")
    else:
        print("\nGPU를 사용할 수 없습니다. CPU를 사용합니다.")
        STATE['device'] = torch.device("cpu")
    
    print(f"\n✅ 시스템 설정이 변경되었습니다.")
    print(f"- 현재 장치: {STATE['device']}")
    for dir_name, dir_path in DEFAULT_DIRS.items():
        print(f"- {dir_name} 디렉토리: {dir_path}")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def main_menu() -> None:
    """메인 메뉴 표시"""
    while True:
        print_header("다중 센서 데이터 분류 시스템")
        print("머신러닝 파이프라인 관리 시스템에 오신 것을 환영합니다.")
        print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
        
        print("1. 데이터 전처리")
        print("2. 모델 파라미터 설정")
        print("3. 모델 학습")
        print("4. 모델 평가")
        print("5. 모델 배포")
        print("6. 저장된 데이터 로드")
        print("7. 저장된 모델 로드")
        print("8. 시스템 설정")
        print("0. 종료\n")
        
        print_status()
        
        choice = get_input("\n메뉴 선택", "0")
        
        if choice == "1":
            preprocess_data_menu()
        elif choice == "2":
            model_params_menu()
        elif choice == "3":
            train_model_menu()
        elif choice == "4":
            evaluate_model_menu()
        elif choice == "5":
            deploy_model_menu()
        elif choice == "6":
            load_data_menu()
        elif choice == "7":
            load_model_menu()
        elif choice == "8":
            system_config_menu()
        elif choice == "0":
            print("\n프로그램을 종료합니다. 감사합니다!")
            break
        else:
            print("\n유효하지 않은 선택입니다. 다시 시도하세요.")
            input("계속하려면 Enter 키를 누르세요...")

def main():
    """메인 함수"""
    try:
        # 필요한 디렉토리 생성
        ensure_dirs()
        
        # 로깅 설정
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 메인 메뉴 실행
        main_menu()
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
        logger.exception("예상치 못한 오류 발생")
        sys.exit(1)

if __name__ == "__main__":
    main()