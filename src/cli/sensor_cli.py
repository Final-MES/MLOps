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
import torch
import json
import numpy as np
from pathlib import Path
import shutil

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기본 CLI 클래스 임포트
from src.cli.base_cli import BaseCLI
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

# 데이터베이스 유틸리티 임포트
from src.utils.db.connector import DBConnector
from src.utils.db.exporter import DBExporter
from src.utils.config_loader import load_db_config, load_export_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class SensorCLI(BaseCLI):
    """
    다중 센서 데이터 분류 시스템 CLI 클래스
    
    센서 데이터 전처리, 모델 학습, 평가, 배포를 위한
    대화형 명령줄 인터페이스를 제공합니다.
    """
    
    def __init__(self):
        """센서 CLI 초기화"""
        super().__init__(title="다중 센서 데이터 분류 시스템")
        
        # 상태 초기화
        self.state = {
            'preprocessed_data': None,
            'model': None,
            'training_history': None,
            'evaluation_result': None,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'current_model_path': None,
            'db_connected': False
        }
        
        # 모델 파라미터
        self.model_params = {
            'sequence_length': 100,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout_rate': 0.3
        }
        
        # 학습 파라미터
        self.training_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 100,
            'patience': 10
        }
        
        # 전처리 파라미터
        self.preprocessing_params = {
            'file_prefix': 'g1',
            'interp_step': 0.001,
            'window_size': 15,
            'train_ratio': 0.6,
            'val_ratio': 0.2,
            'test_ratio': 0.2
        }
        
        # 기본 디렉토리 설정
        self.paths = {
            'data_dir': os.path.join(project_root, 'data', 'raw'),
            'output_dir': os.path.join(project_root, 'data', 'processed'),
            'model_dir': os.path.join(project_root, 'models'),
            'plot_dir': os.path.join(project_root, 'plots'),
            'deploy_dir': os.path.join(project_root, 'deployment')
        }
        
        # 필요한 디렉토리 생성
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info("센서 데이터 분류 CLI 초기화 완료")
    
    def print_status(self) -> None:
        """현재 상태 출력"""
        print("\n현재 상태:")
        print("-" * 40)
        
        # 데이터 상태
        if self.state['preprocessed_data'] is not None:
            train_data, valid_data, test_data = self.state['preprocessed_data']
            print(f"✅ 전처리된 데이터: 학습 {train_data.shape}, 검증 {valid_data.shape}, 테스트 {test_data.shape}")
        else:
            print("❌ 전처리된 데이터: 없음")
            
        # 모델 파라미터
        print(f"모델: (설정된 파라미터: 은닉층 {self.model_params['hidden_size']}, "
              f"레이어 {self.model_params['num_layers']}, 드롭아웃 {self.model_params['dropout_rate']})")
        
        # 학습 상태
        if self.state['training_history'] is not None:
            val_acc = self.state['training_history']['valid_accuracy'][-1]
            print(f"✅ 학습 완료: 검증 정확도 {val_acc:.4f}")
        else:
            print("❌ 학습: 미완료")
        
        # 평가 상태
        if self.state['evaluation_result'] is not None:
            acc = self.state['evaluation_result']['accuracy']
            print(f"✅ 평가 완료: 테스트 정확도 {acc:.4f}")
        else:
            print("❌ 평가: 미완료")
        
        # 배포 상태
        if self.state['current_model_path'] is not None:
            print(f"✅ 배포 준비 완료: {os.path.basename(self.state['current_model_path'])}")
        else:
            print("❌ 배포: 미준비")
        
        # 데이터베이스 연결 상태
        if self.state['db_connected']:
            print(f"✅ 데이터베이스: 연결됨")
        else:
            print("❌ 데이터베이스: 연결되지 않음")

        print("-" * 40)
    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header("다중 센서 데이터 분류 시스템")
            print("머신러닝 파이프라인 관리 시스템에 오신 것을 환영합니다.")
            print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
            
            menu_options = [
                "데이터 전처리",
                "모델 파라미터 설정",
                "모델 학습",
                "모델 평가",
                "모델 배포",
                "저장된 데이터 로드",
                "저장된 모델 로드",
                "시스템 설정",
                "종료"
            ]
            
            self.print_status()
            choice = self.show_menu(menu_options, "메인 메뉴")
            
            if choice == 0:
                self.preprocess_data_menu()
            elif choice == 1:
                self.model_params_menu()
            elif choice == 2:
                self.train_model_menu()
            elif choice == 3:
                self.evaluate_model_menu()
            elif choice == 4:
                self.deploy_model_menu()
            elif choice == 5:
                self.load_data_menu()
            elif choice == 6:
                self.load_model_menu()
            elif choice == 7:
                self.system_config_menu()
            elif choice == 8:
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
    
    def preprocess_data_menu(self) -> None:
        """데이터 전처리 메뉴"""
        self.print_header("데이터 전처리")
        
        print("센서 데이터를 로드하고 전처리합니다.")
        print("이 단계에서는 데이터 파일 로드, 보간, 이동 평균 필터링을 수행합니다.\n")
        
        # 데이터 디렉토리 설정
        data_dir = self.get_input("데이터 디렉토리 경로", self.paths['data_dir'])
        if not os.path.exists(data_dir):
            self.show_warning(f"디렉토리 '{data_dir}'이(가) 존재하지 않습니다.")
            create_dir = self.get_yes_no_input("디렉토리를 생성하시겠습니까?")
            if create_dir:
                os.makedirs(data_dir, exist_ok=True)
                self.show_success(f"디렉토리 '{data_dir}'이(가) 생성되었습니다.")
            else:
                self.show_error("전처리를 취소합니다.")
                self.wait_for_user()
                return
        
        # 전처리 매개변수 설정
        print("\n전처리 매개변수 설정:")
        self.preprocessing_params['file_prefix'] = self.get_input(
            "데이터 파일 접두사", self.preprocessing_params['file_prefix']
        )
        self.preprocessing_params['interp_step'] = self.get_numeric_input(
            "보간 간격 (초)", self.preprocessing_params['interp_step'], min_val=0.0001, max_val=1.0
        )
        self.preprocessing_params['window_size'] = self.get_numeric_input(
            "이동 평균 윈도우 크기", self.preprocessing_params['window_size'], min_val=1, max_val=100
        )
        
        # 데이터 분할 비율 설정
        print("\n데이터 분할 비율 설정:")
        while True:
            train_ratio = self.get_numeric_input(
                "학습 데이터 비율", self.preprocessing_params['train_ratio'], min_val=0.1, max_val=0.9
            )
            val_ratio = self.get_numeric_input(
                "검증 데이터 비율", self.preprocessing_params['val_ratio'], min_val=0.05, max_val=0.5
            )
            test_ratio = self.get_numeric_input(
                "테스트 데이터 비율", self.preprocessing_params['test_ratio'], min_val=0.05, max_val=0.5
            )
            
            # 합이 1이 되는지 확인
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) < 0.001:  # 부동소수점 오차 허용
                self.preprocessing_params['train_ratio'] = train_ratio
                self.preprocessing_params['val_ratio'] = val_ratio
                self.preprocessing_params['test_ratio'] = test_ratio
                break
            else:
                self.show_warning(f"비율의 합이 1이 되어야 합니다. 현재 합: {total_ratio:.2f}")
        
        # 전처리 시작 확인
        print("\n전처리 설정이 완료되었습니다. 전처리를 시작합니다...")
        
        try:
            # 데이터 처리기 초기화
            processor = SensorDataProcessor(
                interpolation_step=self.preprocessing_params['interp_step'],
                window_size=self.preprocessing_params['window_size']
            )
            
            # 전처리 진행 표시
            self.show_message("\n[1/3] 센서 데이터 로드 및 보간 중...")
            interpolated_data = processor.load_and_interpolate_sensor_data(
                data_dir, prefix=self.preprocessing_params['file_prefix']
            )
            
            if not interpolated_data:
                self.show_error(f"센서 데이터를 로드할 수 없습니다. 파일 접두사 '{self.preprocessing_params['file_prefix']}'을(를) 확인하세요.")
                self.wait_for_user()
                return
            
            self.show_message("\n[2/3] 센서 데이터 결합 및 전처리 중...")
            processed_data = processor.combine_and_preprocess_sensor_data(interpolated_data)
            
            self.show_message("\n[3/3] 데이터 분할 및 결합 중...")
            train_data, valid_data, test_data = processor.split_and_combine_data(
                processed_data,
                train_ratio=self.preprocessing_params['train_ratio'],
                valid_ratio=self.preprocessing_params['val_ratio'],
                test_ratio=self.preprocessing_params['test_ratio']
            )
            
            # 결과 저장
            self.state['preprocessed_data'] = (train_data, valid_data, test_data)
            
            # 시퀀스 길이 설정 (전체 데이터 길이의 10% 정도로 제안)
            suggested_seq_len = min(int(train_data.shape[0] * 0.1), 100)
            suggested_seq_len = max(suggested_seq_len, 10)  # 최소값 확보
            # 5의 배수로 조정
            suggested_seq_len = (suggested_seq_len // 5) * 5
            if suggested_seq_len != self.model_params['sequence_length']:
                self.model_params['sequence_length'] = suggested_seq_len
            
            # 전처리 결과 출력
            self.show_success("\n전처리가 완료되었습니다.")
            self.show_message(f"- 학습 데이터: {train_data.shape} 샘플")
            self.show_message(f"- 검증 데이터: {valid_data.shape} 샘플")
            self.show_message(f"- 테스트 데이터: {test_data.shape} 샘플")
            self.show_message(f"- 제안된 시퀀스 길이: {self.model_params['sequence_length']}")
            
            # 전처리 데이터 저장 여부 확인
            save_data = self.get_yes_no_input("\n전처리된 데이터를 저장하시겠습니까?")
            if save_data:
                output_dir = self.paths['output_dir']
                os.makedirs(output_dir, exist_ok=True)
                
                current_time = time.strftime("%Y%m%d_%H%M%S")
                train_path = os.path.join(output_dir, f"train_data_{current_time}.npy")
                valid_path = os.path.join(output_dir, f"valid_data_{current_time}.npy")
                test_path = os.path.join(output_dir, f"test_data_{current_time}.npy")
                
                np.save(train_path, train_data)
                np.save(valid_path, valid_data)
                np.save(test_path, test_data)
                
                self.show_success("\n전처리된 데이터가 저장되었습니다:")
                self.show_message(f"- 학습 데이터: {train_path}")
                self.show_message(f"- 검증 데이터: {valid_path}")
                self.show_message(f"- 테스트 데이터: {test_path}")
            
        except Exception as e:
            self.show_error(f"\n전처리 중 예외가 발생했습니다: {str(e)}")
            logger.exception("전처리 중 예외 발생")
        
        self.wait_for_user()
    
    def model_params_menu(self) -> None:
        """모델 파라미터 설정 메뉴"""
        self.print_header("모델 파라미터 설정")
        
        print("LSTM 분류 모델의 파라미터를 설정합니다.\n")
        
        # 현재 설정 표시
        print("현재 설정:")
        print(f"- 시퀀스 길이: {self.model_params['sequence_length']}")
        print(f"- 은닉층 크기: {self.model_params['hidden_size']}")
        print(f"- LSTM 레이어 수: {self.model_params['num_layers']}")
        print(f"- 드롭아웃 비율: {self.model_params['dropout_rate']}")
        
        # 파라미터 설정
        print("\n새 파라미터 설정:")
        
        # 시퀀스 길이
        self.model_params['sequence_length'] = self.get_numeric_input(
            "시퀀스 길이", self.model_params['sequence_length'], min_val=5, max_val=200
        )
        
        # 은닉층 크기
        self.model_params['hidden_size'] = self.get_numeric_input(
            "은닉층 크기", self.model_params['hidden_size'], min_val=16, max_val=512
        )
        
        # LSTM 레이어 수
        self.model_params['num_layers'] = self.get_numeric_input(
            "LSTM 레이어 수", self.model_params['num_layers'], min_val=1, max_val=5
        )
        
        # 드롭아웃 비율
        self.model_params['dropout_rate'] = self.get_numeric_input(
            "드롭아웃 비율", self.model_params['dropout_rate'], min_val=0.0, max_val=0.9
        )
        
        # 학습 파라미터 설정
        print("\n학습 파라미터 설정:")
        
        # 배치 크기
        self.training_params['batch_size'] = self.get_numeric_input(
            "배치 크기", self.training_params['batch_size'], min_val=8, max_val=256
        )
        
        # 학습률
        self.training_params['learning_rate'] = self.get_numeric_input(
            "학습률", self.training_params['learning_rate'], min_val=0.0001, max_val=0.1
        )
        
        # 에폭 수
        self.training_params['epochs'] = self.get_numeric_input(
            "최대 에폭 수", self.training_params['epochs'], min_val=10, max_val=1000
        )
        
        # 조기 종료 인내 횟수
        self.training_params['patience'] = self.get_numeric_input(
            "조기 종료 인내 횟수", self.training_params['patience'], min_val=3, max_val=50
        )
        
        self.show_success("\n모델 파라미터 설정이 완료되었습니다.")
        
        self.wait_for_user()
    
    def train_model_menu(self) -> None:
        """모델 학습 메뉴"""
        self.print_header("모델 학습")
        
        # 데이터 확인
        if self.state['preprocessed_data'] is None:
            self.show_error("전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
            self.wait_for_user()
            return
        
        train_data, valid_data, test_data = self.state['preprocessed_data']
        
        # 학습 파라미터 확인
        print("학습에 사용할 설정:")
        print(f"- 장치: {self.state['device']}")
        print(f"- 시퀀스 길이: {self.model_params['sequence_length']}")
        print(f"- 은닉층 크기: {self.model_params['hidden_size']}")
        print(f"- LSTM 레이어 수: {self.model_params['num_layers']}")
        print(f"- 드롭아웃 비율: {self.model_params['dropout_rate']}")
        print(f"- 배치 크기: {self.training_params['batch_size']}")
        print(f"- 학습률: {self.training_params['learning_rate']}")
        print(f"- 최대 에폭 수: {self.training_params['epochs']}")
        print(f"- 조기 종료 인내 횟수: {self.training_params['patience']}")
        
        # 학습 시작 확인
        start_training = self.get_yes_no_input("\n위 설정으로 학습을 시작하시겠습니까?")
        if not start_training:
            self.show_message("학습을 취소합니다.")
            self.wait_for_user()
            return
        
        try:
            # 시퀀스 데이터 준비
            self.show_message("\n[1/6] 시퀀스 데이터 준비 중...")
            X_train, y_train = prepare_sequence_data(train_data, sequence_length=self.model_params['sequence_length'])
            X_valid, y_valid = prepare_sequence_data(valid_data, sequence_length=self.model_params['sequence_length'])
            X_test, y_test = prepare_sequence_data(test_data, sequence_length=self.model_params['sequence_length'])
            
            self.show_message(f"시퀀스 데이터 준비 완료:")
            self.show_message(f"- 학습 데이터: {X_train.shape}, 레이블: {y_train.shape}")
            self.show_message(f"- 검증 데이터: {X_valid.shape}, 레이블: {y_valid.shape}")
            self.show_message(f"- 테스트 데이터: {X_test.shape}, 레이블: {y_test.shape}")
            
            # 클래스 분포 시각화를 위한 임시 디렉토리
            plot_dir = self.paths['plot_dir']
            os.makedirs(plot_dir, exist_ok=True)
            
            self.show_message("\n[2/6] 데이터 분포 분석 중...")
            plot_class_distribution(y_train, plot_dir, filename='train_class_distribution.png')
            self.show_message(f"학습 데이터 클래스 분포 시각화 저장: {os.path.join(plot_dir, 'train_class_distribution.png')}")
            
            # 데이터 로더 준비
            self.show_message("\n[3/6] 데이터 로더 준비 중...")
            train_loader, val_loader = prepare_dataloaders(
                X_train, y_train, X_valid, y_valid, self.state['device'],
                batch_size=self.training_params['batch_size']
            )
            
            # 모델 초기화
            self.show_message("\n[4/6] 모델 초기화 중...")
            input_size = X_train.shape[2]  # 특성 수
            num_classes = len(np.unique(y_train))  # 클래스 수
            
            model = MultiSensorLSTMClassifier(
                input_size=input_size,
                hidden_size=self.model_params['hidden_size'],
                num_layers=self.model_params['num_layers'],
                num_classes=num_classes,
                dropout_rate=self.model_params['dropout_rate']
            ).to(self.state['device'])
            
            # 모델 정보 출력
            model_info = model.get_model_info()
            self.show_message(f"모델 초기화 완료:")
            self.show_message(f"- 모델 유형: {model_info['model_type']}")
            self.show_message(f"- 입력 크기: {model_info['input_size']}")
            self.show_message(f"- 은닉층 크기: {model_info['hidden_size']}")
            self.show_message(f"- LSTM 레이어 수: {model_info['num_layers']}")
            self.show_message(f"- 출력 클래스 수: {model_info['num_classes']}")
            self.show_message(f"- 파라미터 수: {model_info['parameter_count']:,}")
            
            # 모델 학습
            self.show_message("\n[5/6] 모델 학습 중...")
            model, history = train_model(
                train_loader=train_loader,
                valid_loader=val_loader,
                model=model,
                device=self.state['device'],
                num_epochs=self.training_params['epochs'],
                learning_rate=self.training_params['learning_rate'],
                early_stopping_patience=self.training_params['patience'],
                model_dir=self.paths['model_dir']
            )
            
            # 상태 업데이트
            self.state['model'] = model
            self.state['training_history'] = history
            
            # 시험 데이터 로더
            test_loader, _ = prepare_dataloaders(
                X_test, y_test, X_test[:1], y_test[:1], self.state['device'],
                batch_size=self.training_params['batch_size']
            )
            
            # 학습 이력 시각화
            self.show_message("\n[6/6] 학습 결과 시각화 중...")
            history_path = plot_training_history(history, plot_dir)
            self.show_message(f"학습 이력 시각화 저장: {history_path}")
            
            # 학습 결과 출력
            self.show_success("\n모델 학습이 완료되었습니다.")
            
            # 최종 성능 출력
            val_acc = history['valid_accuracy'][-1]
            val_loss = history['valid_loss'][-1]
            self.show_message(f"\n최종 검증 성능:")
            self.show_message(f"- 검증 정확도: {val_acc:.4f}")
            self.show_message(f"- 검증 손실: {val_loss:.4f}")
            
            # 모델 저장 경로
            model_save_path = os.path.join(
                self.paths['model_dir'],
                f"sensor_classifier_{time.strftime('%Y%m%d_%H%M%S')}.pth"
            )
            
            # 모델 저장
            torch.save(model.state_dict(), model_save_path)
            self.state['current_model_path'] = model_save_path
            self.show_message(f"\n모델이 저장되었습니다: {model_save_path}")
            
            # 모델 정보 저장
            model_info_path = save_model_info(model, self.paths['model_dir'], self.model_params['sequence_length'])
            self.show_message(f"모델 정보가 저장되었습니다: {model_info_path}")
            
            # 시각화 결과 확인 안내
            self.show_message(f"\n학습 이력 시각화는 '{history_path}'에서 확인할 수 있습니다.")
            
        except Exception as e:
            self.show_error(f"\n모델 학습 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 학습 중 예외 발생")
        
        self.wait_for_user()
    
    def evaluate_model_menu(self) -> None:
        """모델 평가 메뉴"""
        self.print_header("모델 평가")
    
    # 모델 확인
        if self.state['model'] is None:
            print("❌ 오류: 학습된 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
    
    # 데이터 확인
        if self.state['preprocessed_data'] is None:
            print("❌ 오류: 전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
    
        train_data, valid_data, test_data = self.state['preprocessed_data']
    
        try:
            # 테스트 데이터 준비
            print("\n[1/4] 테스트 데이터 준비 중...")
            X_test, y_test = prepare_sequence_data(test_data, sequence_length=self.state['model_params']['sequence_length'])
            print(f"테스트 데이터 준비 완료: {X_test.shape}, 레이블: {y_test.shape}")
        
        # 테스트 데이터 로더 준비
            print("\n[2/4] 테스트 데이터 로더 준비 중...")
            test_loader, _ = prepare_dataloaders(
                X_test, y_test, X_test[:1], y_test[:1], self.state['device'],
                batch_size=self.state['training_params']['batch_size']
            )
        
            # 모델 평가
            print("\n[3/4] 모델 평가 중...")
            evaluation_result = evaluate_model(self.state['model'], test_loader, self.state['device'])
        
            # 상태 업데이트
            self.state['evaluation_result'] = evaluation_result
        
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
        
            model = self.state['model']
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    test_predictions.extend(predicted.cpu().numpy())
                    test_labels.extend(labels.cpu().numpy())
        
            # 혼동 행렬 시각화
            cm_path = plot_confusion_matrix(test_labels, test_predictions, self.paths['plot_dir'])
            print(f"혼동 행렬 시각화 저장: {cm_path}")
        
            # 어텐션 가중치 시각화
            sample_inputs = torch.from_numpy(X_test[:10]).to(self.state['device'])
            sample_labels = torch.from_numpy(y_test[:10]).to(self.state['device'])
        
            attn_path = plot_attention_weights(
                model=model,
                data=sample_inputs,
                labels=sample_labels,
                plot_dir=self.paths['plot_dir']
                )
            print(f"어텐션 가중치 시각화 저장: {attn_path}")
        
            # 특성 중요도 시각화
            feature_path = plot_feature_importance(
                model=model,
                data=sample_inputs,
                labels=sample_labels,
                plot_dir=self.paths['plot_dir']
            )
            print(f"특성 중요도 시각화 저장: {feature_path}")
        
            # 평가 결과 저장
            eval_path = save_evaluation_result(evaluation_result, self.paths['output_dir'])
            print(f"\n평가 결과가 저장되었습니다: {eval_path}")
        
        except Exception as e:
            print(f"\n❌ 오류: 모델 평가 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 평가 중 예외 발생")
    
        input("\n계속하려면 Enter 키를 누르세요...")

    def deploy_model_menu(self) -> None:
        """모델 배포 메뉴"""
        self.print_header("모델 배포")
    
        # 모델 확인
        if self.state['model'] is None or self.paths['current_model_path'] is None:
            print("❌ 오류: 배포할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
    
        print("모델 배포는 학습된 모델을 배포 디렉토리에 복사하고")
        print("추론을 위한 필요한 파일들을 준비하는 단계입니다.\n")
    
        # 배포 디렉토리 설정
        deploy_dir = BaseCLI.get_input("배포 디렉토리 경로", self.paths['deploy_dir'])
    
        try:
            # 배포 디렉토리 생성
            os.makedirs(deploy_dir, exist_ok=True)
        
            # 타임스탬프로 하위 디렉토리 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            deploy_subdir = os.path.join(deploy_dir, f"deployment_{timestamp}")
            os.makedirs(deploy_subdir, exist_ok=True)
        
            print("\n[1/5] 모델 파일 복사 중...")
            model_filename = os.path.basename(self.state['current_model_path'])
            deploy_model_path = os.path.join(deploy_subdir, model_filename)
            shutil.copy2(self.state['current_model_path'], deploy_model_path)
            print(f"모델 파일 복사 완료: {deploy_model_path}")
        
            # 모델 정보 파일 복사
            print("\n[2/5] 모델 정보 파일 복사 중...")
            model_info_src = os.path.join(self.paths['model_dir'], 'model_info.json')
            model_info_dst = os.path.join(deploy_subdir, 'model_info.json')
            if os.path.exists(model_info_src):
                shutil.copy2(model_info_src, model_info_dst)
                print(f"모델 정보 파일 복사 완료: {model_info_dst}")
            else:
                # 모델 정보 파일이 없으면 새로 생성
                model_info = self.state['model'].get_model_info()
                model_info.update({
                    "sequence_length": self.state['model_params']['sequence_length'],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                })
                with open(model_info_dst, 'w') as f:
                    json.dump(model_info, f, indent=4)
                print(f"모델 정보 파일 생성 완료: {model_info_dst}")
        
            # 전처리 설정 저장
            print("\n[3/5] 전처리 설정 저장 중...")
            preprocess_config = {
                "file_prefix": self.state['preprocessing_params']['file_prefix'],
                "interp_step": self.state['preprocessing_params']['interp_step'],
                "window_size": self.state['preprocessing_params']['window_size'],
                "sequence_length": self.state['model_params']['sequence_length']
            }
            preprocess_config_path = os.path.join(deploy_subdir, 'preprocess_config.json')
            with open(preprocess_config_path, 'w') as f:
                json.dump(preprocess_config, f, indent=4)
            print(f"전처리 설정 저장 완료: {preprocess_config_path}")
        
            # 평가 결과 복사 (있는 경우)
            print("\n[4/5] 평가 결과 복사 중...")
            if self.state['evaluation_result'] is not None:
                eval_result_path = os.path.join(deploy_subdir, 'evaluation_result.json')
                with open(eval_result_path, 'w') as f:
                # NumPy 배열을 일반 리스트로 변환
                    eval_result = self.state['evaluation_result'].copy()
                    eval_result_path = os.path.join(deploy_subdir, 'evaluation_result.json')
                    with open(eval_result_path, 'w') as f:
                        json.dump(eval_result, f, indent=4)
                    print(f"평가 결과 저장 완료: {eval_result_path}")
            else:
                print("평가 결과가 없어 복사를 건너뜁니다.")
        
            # 시각화 결과 복사
            print("\n[5/5] 시각화 결과 복사 중...")
            plot_src_dir = self.paths['plot_dir']
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
            inference_script = """#!/usr/bin/env python"""

        
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
- 모델 유형: {self.state['model'].get_model_info()['model_type']}
- 입력 크기: {self.state['model'].get_model_info()['input_size']}
- 은닉층 크기: {self.state['model'].get_model_info()['hidden_size']}
- LSTM 레이어 수: {self.state['model'].get_model_info()['num_layers']}
- 시퀀스 길이: {self.state['model_params']['sequence_length']}
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

    def load_data_menu(self) -> None:
        """저장된 데이터 로드 메뉴"""
        self.print_header("저장된 데이터 로드")
    
        print("이전에 저장한 전처리 데이터를 로드합니다.\n")

        # 저장된 데이터 파일 목록 표시
        output_dir = self.paths['output_dir']
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
            choice = self.get_input("\n로드할 데이터 세트 번호를 입력하세요", "1")
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
            self.paths['preprocessed_data'] = (train_data, valid_data, test_data)
        
            # 시퀀스 길이 제안
            suggested_seq_len = min(int(train_data.shape[0] * 0.1), 100)
            suggested_seq_len = max(suggested_seq_len, 10)  # 최소값 확보
            # 5의 배수로 조정
            suggested_seq_len = (suggested_seq_len // 5) * 5
            self.model_params['sequence_length'] = suggested_seq_len
        
            print("\n✅ 데이터 로드가 완료되었습니다.")
            print(f"- 학습 데이터: {train_data.shape} 샘플")
            print(f"- 검증 데이터: {valid_data.shape} 샘플")
            print(f"- 테스트 데이터: {test_data.shape} 샘플")
            print(f"- 제안된 시퀀스 길이: {suggested_seq_len}")
        
        except Exception as e:
            print(f"\n❌ 오류: 데이터 로드 중 예외가 발생했습니다: {str(e)}")
            logger.exception("데이터 로드 중 예외 발생")
    
        input("\n계속하려면 Enter 키를 누르세요...")
    def load_model_menu(self) -> None:
        """저장된 모델 로드 메뉴"""
        self.print_header("저장된 모델 로드")
    
        print("이전에 저장한 모델을 로드합니다.\n")
    
        # 저장된 모델 파일 목록 표시
        model_dir = self.paths['model_dir']
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
            choice = self.get_input("\n로드할 모델 번호를 입력하세요", "1")
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
            input_size = BaseCLI.get_numeric_input("입력 특성 수", 4, min_val=1)
            hidden_size = BaseCLI.get_numeric_input("은닉층 크기", self.state['model_params']['hidden_size'], min_val=8)
            num_layers = BaseCLI.get_numeric_input("LSTM 레이어 수", self.state['model_params']['num_layers'], min_val=1, max_val=5)
            num_classes =BaseCLI.get_numeric_input("출력 클래스 수", 4, min_val=2)
            sequence_length = BaseCLI.get_numeric_input("시퀀스 길이", self.state['model_params']['sequence_length'], min_val=5)

            # 모델 파라미터 업데이트
            self.state['model_params']['hidden_size'] = hidden_size
            self.state['model_params']['num_layers'] = num_layers
            self.state['model_params']['sequence_length'] = sequence_length
        
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
            self.model_params['hidden_size'] = model_info.get('hidden_size', self.model_params['hidden_size'])
            self.model_params['num_layers'] = model_info.get('num_layers', self.model_params['num_layers'])
            self.model_params['sequence_length'] = model_info.get('sequence_length', self.model_params['sequence_length'])
        try:
            print(f"\n모델 '{selected_model}' 로드 중...")
        
            # 모델 초기화
            model = MultiSensorLSTMClassifier(
                input_size=model_info['input_size'],
                hidden_size=model_info['hidden_size'],
                num_layers=model_info['num_layers'],
                num_classes=model_info['num_classes'],
                dropout_rate=self.model_params['dropout_rate']
            ).to(self.state['device'])
        
            # 모델 가중치 로드
            model_path = os.path.join(model_dir, selected_model)
            model.state_dict(torch.load(model_path, map_location=self.state['device']))
            model.eval()
        
            # 상태 업데이트
            self.state['model'] = model
            self.state['current_model_path'] = model_path
        
            print("\n✅ 모델 로드가 완료되었습니다.")
            print(f"- 모델 파일: {selected_model}")
            print(f"- 입력 크기: {model_info['input_size']}")
            print(f"- 은닉층 크기: {model_info['hidden_size']}")
            print(f"- LSTM 레이어 수: {model_info['num_layers']}")
            print(f"- 출력 클래스 수: {model_info['num_classes']}")
            print(f"- 시퀀스 길이: {model_info['sequence_length']}")
        
        except Exception as e:
            print(f"\n❌ 오류: 모델 로드 중 예외가 발생했습니다: {str(e)}")
            logger.exception("모델 로드 중 예외 발생")
    
        input("\n계속하려면 Enter 키를 누르세요...")

    def system_config_menu(self) -> None:
        """시스템 설정 메뉴"""
        self.print_header("시스템 설정")
    
        print("시스템 설정을 변경합니다.\n")
    
        # 디렉토리 설정
        print("디렉토리 설정:")
        for dir_name, dir_path in self.paths.items():
            new_path = BaseCLI.get_input(f"{dir_name} 디렉토리", dir_path)
            self.paths[dir_name] = new_path
            os.makedirs(new_path, exist_ok=True)
    
        # 장치 설정
        if torch.cuda.is_available():
            use_gpu = self.get_yes_no_input("\nGPU를 사용하시겠습니까?", default=True)
            self.state['device'] = torch.device("cuda" if use_gpu else "cpu")
        else:
            print("\nGPU를 사용할 수 없습니다. CPU를 사용합니다.")
            self.state['device'] = torch.device("cpu")
    
            print(f"\n✅ 시스템 설정이 변경되었습니다.")
            print(f"- 현재 장치: {self.state['device']}")
            for dir_name, dir_path in self.paths.items():
                print(f"- {dir_name} 디렉토리: {dir_path}")
    
            input("\n계속하려면 Enter 키를 누르세요...")
    def ensure_dirs(self) -> None:
        """필요한 디렉토리 생성"""
        for dir_path in self.paths.values():
            os.makedirs(dir_path, exist_ok=True)

    def _ensure_db_connection(self) -> bool:
        """
        데이터베이스 연결 확인 및 필요시 연결 설정
        
        Returns:
            bool: 연결 성공 여부
        """
        # 이미 연결되어 있으면 True 반환
        if self.db_connector and self.state['db_connected']:
            return True
            
        # 연결 설정 메뉴 표시
        self.print_header("데이터베이스 연결 설정")
        
        print("데이터베이스 연결 정보를 설정합니다.\n")
        
        # 프로필 선택
        profiles = ["default", "development", "production"]
        # 설정 파일 확인
        config_path = os.path.join(project_root, "config", "db_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                if "connections" in config:
                    profiles = list(config["connections"].keys())
            except:
                pass  # 오류 발생 시 기본 프로필 사용
        
        print("설정 프로필 선택:")
        for i, profile in enumerate(profiles, 1):
            print(f"{i}. {profile}")
        
        profile_choice = self.get_input("선택할 프로필 번호", "1")
        try:
            profile_idx = int(profile_choice) - 1
            if 0 <= profile_idx < len(profiles):
                selected_profile = profiles[profile_idx]
            else:
                self.show_error(f"유효한 번호를 입력하세요 (1-{len(profiles)})")
                return False
        except ValueError:
            self.show_error("숫자를 입력하세요")
            return False
        
        # DB 커넥터 초기화 (아직 연결은 안 함)
        if not self.db_connector:
            self.db_connector = DBConnector(config_profile=selected_profile)
        
        # 데이터베이스 유형 선택
        print("\n데이터베이스 유형 선택:")
        db_types = ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Oracle"]
        db_type_idx = self.show_menu(db_types, "데이터베이스 유형")
        db_type_lower = db_types[db_type_idx].lower()
        
        # SQLite는 파일 경로만 필요
        if db_type_lower == 'sqlite':
            db_file = self.get_input("SQLite 데이터베이스 파일 경로", "database.db")
            
            try:
                # 연결 시도
                if self.db_connector.connect(db_type_lower, database=db_file):
                    self.state['db_connected'] = True
                    self.show_success(f"SQLite 데이터베이스 '{db_file}'에 연결되었습니다.")
                    return True
                else:
                    self.show_error("SQLite 데이터베이스 연결 실패")
                    return False
            except Exception as e:
                self.show_error(f"SQLite 데이터베이스 연결 실패: {str(e)}")
                logger.exception("SQLite 연결 실패")
                return False
        
        # 다른 데이터베이스는 연결 정보 필요
        print(f"\n{db_types[db_type_idx]} 연결 정보 입력:")
        
        # 연결 정보 입력 (설정 파일의 값을 기본값으로 사용)
        connection_params = self.db_connector.get_connection_params()
        
        host = self.get_input("호스트", connection_params.get('host', 'localhost'))
        
        # 포트 기본값 설정
        default_ports = {
            'mysql': 3306, 'postgresql': 5432, 'sqlserver': 1433, 'oracle': 1521
        }
        default_port = default_ports.get(db_type_lower, 3306)
        
        # 포트 처리
        if 'port' in connection_params and isinstance(connection_params['port'], dict):
            if db_type_lower in connection_params['port']:
                default_port = connection_params['port'][db_type_lower]
        
        port = self.get_numeric_input("포트", default_port, min_val=1, max_val=65535)
        database = self.get_input("데이터베이스 이름", connection_params.get('database', ''))
        username = self.get_input("사용자 이름", connection_params.get('username', ''))
        
        # 비밀번호는 설정에 값이 있어도 표시하지 않음
        password = self.get_input("비밀번호 (입력하지 않으면 설정 파일의 값 사용)")
        if not password and 'password' in connection_params:
            password = connection_params['password']
        
        # 데이터베이스 연결 시도
        try:
            if self.db_connector.connect(
                db_type=db_type_lower,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password
            ):
                self.state['db_connected'] = True
                self.show_success(f"{db_types[db_type_idx]} 데이터베이스에 연결되었습니다.")
                return True
            else:
                self.show_error("데이터베이스 연결 실패")
                return False
        except Exception as e:
            self.show_error(f"데이터베이스 연결 실패: {str(e)}")
            logger.exception("데이터베이스 연결 실패")
            return False
    
    def db_integration_menu(self) -> None:
        """데이터베이스 통합 메뉴"""
        while True:
            self.print_header("데이터베이스 통합")
            
            print("모델 결과 및 센서 데이터를 데이터베이스에 저장합니다.\n")
            
            menu_options = [
                "데이터베이스 연결 설정",
                "센서 데이터 저장",
                "예측 이력 저장",
                "돌아가기"
            ]
            
            choice = self.show_menu(menu_options, "데이터 DB로 전송")
            
            if choice == 1:
                # 데이터베이스 연결 설정
                self._ensure_db_connection()
            elif choice == 2:
                # 센서 데이터 저장 기능
                self.show_message("모델 평가 결과 저장 기능은 아직 구현되지 않았습니다.")
                # 이 부분은 다음 단계에서 구현할 예정입니다
            elif choice == 3:
                # 예측 이력 저장
                self.show_message("예측 이력 저장 기능은 아직 구현되지 않았습니다.")
                # 이 부분은 다음 단계에서 구현할 예정입니다
            elif choice == 4:
                # 메인 메뉴로 돌아가기
                return

    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header("다중 센서 데이터 분류 시스템")
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
            print("9. 데이터 DB로 전송")
            print("0. 종료\n")
        
            self.print_status()
        
            choice = BaseCLI.get_input("\n메뉴 선택", "0")
        
            if choice == "1":
                self.preprocess_data_menu()
            elif choice == "2":
                self.model_params_menu()
            elif choice == "3":
                self.train_model_menu()
            elif choice == "4":
                self.evaluate_model_menu()
            elif choice == "5":
                self.deploy_model_menu()
            elif choice == "6":
                self.load_data_menu()
            elif choice == "7":
                self.load_model_menu()
            elif choice == "8":
                self.system_config_menu()
            elif choice == "9":
                self.db_integration_menu()  # 새로운 메뉴 메서드 호출
            elif choice == "0":
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
            else:
                print("\n유효하지 않은 선택입니다. 다시 시도하세요.")
                input("계속하려면 Enter 키를 누르세요...")
    def run(self) -> None:
        """CLI 실행"""
        try:
        # 필요한 디렉토리 생성
            self.ensure_dirs()
        
            # 메인 메뉴 실행
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.show_error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            logger.exception("예상치 못한 오류 발생")
    

    def main(self):
        """메인 함수"""
        try:
            # 필요한 디렉토리 생성
            self.ensure_dirs()
        
            # 로깅 설정
            log_dir = os.path.join(project_root, 'logs')
            os.makedirs(log_dir, exist_ok=True)
        
            # 메인 메뉴 실행
            SensorCLI.main_menu()
        
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
            logger.exception("예상치 못한 오류 발생")
            sys.exit(1)

if __name__ == "__main__":
    SensorCLI.main()