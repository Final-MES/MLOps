#!/usr/bin/env python
"""
이미지 데이터 분석 시스템 CLI 인터페이스

이 스크립트는 이미지 데이터 전처리, 모델 학습, 평가, 배포를 위한
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
from src.data.image.preprocessor import ImagePreprocessor
from src.data.image.augmentation import ImageAugmentor
from src.utils.visualization import (
    plot_training_history, plot_confusion_matrix, plot_class_distribution
)
from src.cli.base_cli import BaseCLI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'image_cli.log'))
    ]
)
logger = logging.getLogger(__name__)

class ImageCLI(BaseCLI):
    """
    이미지 데이터 분석 CLI 클래스
    
    이 클래스는 이미지 데이터 분석을 위한 대화형 명령줄 인터페이스를 제공합니다.
    """
    
    def __init__(self):
        """이미지 CLI 초기화"""
        super().__init__(title="이미지 데이터 분석 시스템")
        
        # 상태 초기화
        self.state = {
            'preprocessed_data': None,
            'model': None,
            'training_history': None,
            'evaluation_result': None,
            'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            'model_params': {
                'model_type': 'cnn',  # 'cnn', 'vgg', 'resnet' 등
                'input_size': (224, 224),
                'num_channels': 3,
                'num_classes': 0,  # 데이터 로드 후 자동 설정
                'hidden_size': 128,
                'dropout_rate': 0.5
            },
            'training_params': {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 50,
                'patience': 5
            },
            'preprocessing_params': {
                'target_size': (224, 224),
                'normalize': True,
                'augmentation': True,
                'train_ratio': 0.7,
                'val_ratio': 0.15,
                'test_ratio': 0.15
            },
            'current_model_path': None
        }
        
        logger.info("이미지 데이터 분석 CLI 초기화 완료")
    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header("이미지 데이터 분석 시스템")
            print("이미지 데이터 분석 파이프라인 관리 시스템에 오신 것을 환영합니다.")
            print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
            
            menu_options = [
                "데이터 로드 및 전처리",
                "모델 파라미터 설정",
                "모델 학습",
                "모델 평가",
                "모델 배포",
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
                self.system_config_menu()
            elif choice == 6:
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
    
    def preprocess_data_menu(self) -> None:
        """데이터 로드 및 전처리 메뉴"""
        self.print_header("데이터 로드 및 전처리")
        
        print("이미지 데이터를 로드하고 전처리합니다.")
        print("이 단계에서는 이미지 크기 조정, 정규화, 증강을 수행합니다.\n")
        
        # 데이터 디렉토리 설정
        data_dir = self.get_input("이미지 데이터 디렉토리 경로", self.paths["data"] / "raw" / "images")
        if not os.path.exists(data_dir):
            self.show_error(f"디렉토리 '{data_dir}'이(가) 존재하지 않습니다.")
            create_dir = self.get_yes_no_input("디렉토리를 생성하시겠습니까?")
            if create_dir:
                os.makedirs(data_dir, exist_ok=True)
                self.show_success(f"디렉토리 '{data_dir}'이(가) 생성되었습니다.")
            else:
                self.show_error("전처리를 취소합니다.")
                return
        
        # 데이터셋 구조 확인
        print("\n데이터셋 구조를 확인하는 중...")
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if not subdirs:
            self.show_error(f"'{data_dir}' 디렉토리에 클래스 하위 디렉토리가 없습니다.")
            self.show_message("이미지 분류를 위해서는 각 클래스별로 하위 디렉토리가 필요합니다.")
            self.show_message("예: /데이터/강아지/, /데이터/고양이/ 등")
            return
        
        self.show_message(f"발견된 클래스: {len(subdirs)}개")
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(data_dir, subdir)
            image_files = [f for f in os.listdir(subdir_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            self.show_message(f"- {subdir}: {len(image_files)}개 이미지")
        
        # 전처리 파라미터 설정
        print("\n전처리 파라미터 설정:")
        
        # 이미지 크기 설정
        current_size = self.state['preprocessing_params']['target_size']
        width = self.get_numeric_input(
            "이미지 너비", current_size[0], min_val=32, max_val=1024
        )
        height = self.get_numeric_input(
            "이미지 높이", current_size[1], min_val=32, max_val=1024
        )
        self.state['preprocessing_params']['target_size'] = (width, height)
        self.state['model_params']['input_size'] = (width, height)
        
        # 정규화 설정
        self.state['preprocessing_params']['normalize'] = self.get_yes_no_input(
            "이미지 정규화 적용", self.state['preprocessing_params']['normalize']
        )
        
        # 증강 설정
        self.state['preprocessing_params']['augmentation'] = self.get_yes_no_input(
            "데이터 증강 적용", self.state['preprocessing_params']['augmentation']
        )
        
        # 데이터 분할 비율
        print("\n데이터 분할 비율 설정:")
        while True:
            train_ratio = self.get_numeric_input(
                "학습 데이터 비율", self.state['preprocessing_params']['train_ratio'], 
                min_val=0.1, max_val=0.9
            )
            val_ratio = self.get_numeric_input(
                "검증 데이터 비율", self.state['preprocessing_params']['val_ratio'], 
                min_val=0.05, max_val=0.5
            )
            test_ratio = self.get_numeric_input(
                "테스트 데이터 비율", self.state['preprocessing_params']['test_ratio'], 
                min_val=0.05, max_val=0.5
            )
            
            # 합이 1이 되는지 확인
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) < 0.001:  # 부동소수점 오차 허용
                self.state['preprocessing_params']['train_ratio'] = train_ratio
                self.state['preprocessing_params']['val_ratio'] = val_ratio
                self.state['preprocessing_params']['test_ratio'] = test_ratio
                break
            else:
                self.show_error(f"비율의 합이 1이 되어야 합니다. 현재 합: {total_ratio:.2f}")
        
        # 전처리 시작 확인
        start_preprocessing = self.get_yes_no_input("\n위 설정으로 전처리를 시작하시겠습니까?")
        if not start_preprocessing:
            self.show_message("전처리를 취소합니다.")
            return
        
        try:
            # 이미지 전처리기 초기화
            preprocessor = ImagePreprocessor(
                target_size=self.state['preprocessing_params']['target_size'],
                normalize=self.state['preprocessing_params']['normalize'],
                color_mode='rgb'
            )
            
            self.show_message("\n[1/4] 이미지 데이터 로드 중...")
            
            # 데이터셋 생성
            dataset_info = preprocessor.create_dataset(
                root_dir=data_dir,
                test_split=self.state['preprocessing_params']['test_ratio'] / 
                            (1 - self.state['preprocessing_params']['train_ratio'])
            )
            
            if not dataset_info:
                self.show_error("데이터셋 생성에 실패했습니다.")
                return
            
            # 클래스 목록 및 수 설정
            class_names = dataset_info['class_names']
            num_classes = len(class_names)
            self.state['model_params']['num_classes'] = num_classes
            
            self.show_message(f"\n[2/4] 클래스 정보: {num_classes}개 클래스 발견")
            for i, class_name in enumerate(class_names):
                self.show_message(f"- 클래스 {i}: {class_name}")
            
            # 데이터 증강 설정
            if self.state['preprocessing_params']['augmentation']:
                self.show_message("\n[3/4] 데이터 증강 적용 중...")
                
                # 증강기 초기화
                augmentor = ImageAugmentor(seed=42)
                
                # 학습 데이터 증강
                x_train_augmented = []
                y_train_augmented = []
                
                # 기존 학습 데이터 포함
                x_train_augmented.extend(dataset_info['x_train'])
                y_train_augmented.extend(dataset_info['y_train'])
                
                # 각 클래스별로 증강 데이터 생성
                for class_idx in range(num_classes):
                    # 해당 클래스의 이미지만 선택
                    class_indices = [i for i, y in enumerate(dataset_info['y_train']) if y == class_idx]
                    class_images = [dataset_info['x_train'][i] for i in class_indices]
                    
                    # 이미지 수에 따라 증강 수 조절
                    num_augmentations = max(1, 500 // len(class_images))  # 최대 약 500개까지 증강
                    
                    for img in class_images:
                        # 각 이미지당 여러 증강 버전 생성
                        for _ in range(num_augmentations):
                            aug_img = augmentor.augment(img)
                            x_train_augmented.append(aug_img)
                            y_train_augmented.append(class_idx)
                
                # 증강된 데이터로 기존 학습 데이터 대체
                dataset_info['x_train'] = np.array(x_train_augmented)
                dataset_info['y_train'] = np.array(y_train_augmented)
                
                self.show_message(f"데이터 증강 완료: {len(x_train_augmented)}개 학습 이미지 (증강 포함)")
            
            # PyTorch 데이터셋 및 데이터 로더 준비
            self.show_message("\n[4/4] PyTorch 데이터셋 및 로더 준비 중...")
            
            # 다음 단계에서 학습에 사용할 데이터 저장
            self.state['preprocessed_data'] = {
                'x_train': dataset_info['x_train'],
                'y_train': dataset_info['y_train'],
                'x_test': dataset_info['x_test'],
                'y_test': dataset_info['y_test'],
                'class_names': dataset_info['class_names'],
                'train_dataset': None,  # 데이터 로더 생성시 설정
                'val_dataset': None,    # 데이터 로더 생성시 설정
                'test_dataset': None    # 데이터 로더 생성시 설정
            }
            
            # 학습/검증 데이터 분리
            val_size = self.state['preprocessing_params']['val_ratio'] / (
                self.state['preprocessing_params']['train_ratio'] + 
                self.state['preprocessing_params']['val_ratio']
            )
            
            # 데이터셋을 분할하여 저장
            from sklearn.model_selection import train_test_split
            X_train_val, X_test = dataset_info['x_train'], dataset_info['x_test']
            y_train_val, y_test = dataset_info['y_train'], dataset_info['y_test']
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=val_size, stratify=y_train_val, random_state=42
            )
            
            # pytorch 데이터셋 및 로더 생성 (실제 학습 시 초기화)
            from torch.utils.data import TensorDataset
            
            # 데이터셋 생성하여 저장
            self.state['preprocessed_data']['train_dataset'] = (X_train, y_train)
            self.state['preprocessed_data']['val_dataset'] = (X_val, y_val)
            self.state['preprocessed_data']['test_dataset'] = (X_test, y_test)
            
            # 클래스 분포 시각화
            plot_dir = self.paths["plot_dir"]
            os.makedirs(plot_dir, exist_ok=True)
            
            plot_class_distribution(
                y_train, 
                plot_dir=plot_dir, 
                filename='train_class_distribution.png',
                class_names=class_names
            )
            
            self.show_success("\n전처리가 완료되었습니다.")
            self.show_message(f"- 학습 데이터: {len(X_train)}개 이미지")
            self.show_message(f"- 검증 데이터: {len(X_val)}개 이미지")
            self.show_message(f"- 테스트 데이터: {len(X_test)}개 이미지")
            self.show_message(f"- 클래스 수: {num_classes}")
            
            # 요약 정보 업데이트
            self.update_state('preprocessed_data', self.state['preprocessed_data'])
            
        except Exception as e:
            self.show_error(f"데이터 전처리 중 오류 발생: {str(e)}")
            logger.exception("데이터 전처리 중 예외 발생")
        
        input("\n계속하려면 Enter 키를 누르세요...")
    
    def model_params_menu(self) -> None:
        """모델 파라미터 설정 메뉴"""
        self.print_header("모델 파라미터 설정")
        
        print("이미지 분류 모델의 파라미터를 설정합니다.\n")
        
        # 사전에 데이터가 로드되었는지 확인
        if self.state['preprocessed_data'] is None:
            self.show_warning("데이터가 로드되지 않았습니다. 일부 설정이 제한될 수 있습니다.")
        else:
            num_classes = self.state['model_params']['num_classes']
            self.show_message(f"데이터셋 정보: {num_classes}개 클래스")
        
        # 현재 설정 표시
        print("\n현재 모델 설정:")
        print(f"- 모델 유형: {self.state['model_params']['model_type']}")
        print(f"- 입력 크기: {self.state['model_params']['input_size']}")
        print(f"- 채널 수: {self.state['model_params']['num_channels']}")
        print(f"- 은닉층 크기: {self.state['model_params']['hidden_size']}")
        print(f"- 드롭아웃 비율: {self.state['model_params']['dropout_rate']}")
        
        # 모델 유형 선택
        print("\n모델 유형 선택:")
        model_types = ["CNN (기본)", "VGG16", "ResNet18", "MobileNetV2"]
        model_type_idx = self.show_menu(model_types, "모델 유형")
        
        model_type_map = {
            0: "cnn",
            1: "vgg16",
            2: "resnet18",
            3: "mobilenetv2"
        }
        
        self.state['model_params']['model_type'] = model_type_map[model_type_idx]
        
        # 입력 채널 설정
        self.state['model_params']['num_channels'] = self.get_numeric_input(
            "입력 채널 수 (1: 흑백, 3: 컬러)", 
            self.state['model_params']['num_channels'],
            min_val=1, 
            max_val=3
        )
        
        # 모델별 추가 파라미터 설정
        if self.state['model_params']['model_type'] == 'cnn':
            # 기본 CNN의 경우 은닉층 크기와 드롭아웃 설정
            self.state['model_params']['hidden_size'] = self.get_numeric_input(
                "은닉층 크기", 
                self.state['model_params']['hidden_size'],
                min_val=32, 
                max_val=1024
            )
            
            self.state['model_params']['dropout_rate'] = self.get_numeric_input(
                "드롭아웃 비율", 
                self.state['model_params']['dropout_rate'],
                min_val=0.0, 
                max_val=0.9
            )
        
        elif self.state['model_params']['model_type'] in ['vgg16', 'resnet18', 'mobilenetv2']:
            # 사전 학습 여부 설정
            use_pretrained = self.get_yes_no_input(
                "사전 학습된 가중치를 사용하시겠습니까?", 
                default=True
            )
            
            self.state['model_params']['pretrained'] = use_pretrained
            
            if use_pretrained:
                # 미세 조정 설정
                self.state['model_params']['fine_tune'] = self.get_yes_no_input(
                    "미세 조정(학습 중 전체 네트워크 업데이트)을 수행하시겠습니까?", 
                    default=False
                )
                
                if not self.state['model_params']['fine_tune']:
                    self.show_message("사전 학습된 가중치를 고정하고 분류기 부분만 학습합니다.")
                else:
                    self.show_message("전체 네트워크를 미세 조정합니다. 학습 시간이 더 길어질 수 있습니다.")
        
        # 학습 파라미터 설정
        print("\n학습 파라미터 설정:")
        
        # 배치 크기
        self.state['training_params']['batch_size'] = self.get_numeric_input(
            "배치 크기", 
            self.state['training_params']['batch_size'],
            min_val=1, 
            max_val=256
        )
        
        # 학습률
        self.state['training_params']['learning_rate'] = self.get_numeric_input(
            "학습률", 
            self.state['training_params']['learning_rate'],
            min_val=0.000001, 
            max_val=0.1
        )
        
        # 에폭 수
        self.state['training_params']['epochs'] = self.get_numeric_input(
            "에폭 수", 
            self.state['training_params']['epochs'],
            min_val=1, 
            max_val=1000
        )
        
        # 조기 종료 인내 횟수
        self.state['training_params']['patience'] = self.get_numeric_input(
            "조기 종료 인내 횟수", 
            self.state['training_params']['patience'],
            min_val=1, 
            max_val=100
        )
        
        # 추가 옵션: 가중치 감소
        self.state['training_params']['weight_decay'] = self.get_numeric_input(
            "가중치 감소 계수", 
            self.state['training_params'].get('weight_decay', 0.0001),
            min_val=0.0, 
            max_val=0.1
        )
        
        # 옵티마이저 선택
        print("\n옵티마이저 선택:")
        optimizers = ["Adam", "SGD", "RMSprop", "AdamW"]
        optimizer_idx = self.show_menu(optimizers, "옵티마이저")
        
        optimizer_map = {
            0: "adam",
            1: "sgd",
            2: "rmsprop",
            3: "adamw"
        }
        
        self.state['training_params']['optimizer'] = optimizer_map[optimizer_idx]
        
        # SGD를 선택한 경우 모멘텀 추가 설정
        if self.state['training_params']['optimizer'] == 'sgd':
            self.state['training_params']['momentum'] = self.get_numeric_input(
                "모멘텀", 
                self.state['training_params'].get('momentum', 0.9),
                min_val=0.0, 
                max_val=1.0
            )
        
        # 학습률 스케줄러 사용 여부
        use_scheduler = self.get_yes_no_input(
            "학습률 스케줄러를 사용하시겠습니까?", 
            default=True
        )
        
        if use_scheduler:
            print("\n학습률 스케줄러 선택:")
            schedulers = ["ReduceLROnPlateau", "StepLR", "CosineAnnealingLR"]
            scheduler_idx = self.show_menu(schedulers, "스케줄러")
            
            scheduler_map = {
                0: "reduce_on_plateau",
                1: "step_lr",
                2: "cosine_annealing"
            }
            
            self.state['training_params']['scheduler'] = scheduler_map[scheduler_idx]
            
            # 스케줄러별 추가 설정
            if self.state['training_params']['scheduler'] == 'reduce_on_plateau':
                self.state['training_params']['scheduler_factor'] = self.get_numeric_input(
                    "감소 계수", 
                    self.state['training_params'].get('scheduler_factor', 0.1),
                    min_val=0.01, 
                    max_val=0.9
                )
                
                self.state['training_params']['scheduler_patience'] = self.get_numeric_input(
                    "스케줄러 인내 횟수", 
                    self.state['training_params'].get('scheduler_patience', 3),
                    min_val=1, 
                    max_val=20
                )
            
            elif self.state['training_params']['scheduler'] == 'step_lr':
                self.state['training_params']['scheduler_step_size'] = self.get_numeric_input(
                    "스텝 크기", 
                    self.state['training_params'].get('scheduler_step_size', 10),
                    min_val=1, 
                    max_val=50
                )
                
                self.state['training_params']['scheduler_gamma'] = self.get_numeric_input(
                    "감마 (축소 비율)", 
                    self.state['training_params'].get('scheduler_gamma', 0.1),
                    min_val=0.01, 
                    max_val=0.9
                )
            
            elif self.state['training_params']['scheduler'] == 'cosine_annealing':
                self.state['training_params']['scheduler_t_max'] = self.get_numeric_input(
                    "T_max (주기)", 
                    self.state['training_params'].get('scheduler_t_max', 10),
                    min_val=1, 
                    max_val=50
                )
        else:
            # 스케줄러 사용하지 않음
            self.state['training_params']['scheduler'] = None
        
        self.show_success("\n모델 파라미터 설정이 완료되었습니다.")
        
        input("\n계속하려면 Enter 키를 누르세요...")
    
def train_model_menu(self) -> None:
    """모델 학습 메뉴"""
    self.print_header("모델 학습")
    
    # 데이터 확인
    if self.state['preprocessed_data'] is None:
        self.show_error("전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
        self.wait_for_user()
        return
    
    # 학습 파라미터 확인
    print("학습에 사용할 설정:")
    print(f"- 장치: {self.state['device']}")
    print(f"- 모델 유형: {self.state['model_params']['model_type']}")
    print(f"- 입력 크기: {self.state['model_params']['input_size']}")
    print(f"- 채널 수: {self.state['model_params']['num_channels']}")
    print(f"- 클래스 수: {self.state['model_params']['num_classes']}")
    print(f"- 배치 크기: {self.state['training_params']['batch_size']}")
    print(f"- 학습률: {self.state['training_params']['learning_rate']}")
    print(f"- 에폭 수: {self.state['training_params']['epochs']}")
    print(f"- 조기 종료 인내 횟수: {self.state['training_params']['patience']}")
    
    # 전이학습 모델인 경우 추가 정보 표시
    if self.state['model_params']['model_type'] in ['vgg16', 'resnet18', 'mobilenetv2']:
        print(f"- 사전 학습 가중치: {'사용' if self.state['model_params'].get('pretrained', True) else '미사용'}")
        print(f"- 미세 조정: {'적용' if self.state['model_params'].get('fine_tune', False) else '미적용'}")
    
    # 학습 시작 확인
    start_training = self.get_yes_no_input("\n위 설정으로 학습을 시작하시겠습니까?")
    if not start_training:
        self.show_message("학습을 취소합니다.")
        self.wait_for_user()
        return
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import TensorDataset, DataLoader
        import numpy as np
        from sklearn.metrics import accuracy_score, classification_report
        import time
        from src.models.image.model_factory import ImageModelFactory
        from src.utils.visualization import plot_training_history
        
        # 데이터 준비
        self.show_message("\n[1/6] 데이터 로더 준비 중...")
        
        # 전처리된 데이터 가져오기
        X_train, y_train = self.state['preprocessed_data']['train_dataset']
        X_val, y_val = self.state['preprocessed_data']['val_dataset']
        
        # 텐서 변환
        # 이미지 데이터 포맷 변환 (N, H, W, C) -> (N, C, H, W)
        X_train_tensor = torch.from_numpy(X_train).permute(0, 3, 1, 2).float()
        X_val_tensor = torch.from_numpy(X_val).permute(0, 3, 1, 2).float()
        
        # 레이블 텐서 변환
        y_train_tensor = torch.from_numpy(y_train).long()
        y_val_tensor = torch.from_numpy(y_val).long()
        
        # 데이터셋 생성
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        # 데이터 로더 생성
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.state['training_params']['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.state['training_params']['batch_size'], 
            shuffle=False
        )
        
        self.show_message(f"데이터 로더 준비 완료:")
        self.show_message(f"- 학습 데이터: {len(train_dataset)}개 이미지")
        self.show_message(f"- 검증 데이터: {len(val_dataset)}개 이미지")
        
        # 모델 초기화
        self.show_message("\n[2/6] 모델 초기화 중...")
        
        # 모델 생성
        model = ImageModelFactory.create_model(
            model_type=self.state['model_params']['model_type'],
            model_params=self.state['model_params'],
            device=self.state['device']
        )
        
        # 모델 정보 출력
        model_info = model.get_model_info()
        self.show_message(f"모델 초기화 완료:")
        self.show_message(f"- 모델 유형: {model_info['model_type']}")
        self.show_message(f"- 파라미터 수: {model_info['parameter_count']:,}")
        
        if 'trainable_params' in model_info:
            trainable_ratio = model_info['trainable_params'] / model_info['parameter_count'] * 100
            self.show_message(f"- 학습 가능 파라미터: {model_info['trainable_params']:,} ({trainable_ratio:.2f}%)")
        
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.CrossEntropyLoss()
        
        # 옵티마이저 선택
        optimizer_type = self.state['training_params'].get('optimizer', 'adam').lower()
        if optimizer_type == 'sgd':
            optimizer = optim.SGD(
                model.parameters(), 
                lr=self.state['training_params']['learning_rate'],
                momentum=self.state['training_params'].get('momentum', 0.9),
                weight_decay=self.state['training_params'].get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'rmsprop':
            optimizer = optim.RMSprop(
                model.parameters(), 
                lr=self.state['training_params']['learning_rate'],
                weight_decay=self.state['training_params'].get('weight_decay', 0.0001)
            )
        elif optimizer_type == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.state['training_params']['learning_rate'],
                weight_decay=self.state['training_params'].get('weight_decay', 0.01)
            )
        else:  # 기본값: adam
            optimizer = optim.Adam(
                model.parameters(), 
                lr=self.state['training_params']['learning_rate'],
                weight_decay=self.state['training_params'].get('weight_decay', 0.0001)
            )
        
        # 학습률 스케줄러 설정
        scheduler = None
        scheduler_type = self.state['training_params'].get('scheduler')
        
        if scheduler_type == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=self.state['training_params'].get('scheduler_factor', 0.1),
                patience=self.state['training_params'].get('scheduler_patience', 3),
                verbose=True
            )
        elif scheduler_type == 'step_lr':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.state['training_params'].get('scheduler_step_size', 10),
                gamma=self.state['training_params'].get('scheduler_gamma', 0.1)
            )
        elif scheduler_type == 'cosine_annealing':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.state['training_params'].get('scheduler_t_max', 10)
            )
        
        # 학습 이력 추적
        history = {
            'train_loss': [], 
            'train_accuracy': [], 
            'valid_loss': [], 
            'valid_accuracy': []
        }
        
        # 조기 종료 설정
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 모델 저장 경로
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        model_save_path = os.path.join(
            self.paths["model_dir"],
            f"{self.state['model_params']['model_type']}_model_{timestamp}.pth"
        )
        
        # 학습 시작
        self.show_message("\n[3/6] 모델 학습 중...")
        
        # 에폭 루프
        for epoch in range(self.state['training_params']['epochs']):
            # 학습 모드
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 학습 데이터 배치 루프
            for inputs, labels in train_loader:
                # 입력과 레이블을 장치로 이동
                inputs = inputs.to(self.state['device'])
                labels = labels.to(self.state['device'])
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순방향 전파
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()
                
                # 통계 누적
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 에폭당 평균 학습 손실 및 정확도
            train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_accuracy)
            
            # 검증 모드
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    # 입력과 레이블을 장치로 이동
                    inputs = inputs.to(self.state['device'])
                    labels = labels.to(self.state['device'])
                    
                    # 순방향 전파
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 통계 누적
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 에폭당 평균 검증 손실 및 정확도
            val_loss = val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            history['valid_loss'].append(val_loss)
            history['valid_accuracy'].append(val_accuracy)
            
            # 학습률 스케줄러 업데이트
            if scheduler is not None:
                if scheduler_type == 'reduce_on_plateau':
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # 에폭 결과 출력
            self.show_message(
                f"에폭 {epoch+1}/{self.state['training_params']['epochs']}, "
                f"학습 손실: {train_loss:.4f}, 학습 정확도: {train_accuracy:.4f}, "
                f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_accuracy:.4f}"
            )
            
            # 조기 종료 확인
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # 최적 모델 저장
                torch.save(model.state_dict(), model_save_path)
                self.show_message(f"새로운 최적 모델 저장: {model_save_path} (검증 손실: {val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.state['training_params']['patience']:
                    self.show_message(f"조기 종료 (에폭 {epoch+1})")
                    break
        
        # 최적 모델 로드
        model.load_state_dict(torch.load(model_save_path))
        
        # 모델 정보 저장
        self.show_message("\n[4/6] 모델 정보 저장 중...")
        model_info = model.get_model_info()
        model_info.update({
            "input_size": self.state['model_params']['input_size'],
            "num_channels": self.state['model_params']['num_channels'],
            "num_classes": self.state['model_params']['num_classes'],
            "class_names": self.state['preprocessed_data']['class_names'],
            "preprocessing": {
                "input_size": self.state['model_params']['input_size'],
                "normalize": self.state['preprocessing_params']['normalize']
            },
            "training": {
                "optimizer": optimizer_type,
                "learning_rate": self.state['training_params']['learning_rate'],
                "epochs": epoch + 1,  # 실제 학습된 에폭 수
                "batch_size": self.state['training_params']['batch_size'],
                "best_val_loss": best_val_loss,
                "best_val_accuracy": max(history['valid_accuracy']),
                "early_stopping": patience_counter >= self.state['training_params']['patience']
            },
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 모델 정보 파일 저장
        model_info_path = os.path.join(self.paths["model_dir"], "model_info.json")
        with open(model_info_path, 'w') as f:
            import json
            json.dump(model_info, f, indent=4)
        
        self.show_message(f"모델 정보 저장 완료: {model_info_path}")
        
        # 학습 이력 시각화
        self.show_message("\n[5/6] 학습 결과 시각화 중...")
        
        plot_dir = self.paths["plot_dir"]
        os.makedirs(plot_dir, exist_ok=True)
        
        history_path = plot_training_history(history, plot_dir)
        self.show_message(f"학습 이력 시각화 저장: {history_path}")
        
        # 클래스별 성능 평가
        self.show_message("\n[6/6] 모델 성능 평가 중...")
        
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(self.state['device'])
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                
                val_predictions.extend(predicted.cpu().numpy())
                val_targets.extend(labels.numpy())
        
        # 클래스 이름 가져오기
        class_names = self.state['preprocessed_data']['class_names']
        
        # 분류 보고서 생성
        report = classification_report(
            val_targets, 
            val_predictions, 
            target_names=class_names,
            output_dict=True
        )
        
        # 평가 결과 출력
        self.show_message("\n클래스별 성능:")
        for class_name in class_names:
            if class_name in report:
                metrics = report[class_name]
                self.show_message(
                    f"- {class_name}: 정밀도={metrics['precision']:.4f}, "
                    f"재현율={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}"
                )
        
        # 상태 업데이트
        self.state['model'] = model
        self.state['training_history'] = history
        self.state['current_model_path'] = model_save_path
        
        # 최종 결과 출력
        self.show_success("\n모델 학습이 완료되었습니다.")
        val_acc = history['valid_accuracy'][-1]
        val_loss = history['valid_loss'][-1]
        self.show_message(f"\n최종 검증 성능:")
        self.show_message(f"- 검증 정확도: {val_acc:.4f}")
        self.show_message(f"- 검증 손실: {val_loss:.4f}")
        self.show_message(f"\n모델이 저장되었습니다: {model_save_path}")
        
    except Exception as e:
        self.show_error(f"\n모델 학습 중 예외가 발생했습니다: {str(e)}")
        import traceback
        logger.exception("모델 학습 중 예외 발생")
        self.show_error(traceback.format_exc())
    
    input("\n계속하려면 Enter 키를 누르세요...")

def evaluate_model_menu(self) -> None:
    """모델 평가 메뉴"""
    self.print_header("모델 평가")
    
    # 모델 확인
    if self.state['model'] is None:
        self.show_error("학습된 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
        self.wait_for_user()
        return
    
    # 데이터 확인
    if self.state['preprocessed_data'] is None:
        self.show_error("전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
        self.wait_for_user()
        return
    
    try:
        import torch
        import torch.nn as nn
        import numpy as np
        from torch.utils.data import TensorDataset, DataLoader
        from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
        import matplotlib.pyplot as plt
        import os
        import seaborn as sns
        from src.models.image.cnn_model import GradCAM
        
        # 테스트 데이터 준비
        self.show_message("\n[1/5] 테스트 데이터 준비 중...")
        
        # 전처리된 테스트 데이터 가져오기
        X_test, y_test = self.state['preprocessed_data']['test_dataset']
        class_names = self.state['preprocessed_data']['class_names']
        
        # 텐서 변환
        # 이미지 데이터 포맷 변환 (N, H, W, C) -> (N, C, H, W)
        X_test_tensor = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()
        y_test_tensor = torch.from_numpy(y_test).long()
        
        # 데이터셋 및 로더 생성
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.state['training_params']['batch_size'], 
            shuffle=False
        )
        
        self.show_message(f"테스트 데이터 준비 완료: {len(test_dataset)}개 이미지")
        
        # 모델 평가 준비
        self.show_message("\n[2/5] 모델 평가 중...")
        model = self.state['model']
        model.eval()
        device = self.state['device']
        
        # 평가 지표
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        class_correct = [0] * len(class_names)
        class_total = [0] * len(class_names)
        
        # 손실 함수
        criterion = nn.CrossEntropyLoss()
        
        # 배치 단위 평가
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 예측
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # 정확도 계산
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # 클래스별 정확도 계산
                for i in range(targets.size(0)):
                    label = targets[i].item()
                    pred = predicted[i].item()
                    class_total[label] += 1
                    if label == pred:
                        class_correct[label] += 1
                
                # 전체 예측 및 타겟 저장 (혼동 행렬 계산용)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # 평균 손실 및 정확도 계산
        test_loss /= len(test_loader)
        accuracy = correct / total
        
        # 평가 결과 출력
        self.show_message("\n평가 결과:")
        self.show_message(f"테스트 손실: {test_loss:.4f}")
        self.show_message(f"테스트 정확도: {accuracy:.4f} ({correct}/{total})")
        
        # 클래스별 정확도 출력
        self.show_message("\n클래스별 정확도:")
        for i in range(len(class_names)):
            if class_total[i] > 0:
                class_acc = class_correct[i] / class_total[i]
                self.show_message(f"- {class_names[i]}: {class_acc:.4f} ({class_correct[i]}/{class_total[i]})")
        
        # 분류 보고서 생성
        report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
        
        # 주요 지표 출력
        self.show_message("\n분류 보고서:")
        for class_name in class_names:
            metrics = report[class_name]
            self.show_message(f"- {class_name}: 정밀도={metrics['precision']:.4f}, 재현율={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
        
        # 혼동 행렬 계산 및 시각화
        self.show_message("\n[3/5] 혼동 행렬 시각화 중...")
        
        # 혼동 행렬 계산
        cm = confusion_matrix(all_targets, all_preds)
        
        # 혼동 행렬 시각화
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # 저장
        plot_dir = self.paths["plot_dir"]
        os.makedirs(plot_dir, exist_ok=True)
        cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.show_message(f"혼동 행렬 시각화 저장: {cm_path}")
        
        # Grad-CAM 시각화
        self.show_message("\n[4/5] Grad-CAM 시각화 중...")
        
        # 타겟 레이어 선택 (모델 유형에 따라 다름)
        target_layer = None
        if hasattr(model, 'features') and hasattr(model, 'classifier'):
            # VGG 또는 MobileNet 구조
            target_layer = model.features[-1]
        elif hasattr(model, 'conv4'):
            # 기본 CNN 구조
            target_layer = model.conv4
        
        if target_layer is not None:
            try:
                # GradCAM 초기화
                grad_cam = GradCAM(model, target_layer)
                
                # 샘플 이미지 선택 (각 클래스별로 1개씩, 최대 10개)
                samples_per_class = 1
                max_samples = min(10, len(class_names) * samples_per_class)
                
                sample_indices = []
                sample_targets = []
                
                # 각 클래스별로 샘플 선택
                for class_idx in range(len(class_names)):
                    class_indices = np.where(y_test == class_idx)[0]
                    if len(class_indices) > 0:
                        selected_indices = class_indices[:samples_per_class]
                        sample_indices.extend(selected_indices)
                        sample_targets.extend([class_idx] * len(selected_indices))
                
                # 최대 샘플 수 제한
                sample_indices = sample_indices[:max_samples]
                sample_targets = sample_targets[:max_samples]
                
                # 샘플별 Grad-CAM 생성
                for i, (idx, target) in enumerate(zip(sample_indices, sample_targets)):
                    # 이미지 및 레이블 가져오기
                    img = X_test[idx]
                    img_tensor = X_test_tensor[idx:idx+1].to(device)
                    true_label = class_names[target]
                    
                    # 예측 수행
                    model.eval()
                    with torch.no_grad():
                        output = model(img_tensor)
                        _, predicted = torch.max(output, 1)
                        pred_label = class_names[predicted.item()]
                        
                    # Grad-CAM 생성
                    cam = grad_cam.generate_grad_cam(img_tensor, predicted.item())
                    cam = cam.cpu().numpy()[0, 0]  # (H, W)
                    
                    # 원본 이미지 가져오기
                    orig_img = np.uint8(img * 255) if img.max() <= 1.0 else np.uint8(img)
                    
                    # 히트맵 생성
                    heatmap = np.uint8(255 * cam)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    
                    # 이미지 형식 맞추기
                    if len(orig_img.shape) == 2:
                        # 흑백 이미지인 경우 3채널로 변환
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2RGB)
                    elif orig_img.shape[2] == 4:
                        # 알파 채널이 있는 경우 RGB로 변환
                        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_RGBA2RGB)
                    
                    # 히트맵 리사이즈
                    heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
                    
                    # 히트맵과 원본 이미지 합성
                    superimposed = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
                    
                    # 시각화
                    plt.figure(figsize=(12, 4))
                    
                    # 원본 이미지
                    plt.subplot(1, 3, 1)
                    plt.imshow(orig_img)
                    plt.title("Original Image")
                    plt.axis('off')
                    
                    # Grad-CAM 히트맵
                    plt.subplot(1, 3, 2)
                    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
                    plt.title("Grad-CAM Heatmap")
                    plt.axis('off')
                    
                    # 합성 이미지
                    plt.subplot(1, 3, 3)
                    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
                    plt.title(f"True: {true_label}\nPred: {pred_label}")
                    plt.axis('off')
                    
                    # 저장
                    gradcam_path = os.path.join(plot_dir, f'gradcam_sample_{i+1}.png')
                    plt.savefig(gradcam_path, dpi=300, bbox_inches='tight')
                    plt.close()
                
                self.show_message(f"Grad-CAM 시각화 저장 완료: {plot_dir}")
                
            except Exception as e:
                self.show_warning(f"Grad-CAM 시각화 중 오류 발생: {str(e)}")
                import traceback
                logger.warning(f"Grad-CAM 시각화 오류: {traceback.format_exc()}")
        else:
            self.show_warning("현재 모델 구조에서는 Grad-CAM을 적용할 수 없습니다.")
        
        # 평가 결과 저장
        self.show_message("\n[5/5] 평가 결과 저장 중...")
        
        # 평가 결과 사전 생성
        evaluation_result = {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'class_accuracy': {class_names[i]: (class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0) 
                              for i in range(len(class_names))},
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # 평가 결과 파일 저장
        eval_path = os.path.join(self.paths["output_dir"], f"evaluation_result_{time.strftime('%Y%m%d_%H%M%S')}.json")
        
        with open(eval_path, 'w') as f:
            import json
            json.dump(evaluation_result, f, indent=4)
        
        # 상태 업데이트
        self.state['evaluation_result'] = evaluation_result
        
        self.show_success(f"\n평가 결과가 저장되었습니다: {eval_path}")
        self.show_message("\n시각화 결과 파일:")
        self.show_message(f"- 혼동 행렬: {cm_path}")
        if target_layer is not None:
            self.show_message(f"- Grad-CAM 샘플: {plot_dir}/gradcam_sample_*.png")
        
    except Exception as e:
        self.show_error(f"\n모델 평가 중 예외가 발생했습니다: {str(e)}")
        import traceback
        logger.exception("모델 평가 중 예외 발생")
        self.show_error(traceback.format_exc())
    
    input("\n계속하려면 Enter 키를 누르세요...")
    
def deploy_model_menu(self) -> None:
    """모델 배포 메뉴"""
    self.print_header("모델 배포")
    
    # 모델 확인
    if self.state['model'] is None or self.state['current_model_path'] is None:
        self.show_error("배포할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
        self.wait_for_user()
        return
    
    print("모델 배포는 학습된 모델을 배포 디렉토리에 복사하고")
    print("추론을 위한 필요한 파일들을 준비하는 단계입니다.\n")
    
    # 배포 디렉토리 설정
    deploy_dir = self.get_input("배포 디렉토리 경로", self.paths["deploy_dir"])
    
    try:
        # 배포 디렉토리 생성
        os.makedirs(deploy_dir, exist_ok=True)
        
        # 타임스탬프로 하위 디렉토리 생성
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        deploy_subdir = os.path.join(deploy_dir, f"deployment_{timestamp}")
        os.makedirs(deploy_subdir, exist_ok=True)
        
        self.show_message("\n[1/5] 모델 파일 복사 중...")
        model_filename = os.path.basename(self.state['current_model_path'])
        deploy_model_path = os.path.join(deploy_subdir, model_filename)
        shutil.copy2(self.state['current_model_path'], deploy_model_path)
        self.show_message(f"모델 파일 복사 완료: {deploy_model_path}")
        
        # 모델 정보 파일 복사
        self.show_message("\n[2/5] 모델 정보 파일 복사 중...")
        model_info_src = os.path.join(self.paths["model_dir"], 'model_info.json')
        model_info_dst = os.path.join(deploy_subdir, 'model_info.json')
        if os.path.exists(model_info_src):
            shutil.copy2(model_info_src, model_info_dst)
            self.show_message(f"모델 정보 파일 복사 완료: {model_info_dst}")
        else:
            # 모델 정보 파일이 없으면 새로 생성
            model_info = self.state['model'].get_model_info()
            model_info.update({
                "input_size": self.state['model_params']['input_size'],
                "num_channels": self.state['model_params']['num_channels'],
                "num_classes": self.state['model_params']['num_classes'],
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            })
            with open(model_info_dst, 'w') as f:
                json.dump(model_info, f, indent=4)
            self.show_message(f"모델 정보 파일 생성 완료: {model_info_dst}")
        
        # 클래스 매핑 정보 저장
        self.show_message("\n[3/5] 클래스 매핑 정보 저장 중...")
        if self.state['preprocessed_data'] is not None and 'class_names' in self.state['preprocessed_data']:
            class_names = self.state['preprocessed_data']['class_names']
            class_mapping = {i: name for i, name in enumerate(class_names)}
            
            class_mapping_path = os.path.join(deploy_subdir, 'class_mapping.json')
            with open(class_mapping_path, 'w') as f:
                json.dump(class_mapping, f, indent=4)
            self.show_message(f"클래스 매핑 정보 저장 완료: {class_mapping_path}")
        else:
            self.show_warning("클래스 매핑 정보를 찾을 수 없습니다.")
        
        # 평가 결과 복사 (있는 경우)
        self.show_message("\n[4/5] 평가 결과 복사 중...")
        if self.state['evaluation_result'] is not None:
            eval_result_path = os.path.join(deploy_subdir, 'evaluation_result.json')
            
            # NumPy 배열을 일반 리스트로 변환하여 JSON 직렬화 가능하게 만듦
            eval_result = {}
            for k, v in self.state['evaluation_result'].items():
                if hasattr(v, 'tolist'):
                    eval_result[k] = v.tolist()
                elif isinstance(v, dict):
                    eval_result[k] = {}
                    for kk, vv in v.items():
                        if hasattr(vv, 'tolist'):
                            eval_result[k][kk] = vv.tolist()
                        else:
                            eval_result[k][kk] = vv
                else:
                    eval_result[k] = v
            
            with open(eval_result_path, 'w') as f:
                json.dump(eval_result, f, indent=4)
            self.show_message(f"평가 결과 저장 완료: {eval_result_path}")
        else:
            self.show_message("평가 결과가 없어 복사를 건너뜁니다.")
        
        # 추론 스크립트 생성
        self.show_message("\n[5/5] 추론 스크립트 생성 중...")
        
        # src/models/image/inference.py 파일을 참고하여 추론 스크립트 생성
        inference_script = """#!/usr/bin/env python"""
    
    except Exception as e:
        self.show_error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        logger.exception("예상치 못한 오류 발생")

    input("\n계속하려면 Enter 키를 누르세요...")

def system_config_menu(self) -> None:
    """시스템 설정 메뉴"""
    self.print_header("시스템 설정")
    
    print("시스템 설정을 변경합니다.\n")
    
    # 여기에 시스템 설정 코드를 구현합니다.
    # TODO: 구현
    
    input("\n계속하려면 Enter 키를 누르세요...")

def print_status(self) -> None:
    """현재 상태 출력"""
    print("\n현재 상태:")
    print("-" * 40)
    
    # 데이터 상태
    if self.state['preprocessed_data'] is not None:
        datasets = self.state['preprocessed_data']
        print(f"✅ 전처리된 데이터: 학습 {len(datasets['train_dataset'])} 이미지, "
                f"검증 {len(datasets['val_dataset'])} 이미지, "
                f"테스트 {len(datasets['test_dataset'])} 이미지")
    else:
        print("❌ 전처리된 데이터: 없음")
    
    # 모델 파라미터
    print(f"모델: {self.state['model_params']['model_type']} "
            f"(설정된 파라미터: 입력 크기 {self.state['model_params']['input_size']}, "
            f"클래스 수 {self.state['model_params']['num_classes']})")
    
    # 학습 상태
    if self.state['training_history'] is not None:
        val_acc = self.state['training_history']['val_accuracy'][-1]
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
    
    print("-" * 40)

def run(self) -> None:
    """CLI 실행"""
    try:
        self.main_menu()
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        self.show_error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
        logger.exception("예상치 못한 오류 발생")

def main():
    """메인 함수: CLI 실행"""
    try:
        # CLI 인스턴스 생성 및 실행
        cli = ImageCLI()
        cli.run()
        
    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        print("로그 파일을 확인하세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
