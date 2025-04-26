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
            print("❌ 오류: 전처리된 데이터가 없습니다. 먼저 데이터 전처리를 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
        
        # 여기에 모델 학습 코드를 구현합니다.
        # TODO: 구현
        
        input("\n계속하려면 Enter 키를 누르세요...")
    
    def evaluate_model_menu(self) -> None:
        """모델 평가 메뉴"""
        self.print_header("모델 평가")
        
        # 모델 확인
        if self.state['model'] is None:
            print("❌ 오류: 학습된 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
        
        # 여기에 모델 평가 코드를 구현합니다.
        # TODO: 구현
        
        input("\n계속하려면 Enter 키를 누르세요...")
    
    def deploy_model_menu(self) -> None:
        """모델 배포 메뉴"""
        self.print_header("모델 배포")
        
        # 모델 확인
        if self.state['model'] is None or self.state['current_model_path'] is None:
            print("❌ 오류: 배포할 모델이 없습니다. 먼저 모델 학습을 수행하세요.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
        
        # 여기에 모델 배포 코드를 구현합니다.
        # TODO: 구현
        
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
