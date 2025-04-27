"""
이미지 모델 팩토리 모듈

이 모듈은 이미지 분류를 위한 다양한 모델을 생성하는 팩토리 패턴을 구현합니다.
모델 유형과 파라미터에 따라 적절한 모델 인스턴스를 반환합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import logging

from src.models.image.cnn_model import BasicCNN, TransferLearningModel

# 로깅 설정
logger = logging.getLogger(__name__)

class ImageModelFactory:
    """
    이미지 모델 팩토리 클래스
    
    다양한 이미지 분류 모델을 생성하는 팩토리 메서드를 제공합니다.
    """
    
    @staticmethod
    def create_model(model_type: str, model_params: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
        """
        모델 생성
        
        Args:
            model_type: 모델 유형 ('cnn', 'vgg16', 'resnet18', 'mobilenetv2' 등)
            model_params: 모델 파라미터 딕셔너리
            device: 모델을 배치할 장치 (CPU/GPU)
            
        Returns:
            nn.Module: 생성된 모델 인스턴스
        """
        model_type = model_type.lower()
        
        # 기본 CNN 모델
        if model_type == 'cnn':
            model = BasicCNN(
                input_size=model_params.get('input_size', (224, 224)),
                num_channels=model_params.get('num_channels', 3),
                num_classes=model_params.get('num_classes', 10),
                hidden_size=model_params.get('hidden_size', 128),
                dropout_rate=model_params.get('dropout_rate', 0.5)
            )
            logger.info(f"BasicCNN 모델 생성 완료: {model.get_model_info()}")
            
        # 전이 학습 모델
        elif model_type in ['vgg16', 'resnet18', 'mobilenetv2']:
            model = TransferLearningModel(
                model_type=model_type,
                num_classes=model_params.get('num_classes', 10),
                pretrained=model_params.get('pretrained', True),
                fine_tune=model_params.get('fine_tune', False),
                dropout_rate=model_params.get('dropout_rate', 0.5)
            )
            logger.info(f"TransferLearningModel({model_type}) 생성 완료: {model.get_model_info()}")
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 지정된, 장치로 모델 이동
        if device is not None:
            model = model.to(device)
            logger.info(f"모델을 장치 {device}로 이동했습니다.")
        
        return model
    
    @staticmethod
    def load_model_from_path(model_path: str, model_info: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 가중치 파일 경로
            model_info: 모델 정보 딕셔너리
            device: 모델을 배치할 장치 (CPU/GPU)
            
        Returns:
            nn.Module: 로드된 모델 인스턴스
        """
        # 모델 유형 결정
        model_type = model_info.get('model_type', '').lower()
        
        if 'basic' in model_type or model_type == 'cnn':
            # BasicCNN 모델
            model = BasicCNN(
                input_size=model_info.get('input_size', (224, 224)),
                num_channels=model_info.get('num_channels', 3),
                num_classes=model_info.get('num_classes', 10),
                hidden_size=model_info.get('hidden_size', 128),
                dropout_rate=model_info.get('dropout_rate', 0.5)
            )
            logger.info(f"BasicCNN 모델 로드: {model_info}")
            
        elif 'transfer' in model_type or model_type in ['vgg16', 'resnet18', 'mobilenetv2']:
            # 전이 학습 모델
            base_model = model_info.get('base_model', model_type.split('_')[-1] if '_' in model_type else 'resnet18')
            model = TransferLearningModel(
                model_type=base_model,
                num_classes=model_info.get('num_classes', 10),
                pretrained=False,  # 저장된 가중치를 로드할 것이므로 사전 학습 X
                fine_tune=model_info.get('fine_tune', False),
                dropout_rate=model_info.get('dropout_rate', 0.5)
            )
            logger.info(f"TransferLearningModel({base_model}) 로드: {model_info}")
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"모델 가중치를 '{model_path}'에서 로드했습니다.")
        
        # 지정된 장치로 모델 이동
        if device is not None:
            model = model.to(device)
            logger.info(f"모델을 장치 {device}로 이동했습니다.")
        
        # 평가 모드로 설정
        model.eval()
        
        return model