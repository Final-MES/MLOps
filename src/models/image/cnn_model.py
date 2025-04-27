"""
이미지 분류를 위한 CNN 모델 모듈

이 모듈은 이미지 분류를 위한 다양한 CNN 모델 구현을 제공합니다:
- 기본 CNN 모델
- 전이 학습 기반 모델 (VGG16, ResNet, MobileNet 등)
- 모델 평가 및 시각화 유틸리티
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class BasicCNN(nn.Module):
    """
    기본 CNN 모델
    
    Args:
        input_size (Tuple[int, int]): 입력 이미지 크기 (height, width)
        num_channels (int): 입력 채널 수 (1: 흑백, 3: 컬러)
        num_classes (int): 분류할 클래스 수
        hidden_size (int): 은닉층 크기
        dropout_rate (float): 드롭아웃 비율
    """
    def __init__(self, 
                input_size: Tuple[int, int] = (224, 224), 
                num_channels: int = 3, 
                num_classes: int = 10,
                hidden_size: int = 128,
                dropout_rate: float = 0.5):
        super(BasicCNN, self).__init__()
        
        self.input_size = input_size
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        
        # 첫 번째 컨볼루션 블록
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 두 번째 컨볼루션 블록
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 세 번째 컨볼루션 블록
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 네 번째 컨볼루션 블록
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 전결합층 입력 크기 계산
        fc_input_size = self._calculate_fc_input_size()
        
        # 전결합층
        self.fc1 = nn.Linear(fc_input_size, hidden_size)
        self.bn5 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _calculate_fc_input_size(self) -> int:
        """전결합층 입력 크기 계산"""
        h, w = self.input_size
        # 4번의 2x2 풀링 적용 후 크기
        h = h // (2 ** 4)
        w = w // (2 ** 4)
        return 256 * h * w
    
    def _initialize_weights(self) -> None:
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x: 입력 텐서, 형태 (batch_size, num_channels, height, width)
            
        Returns:
            torch.Tensor: 클래스별 로짓, 형태 (batch_size, num_classes)
        """
        # 첫 번째 컨볼루션 블록
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # 두 번째 컨볼루션 블록
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # 세 번째 컨볼루션 블록
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # 네 번째 컨볼루션 블록
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # 평탄화
        x = x.view(x.size(0), -1)
        
        # 전결합층
        x = F.relu(self.bn5(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 구성 정보
        """
        return {
            "model_type": "BasicCNN",
            "input_size": self.input_size,
            "num_channels": self.num_channels,
            "num_classes": self.num_classes,
            "hidden_size": self.hidden_size,
            "dropout_rate": self.dropout_rate,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }


class TransferLearningModel(nn.Module):
    """
    전이 학습 기반 이미지 분류 모델
    
    Args:
        model_type (str): 기본 모델 유형 ('vgg16', 'resnet18', 'mobilenetv2' 등)
        num_classes (int): 분류할 클래스 수
        pretrained (bool): 사전 학습된 가중치 사용 여부
        fine_tune (bool): 특성 추출기 미세 조정 여부
        dropout_rate (float): 드롭아웃 비율
    """
    def __init__(self, 
                model_type: str = 'resnet18', 
                num_classes: int = 10,
                pretrained: bool = True, 
                fine_tune: bool = False, 
                dropout_rate: float = 0.5):
        super(TransferLearningModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.fine_tune = fine_tune
        self.dropout_rate = dropout_rate
        
        # 모델 유형별 특성 추출기 생성
        if self.model_type == 'vgg16':
            self.features = models.vgg16(pretrained=pretrained).features
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(dropout_rate),
                nn.Linear(4096, num_classes)
            )
            
        elif self.model_type == 'resnet18':
            model = models.resnet18(pretrained=pretrained)
            self.features = nn.Sequential(*list(model.children())[:-1])
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(512, num_classes)
            )
            
        elif self.model_type == 'mobilenetv2':
            model = models.mobilenet_v2(pretrained=pretrained)
            self.features = model.features
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(1280, num_classes)
            )
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 특성 추출기 고정 (미세 조정을 하지 않는 경우)
        if not fine_tune:
            for param in self.features.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x: 입력 텐서, 형태 (batch_size, 3, height, width)
            
        Returns:
            torch.Tensor: 클래스별 로짓, 형태 (batch_size, num_classes)
        """
        # 특성 추출
        x = self.features(x)
        
        # 모델별 추가 처리
        if self.model_type == 'vgg16':
            x = self.pool(x)
        elif self.model_type == 'mobilenetv2':
            x = self.pool(x)
        
        # 평탄화
        x = torch.flatten(x, 1)
        
        # 분류
        x = self.classifier(x)
        
        return x
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 구성 정보
        """
        return {
            "model_type": f"TransferLearning_{self.model_type}",
            "base_model": self.model_type,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "fine_tune": self.fine_tune,
            "dropout_rate": self.dropout_rate,
            "parameter_count": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class GradCAM:
    """
    Grad-CAM 시각화 클래스
    
    CNN의 판단 근거를 시각화하는 Grad-CAM 기법 구현
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        초기화
        
        Args:
            model: CNN 모델
            target_layer: 시각화 대상 레이어
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 훅 등록
        self.register_hooks()
        
    def register_hooks(self) -> None:
        """대상 레이어에 훅 등록"""
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        # 훅 등록
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
        
    def generate_grad_cam(self, input_tensor: torch.Tensor, target_class: Optional[int] = None) -> torch.Tensor:
        """
        Grad-CAM 생성
        
        Args:
            input_tensor: 입력 이미지 텐서
            target_class: 타겟 클래스 인덱스 (None이면 최대 확률 클래스)
            
        Returns:
            torch.Tensor: Grad-CAM 히트맵
        """
        # 모델 예측
        model_output = self.model(input_tensor)
        
        # 타겟 클래스 결정
        if target_class is None:
            target_class = torch.argmax(model_output, dim=1).item()
        
        # 역전파
        self.model.zero_grad()
        one_hot = torch.zeros_like(model_output)
        one_hot[0, target_class] = 1
        model_output.backward(gradient=one_hot)
        
        # Grad-CAM 계산
        gradients = self.gradients.detach()
        activations = self.activations.detach()
        
        # 채널별 가중치 계산
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        
        # 가중치와 활성화 맵의 가중합
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        
        # ReLU 적용
        cam = F.relu(cam)
        
        # 정규화
        cam = F.interpolate(
            cam, 
            size=input_tensor.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam