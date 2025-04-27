"""
이미지 모델 추론 모듈

이 모듈은 훈련된 이미지 분류 모델을 사용한 추론 기능을 제공합니다:
- 이미지 입력 전처리
- 모델 예측 수행
- 결과 해석 및 시각화
"""

import torch
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

from src.models.image.model_factory import ImageModelFactory
from src.models.image.cnn_model import GradCAM
from src.data.image.preprocessor import ImagePreprocessor

# 로깅 설정
logger = logging.getLogger(__name__)

class ImageInferenceEngine:
    """
    이미지 모델 추론 엔진
    
    훈련된 이미지 분류 모델을 사용한 추론을 수행합니다.
    """
    
    def __init__(self, 
                model_path: str, 
                model_info_path: str,
                device: Optional[torch.device] = None):
        """
        추론 엔진 초기화
        
        Args:
            model_path: 모델 가중치 파일 경로
            model_info_path: 모델 정보 파일 경로
            device: 추론에 사용할 장치 (CPU/GPU)
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 정보 로드
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)
        
        # 모델 로드
        self.model = ImageModelFactory.load_model_from_path(
            model_path=model_path,
            model_info=self.model_info,
            device=self.device
        )
        
        # 전처리기 초기화
        self.preprocessor = ImagePreprocessor(
            target_size=self.model_info.get('input_size', (224, 224)),
            normalize=True,
            color_mode='rgb' if self.model_info.get('num_channels', 3) == 3 else 'grayscale'
        )
        
        # 클래스 레이블 매핑
        self.class_mapping = self.model_info.get('class_mapping', {})
        self.inverse_class_mapping = {int(idx): label for idx, label in self.class_mapping.items()}
        
        # 평가 모드로 설정
        self.model.eval()
        
        # GradCAM을 위한 대상 레이어 설정
        self.gradcam = None
        
        logger.info(f"이미지 추론 엔진 초기화 완료: {self.model_info.get('model_type')} 모델")
    
    def setup_gradcam(self) -> bool:
        """
        GradCAM 설정
        
        Returns:
            bool: 설정 성공 여부
        """
        try:
            # 모델 유형에 따라 대상 레이어 선택
            if self.model_info.get('model_type') == 'BasicCNN':
                # BasicCNN의 경우 마지막 컨볼루션 레이어 사용
                target_layer = self.model.conv4
            elif 'TransferLearning' in self.model_info.get('model_type', ''):
                # 전이 학습 모델의 경우 모델에 따라 다른 레이어 선택
                base_model = self.model_info.get('base_model', '').lower()
                if base_model == 'vgg16':
                    target_layer = self.model.features[-1]
                elif base_model == 'resnet18':
                    target_layer = list(self.model.features.children())[-2]
                elif base_model == 'mobilenetv2':
                    target_layer = self.model.features[-1]
                else:
                    logger.warning(f"알 수 없는 기본 모델 유형: {base_model}")
                    return False
            else:
                logger.warning(f"GradCAM을 지원하지 않는 모델 유형: {self.model_info.get('model_type')}")
                return False
            
            # GradCAM 초기화
            self.gradcam = GradCAM(self.model, target_layer)
            logger.info(f"GradCAM 설정 완료")
            return True
            
        except Exception as e:
            logger.error(f"GradCAM 설정 중 오류 발생: {str(e)}")
            return False
    
    def preprocess_image(self, image_path: Union[str, np.ndarray]) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image_path: 이미지 파일 경로 또는 NumPy 배열
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # 이미지 로드
        if isinstance(image_path, str):
            image = self.preprocessor.load_image(image_path)
            if image is None:
                raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        else:
            # NumPy 배열인 경우 전처리
            image = self.preprocessor.preprocess_image(image_path)
        
        # 텐서 변환 및 차원 조정
        image_tensor = torch.tensor(image, dtype=torch.float32)
        if image_tensor.dim() == 3:  # (H, W, C)
            # 채널 차원 조정: (H, W, C) -> (C, H, W)
            image_tensor = image_tensor.permute(2, 0, 1)
            # 배치 차원 추가: (C, H, W) -> (1, C, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 2:  # 흑백 이미지 (H, W)
            # 채널 차원 추가: (H, W) -> (1, H, W)
            image_tensor = image_tensor.unsqueeze(0)
            # 배치 차원 추가: (1, H, W) -> (1, 1, H, W)
            image_tensor = image_tensor.unsqueeze(0)
        
        # 장치로 이동
        image_tensor = image_tensor.to(self.device)
        
        return image_tensor
    
    def predict(self, image_path: Union[str, np.ndarray], return_probs: bool = False) -> Union[int, Dict[str, Any]]:
        """
        이미지 분류 예측 수행
        
        Args:
            image_path: 이미지 파일 경로 또는 NumPy 배열
            return_probs: 각 클래스별 확률 반환 여부
            
        Returns:
            Union[int, Dict[str, Any]]: 예측 클래스 또는 예측 정보
        """
        # 이미지 전처리
        image_tensor = self.preprocess_image(image_path)
        
        # 모델 추론
        with torch.no_grad():
            model_output = self.model(image_tensor)
        
        # 예측 클래스 및 확률 계산
        probabilities = torch.softmax(model_output, dim=1)[0].cpu().numpy()
        predicted_class = int(torch.argmax(model_output, dim=1).item())
        
        # 클래스 레이블 가져오기
        if predicted_class in self.inverse_class_mapping:
            predicted_label = self.inverse_class_mapping[predicted_class]
        else:
            predicted_label = f"Class_{predicted_class}"
        
        # 결과 반환
        if return_probs:
            class_probs = {
                self.inverse_class_mapping.get(i, f"Class_{i}"): float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'probability': float(probabilities[predicted_class]),
                'class_probabilities': class_probs
            }
        else:
            return predicted_class
    
    def visualize_gradcam(self, image_path: Union[str, np.ndarray], 
                        target_class: Optional[int] = None,
                        output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        GradCAM 시각화
        
        Args:
            image_path: 이미지 파일 경로 또는 NumPy 배열
            target_class: 시각화 대상 클래스 (None: 예측 클래스)
            output_path: 시각화 이미지 저장 경로
            
        Returns:
            Optional[plt.Figure]: 시각화 그림 객체 (None: GradCAM 실패)
        """
        # GradCAM 설정이 안 되어 있으면 설정
        if self.gradcam is None:
            success = self.setup_gradcam()
            if not success:
                logger.error("GradCAM 설정에 실패했습니다.")
                return None
        
        # 원본 이미지 로드
        if isinstance(image_path, str):
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = image_path.copy()
            if original_image.ndim == 2:  # 흑백 이미지
                original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        
        # 이미지 크기 조정
        original_image = cv2.resize(
            original_image, 
            (self.model_info.get('input_size', (224, 224))[1], self.model_info.get('input_size', (224, 224))[0])
        )
        
        # 이미지 전처리 및 예측
        image_tensor = self.preprocess_image(image_path)
        
        # 타겟 클래스 결정
        if target_class is None:
            prediction = self.predict(image_path, return_probs=True)
            target_class = prediction['predicted_class']
            predicted_label = prediction['predicted_label']
            prob = prediction['probability']
        else:
            predicted_label = self.inverse_class_mapping.get(target_class, f"Class_{target_class}")
            with torch.no_grad():
                output = self.model(image_tensor)
                probabilities = torch.softmax(output, dim=1)[0].cpu().numpy()
                prob = float(probabilities[target_class])
        
        # GradCAM 생성
        cam = self.gradcam.generate_grad_cam(image_tensor, target_class)
        cam = cam.cpu().numpy()[0, 0]  # (H, W)
        
        # 히트맵 생성
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # 원본 이미지와 히트맵 합성
        alpha = 0.5
        superimposed_img = heatmap * alpha + original_image
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
        
        # 시각화
        plt.figure(figsize=(12, 5))
        
        # 원본 이미지
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image")
        plt.axis('off')
        
        # GradCAM 히트맵
        plt.subplot(1, 3, 2)
        plt.imshow(heatmap)
        plt.title("GradCAM Heatmap")
        plt.axis('off')
        
        # 합성 이미지
        plt.subplot(1, 3, 3)
        plt.imshow(superimposed_img)
        plt.title(f"Prediction: {predicted_label} ({prob:.2f})")
        plt.axis('off')
        
        plt.tight_layout()
        
        # 저장
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"GradCAM 시각화 저장 완료: {output_path}")
        
        return plt.gcf()
    
    def batch_predict(self, image_paths: List[Union[str, np.ndarray]]) -> List[Dict[str, Any]]:
        """
        배치 예측 수행
        
        Args:
            image_paths: 이미지 파일 경로 또는 NumPy 배열 목록
            
        Returns:
            List[Dict[str, Any]]: 각 이미지별 예측 결과
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, return_probs=True)
            results.append(result)
        
        return results


