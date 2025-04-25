"""
이미지 데이터 증강 모듈

이 모듈은 이미지 데이터 증강을 위한 기능을 제공합니다:
- 기본 변환 (회전, 플립, 크롭)
- 색상 변환 (밝기, 대비, 채도)
- 노이즈 추가
- 기하학적 변환 (투영, 왜곡)
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Any, Optional, Union, Callable
import logging
import random

# 로깅 설정
logger = logging.getLogger(__name__)

class ImageAugmentor:
    """이미지 데이터 증강을 위한 클래스"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        증강기 초기화
        
        Args:
            seed: 랜덤 시드 (재현성 확보)
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def set_seed(self, seed: int) -> None:
        """
        랜덤 시드 설정
        
        Args:
            seed: 랜덤 시드
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def random_flip(self, image: np.ndarray, horizontal_prob: float = 0.5, 
                   vertical_prob: float = 0.1) -> np.ndarray:
        """
        랜덤 플립 변환
        
        Args:
            image: 입력 이미지
            horizontal_prob: 수평 플립 확률
            vertical_prob: 수직 플립 확률
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        img = image.copy()
        
        # 수평 플립
        if self.rng.random() < horizontal_prob:
            img = cv2.flip(img, 1)
        
        # 수직 플립
        if self.rng.random() < vertical_prob:
            img = cv2.flip(img, 0)
        
        return img
    
    def random_rotation(self, image: np.ndarray, max_angle: float = 30) -> np.ndarray:
        """
        랜덤 회전 변환
        
        Args:
            image: 입력 이미지
            max_angle: 최대 회전 각도
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        height, width = image.shape[:2]
        angle = self.rng.uniform(-max_angle, max_angle)
        
        # 회전 중심점과 변환 행렬
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # 회전된 이미지 크기 계산
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        # 중심점 이동
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        
        # 회전 적용
        rotated = cv2.warpAffine(image, rotation_matrix, (bound_w, bound_h))
        
        # 원본 크기로 리사이즈
        rotated = cv2.resize(rotated, (width, height))
        
        return rotated
    
    def random_crop(self, image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """
        랜덤 크롭 변환
        
        Args:
            image: 입력 이미지
            crop_ratio: 크롭 비율 (원본 크기 대비)
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        height, width = image.shape[:2]
        
        # 크롭 크기 계산
        crop_height = int(height * crop_ratio)
        crop_width = int(width * crop_ratio)
        
        # 랜덤 시작점 선택
        start_x = self.rng.randint(0, width - crop_width + 1)
        start_y = self.rng.randint(0, height - crop_height + 1)
        
        # 크롭 적용
        cropped = image[start_y:start_y+crop_height, start_x:start_x+crop_width]
        
        # 원본 크기로 리사이즈
        resized = cv2.resize(cropped, (width, height))
        
        return resized
    
    def random_brightness(self, image: np.ndarray, max_delta: float = 0.2) -> np.ndarray:
        """
        랜덤 밝기 변환
        
        Args:
            image: 입력 이미지
            max_delta: 최대 밝기 변화량
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        img = image.copy()
        
        # 0~1 범위의 이미지로 변환
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # 밝기 조정
        delta = self.rng.uniform(-max_delta, max_delta)
        img = img + delta
        
        # 값 범위 제한
        img = np.clip(img, 0.0, 1.0)
        
        # 원본 타입으로 변환
        if image.max() > 1.0:
            img = (img * 255.0).astype(image.dtype)
        
        return img
    
    def random_contrast(self, image: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
        """
        랜덤 대비 변환
        
        Args:
            image: 입력 이미지
            lower: 최소 대비 배율
            upper: 최대 대비 배율
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        img = image.copy()
        
        # 0~1 범위의 이미지로 변환
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # 대비 조정
        factor = self.rng.uniform(lower, upper)
        mean = img.mean()
        img = (img - mean) * factor + mean
        
        # 값 범위 제한
        img = np.clip(img, 0.0, 1.0)
        
        # 원본 타입으로 변환
        if image.max() > 1.0:
            img = (img * 255.0).astype(image.dtype)
        
        return img
    
    def random_hue(self, image: np.ndarray, max_delta: float = 0.05) -> np.ndarray:
        """
        랜덤 색조 변환
        
        Args:
            image: 입력 이미지 (RGB)
            max_delta: 최대 색조 변화량
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        # RGB 이미지만 처리
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        
        img = image.copy()
        
        # 0~1 범위의 이미지로 변환
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # RGB -> HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 색조 조정
        delta = self.rng.uniform(-max_delta, max_delta)
        hsv[:, :, 0] = (hsv[:, :, 0] + delta * 180) % 180
        
        # HSV -> RGB 변환
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 원본 타입으로 변환
        if image.max() > 1.0:
            img = (img * 255.0).astype(image.dtype)
        
        return img
    
    def random_saturation(self, image: np.ndarray, lower: float = 0.8, upper: float = 1.2) -> np.ndarray:
        """
        랜덤 채도 변환
        
        Args:
            image: 입력 이미지 (RGB)
            lower: 최소 채도 배율
            upper: 최대 채도 배율
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        # RGB 이미지만 처리
        if image.ndim != 3 or image.shape[2] != 3:
            return image
        
        img = image.copy()
        
        # 0~1 범위의 이미지로 변환
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # RGB -> HSV 변환
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        
        # 채도 조정
        factor = self.rng.uniform(lower, upper)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0.0, 1.0)
        
        # HSV -> RGB 변환
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        
        # 원본 타입으로 변환
        if image.max() > 1.0:
            img = (img * 255.0).astype(image.dtype)
        
        return img
    
    def add_gaussian_noise(self, image: np.ndarray, mean: float = 0, 
                         std: float = 0.05) -> np.ndarray:
        """
        가우시안 노이즈 추가
        
        Args:
            image: 입력 이미지
            mean: 노이즈 평균
            std: 노이즈 표준편차
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        img = image.copy()
        
        # 0~1 범위의 이미지로 변환
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        # 노이즈 생성 및 추가
        noise = self.rng.normal(mean, std, img.shape)
        img = img + noise
        
        # 값 범위 제한
        img = np.clip(img, 0.0, 1.0)
        
        # 원본 타입으로 변환
        if image.max() > 1.0:
            img = (img * 255.0).astype(image.dtype)
        
        return img
    
    def random_perspective_transform(self, image: np.ndarray, 
                                   max_distortion: float = 0.1) -> np.ndarray:
        """
        랜덤 투영 변환
        
        Args:
            image: 입력 이미지
            max_distortion: 최대 왜곡 정도
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        height, width = image.shape[:2]
        
        # 원본 이미지의 꼭지점
        src_points = np.array([
            [0, 0],           # 좌상단
            [width - 1, 0],   # 우상단
            [width - 1, height - 1],  # 우하단
            [0, height - 1]   # 좌하단
        ], dtype=np.float32)
        
        # 목표 이미지의 꼭지점 (랜덤 왜곡 적용)
        dst_points = np.array([
            [self.rng.uniform(0, width * max_distortion),
             self.rng.uniform(0, height * max_distortion)],
            [self.rng.uniform(width * (1 - max_distortion), width - 1),
             self.rng.uniform(0, height * max_distortion)],
            [self.rng.uniform(width * (1 - max_distortion), width - 1),
             self.rng.uniform(height * (1 - max_distortion), height - 1)],
            [self.rng.uniform(0, width * max_distortion),
             self.rng.uniform(height * (1 - max_distortion), height - 1)]
        ], dtype=np.float32)
        
        # 투영 변환 행렬 계산
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 투영 변환 적용
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return transformed
    
    def random_blur(self, image: np.ndarray, k_min: int = 1, k_max: int = 5) -> np.ndarray:
        """
        랜덤 블러 효과
        
        Args:
            image: 입력 이미지
            k_min: 최소 커널 크기 (홀수)
            k_max: 최대 커널 크기 (홀수)
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        # 홀수 커널 크기 선택
        k_size = self.rng.randint(k_min // 2, k_max // 2 + 1) * 2 + 1
        
        # 가우시안 블러 적용
        blurred = cv2.GaussianBlur(image, (k_size, k_size), 0)
        
        return blurred
    
    def augment(self, image: np.ndarray, augmentations: List[str] = None) -> np.ndarray:
        """
        이미지에 여러 증강 기법 적용
        
        Args:
            image: 입력 이미지
            augmentations: 적용할 증강 기법 목록 (None이면 모든 기법 적용)
            
        Returns:
            np.ndarray: 증강된 이미지
        """
        img = image.copy()
        
        # 기본 증강 기법 목록
        all_augmentations = [
            'flip', 'rotation', 'crop', 'brightness', 'contrast',
            'hue', 'saturation', 'noise', 'perspective', 'blur'
        ]
        
        # 적용할 증강 기법 선택
        if augmentations is None:
            selected_augmentations = all_augmentations
        else:
            selected_augmentations = [aug for aug in augmentations if aug in all_augmentations]
        
        # 증강 기법 적용
        for aug in selected_augmentations:
            if aug == 'flip':
                img = self.random_flip(img)
            elif aug == 'rotation':
                img = self.random_rotation(img)
            elif aug == 'crop':
                img = self.random_crop(img)
            elif aug == 'brightness':
                img = self.random_brightness(img)
            elif aug == 'contrast':
                img = self.random_contrast(img)
            elif aug == 'hue':
                img = self.random_hue(img)
            elif aug == 'saturation':
                img = self.random_saturation(img)
            elif aug == 'noise':
                img = self.add_gaussian_noise(img)
            elif aug == 'perspective':
                img = self.random_perspective_transform(img)
            elif aug == 'blur':
                img = self.random_blur(img)
        
        return img
    
    def augment_batch(self, images: List[np.ndarray], augmentations: List[str] = None) -> List[np.ndarray]:
        """
        이미지 배치에 증강 적용
        
        Args:
            images: 입력 이미지 목록
            augmentations: 적용할 증강 기법 목록
            
        Returns:
            List[np.ndarray]: 증강된 이미지 목록
        """
        return [self.augment(img, augmentations) for img in images]
    
    def generate_augmented_batch(self, image: np.ndarray, n_samples: int = 5, 
                               augmentations: List[str] = None) -> List[np.ndarray]:
        """
        단일 이미지에서 증강된 배치 생성
        
        Args:
            image: 입력 이미지
            n_samples: 생성할 샘플 수
            augmentations: 적용할 증강 기법 목록
            
        Returns:
            List[np.ndarray]: 증강된 이미지 목록
        """
        return [self.augment(image, augmentations) for _ in range(n_samples)]


# 증강기 사용 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 증강기 생성
    augmentor = ImageAugmentor(seed=42)
    
    # 테스트 이미지 생성 (랜덤 노이즈)
    test_image = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    # 개별 증강 테스트
    flipped = augmentor.random_flip(test_image)
    rotated = augmentor.random_rotation(test_image)
    brightened = augmentor.random_brightness(test_image)
    
    print("개별 증강 완료")
    
    # 복합 증강 테스트
    augmented = augmentor.augment(test_image)
    print("복합 증강 완료")
    
    # 배치 증강 테스트
    batch = [test_image] * 5
    augmented_batch = augmentor.augment_batch(batch)
    print(f"배치 증강 완료: {len(augmented_batch)}개 이미지")