"""
이미지 데이터 전처리 모듈

이 모듈은 이미지 데이터 전처리를 위한 기능을 제공합니다:
- 이미지 로드 및 저장
- 크기 조정 및 정규화
- 데이터 증강
- 이미지 변환
"""

import os
import numpy as np
import cv2
import glob
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import logging
import random
from PIL import Image

# 로깅 설정
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """이미지 데이터 전처리를 위한 클래스"""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224), 
                 normalize: bool = True, 
                 color_mode: str = 'rgb'):
        """
        이미지 전처리기 초기화
        
        Args:
            target_size: 대상 이미지 크기 (height, width)
            normalize: 픽셀값 정규화 여부 (0~1 범위로)
            color_mode: 색상 모드 ('rgb', 'bgr', 'grayscale')
        """
        self.target_size = target_size
        self.normalize = normalize
        
        if color_mode.lower() not in ['rgb', 'bgr', 'grayscale']:
            logger.warning(f"지원하지 않는 색상 모드: {color_mode}. 'rgb'로 설정합니다.")
            self.color_mode = 'rgb'
        else:
            self.color_mode = color_mode.lower()
    
    def load_image(self, image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """
        이미지 파일 로드
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Optional[np.ndarray]: 로드된 이미지 또는 실패 시 None
        """
        try:
            # OpenCV로 이미지 로드
            image = cv2.imread(str(image_path))
            
            if image is None:
                logger.error(f"이미지를 로드할 수 없습니다: {image_path}")
                return None
            
            # 색상 모드 변환
            if self.color_mode == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif self.color_mode == 'grayscale':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = np.expand_dims(image, axis=-1)  # 차원 추가 (H, W) -> (H, W, 1)
            
            # 크기 조정
            image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
            
            # 정규화
            if self.normalize:
                image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"이미지 로드 중 오류 발생: {e}")
            return None
    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> bool:
        """
        이미지 저장
        
        Args:
            image: 저장할 이미지 배열
            output_path: 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            
            # 정규화된 이미지인 경우 0-255 범위로 변환
            if self.normalize and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 색상 모드 변환
            if self.color_mode == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif self.color_mode == 'grayscale' and image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze()  # (H, W, 1) -> (H, W)
            
            # 이미지 저장
            cv2.imwrite(str(output_path), image)
            return True
            
        except Exception as e:
            logger.error(f"이미지 저장 중 오류 발생: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        단일 이미지 전처리
        
        Args:
            image: 전처리할 이미지
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 크기 조정
        processed_image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # 색상 모드 변환
        if self.color_mode == 'rgb' and processed_image.ndim == 3 and processed_image.shape[2] == 3:
            if processed_image.dtype == np.uint8:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        elif self.color_mode == 'grayscale':
            if processed_image.ndim == 3 and processed_image.shape[2] == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                processed_image = np.expand_dims(processed_image, axis=-1)  # 차원 추가
        
        # 정규화
        if self.normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
        
        return processed_image
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        이미지 배치 전처리
        
        Args:
            images: 전처리할 이미지 목록
            
        Returns:
            np.ndarray: 전처리된 이미지 배치 (N, H, W, C)
        """
        processed_images = []
        
        for image in images:
            processed_image = self.preprocess_image(image)
            processed_images.append(processed_image)
        
        return np.array(processed_images)
    
    def load_from_directory(self, directory: Union[str, Path], 
                           pattern: str = '*.jpg', recursive: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """
        디렉토리에서 이미지 로드
        
        Args:
            directory: 이미지가 있는 디렉토리
            pattern: 이미지 파일 패턴
            recursive: 하위 디렉토리 포함 여부
            
        Returns:
            Tuple[List[np.ndarray], List[str]]: (이미지 목록, 파일 경로 목록)
        """
        # 이미지 파일 경로 찾기
        if recursive:
            file_pattern = os.path.join(str(directory), '**', pattern)
            file_paths = glob.glob(file_pattern, recursive=True)
        else:
            file_pattern = os.path.join(str(directory), pattern)
            file_paths = glob.glob(file_pattern)
        
        # 파일 정렬
        file_paths.sort()
        
        # 이미지 로드
        images = []
        valid_paths = []
        
        for file_path in file_paths:
            image = self.load_image(file_path)
            if image is not None:
                images.append(image)
                valid_paths.append(file_path)
        
        logger.info(f"디렉토리에서 {len(images)}개 이미지 로드 완료: {directory}")
        
        return images, valid_paths
    
    def load_from_subdirectories(self, root_dir: Union[str, Path], 
                                pattern: str = '*.jpg') -> Dict[str, List[np.ndarray]]:
        """
        하위 디렉토리별로 이미지 로드 (클래스별 분류 구조 가정)
        
        Args:
            root_dir: 루트 디렉토리
            pattern: 이미지 파일 패턴
            
        Returns:
            Dict[str, List[np.ndarray]]: 클래스별 이미지 목록
        """
        class_images = {}
        
        # 하위 디렉토리 (클래스) 목록
        subdirs = [d for d in os.listdir(str(root_dir)) 
                   if os.path.isdir(os.path.join(str(root_dir), d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(str(root_dir), subdir)
            images, _ = self.load_from_directory(subdir_path, pattern)
            class_images[subdir] = images
            
            logger.info(f"클래스 '{subdir}'에서 {len(images)}개 이미지 로드 완료")
        
        return class_images
    
    def create_dataset(self, root_dir: Union[str, Path], pattern: str = '*.jpg', 
                      test_split: float = 0.2, shuffle: bool = True) -> Dict[str, Any]:
        """
        이미지 데이터셋 생성 (클래스별 하위 디렉토리 구조 가정)
        
        Args:
            root_dir: 루트 디렉토리
            pattern: 이미지 파일 패턴
            test_split: 테스트 데이터 비율
            shuffle: 셔플 여부
            
        Returns:
            Dict[str, Any]: 데이터셋 사전 {'x_train', 'y_train', 'x_test', 'y_test', 'class_names'}
        """
        # 클래스 목록 (하위 디렉토리)
        class_dirs = [d for d in os.listdir(str(root_dir)) 
                      if os.path.isdir(os.path.join(str(root_dir), d))]
        class_dirs.sort()  # 클래스 정렬
        
        if not class_dirs:
            logger.error(f"클래스 디렉토리를 찾을 수 없습니다: {root_dir}")
            return {}
        
        # 클래스 인덱스 매핑
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
        
        # 데이터 로드
        images = []
        labels = []
        
        for cls_name in class_dirs:
            cls_dir = os.path.join(str(root_dir), cls_name)
            cls_images, _ = self.load_from_directory(cls_dir, pattern)
            
            for img in cls_images:
                images.append(img)
                labels.append(class_to_idx[cls_name])
            
            logger.info(f"클래스 '{cls_name}' (인덱스 {class_to_idx[cls_name]})에서 {len(cls_images)}개 이미지 로드 완료")
        
        # 배열 변환
        X = np.array(images)
        y = np.array(labels)
        
        # 데이터 분할
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        
        test_size = int(len(X) * test_split)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        logger.info(f"데이터셋 생성 완료: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return {
            'x_train': X_train,
            'y_train': y_train,
            'x_test': X_test,
            'y_test': y_test,
            'class_names': class_dirs,
            'class_to_idx': class_to_idx
        }
    
    def extract_features(self, image: np.ndarray, method: str = 'hog') -> np.ndarray:
        """
        이미지에서 특성 추출
        
        Args:
            image: 입력 이미지
            method: 특성 추출 방법 ('hog', 'lbp', 'orb', 'sift')
            
        Returns:
            np.ndarray: 추출된 특성 벡터
        """
        # 이미지 전처리
        img = self.preprocess_image(image)
        
        # 그레이스케일 변환 (필요한 경우)
        if self.color_mode != 'grayscale' and img.ndim == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img if img.ndim == 2 else img[:, :, 0]
        
        # 8비트 정수형으로 변환 (필요한 경우)
        if gray.dtype != np.uint8:
            if gray.max() <= 1.0:
                gray = (gray * 255).astype(np.uint8)
            else:
                gray = gray.astype(np.uint8)
        
        if method == 'hog':
            # HOG (Histogram of Oriented Gradients) 특성 추출
            try:
                from skimage.feature import hog
                features, _ = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                              cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
                return features
            except ImportError:
                logger.warning("scikit-image가 설치되지 않았습니다. 간단한 특성 추출로 대체합니다.")
                # 간단한 특성: 픽셀 히스토그램
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                return hist.flatten()
                
        elif method == 'lbp':
            # LBP (Local Binary Pattern) 특성 추출
            try:
                from skimage.feature import local_binary_pattern
                radius = 3
                n_points = 8 * radius
                lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
                hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
                # 히스토그램 정규화
                hist = hist.astype(np.float32) / (hist.sum() + 1e-7)
                return hist
            except ImportError:
                logger.warning("scikit-image가 설치되지 않았습니다. 간단한 특성 추출로 대체합니다.")
                # 간단한 특성: 픽셀 히스토그램
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                return hist.flatten()
                
        elif method == 'orb':
            # ORB (Oriented FAST and Rotated BRIEF) 특성 추출
            orb = cv2.ORB_create()
            keypoints, descriptors = orb.detectAndCompute(gray, None)
            if descriptors is None:
                logger.warning("ORB 특성을 추출할 수 없습니다. 빈 배열 반환.")
                return np.array([])
            # 고정 길이 특성 벡터 생성
            return descriptors.flatten()
            
        elif method == 'sift':
            # SIFT (Scale-Invariant Feature Transform) 특성 추출
            try:
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                if descriptors is None:
                    logger.warning("SIFT 특성을 추출할 수 없습니다. 빈 배열 반환.")
                    return np.array([])
                # 고정 길이 특성 벡터 생성
                return descriptors.flatten()
            except:
                logger.warning("SIFT를 사용할 수 없습니다. ORB로 대체합니다.")
                return self.extract_features(image, method='orb')
        
        else:
            logger.warning(f"지원하지 않는 특성 추출 방법: {method}. HOG로 대체합니다.")
            return self.extract_features(image, method='hog')
    
    def extract_deep_features(self, image: np.ndarray, model_name: str = 'vgg16') -> np.ndarray:
        """
        사전학습된 CNN 모델을 사용한 딥 특성 추출
        
        Args:
            image: 입력 이미지
            model_name: 모델 이름 ('vgg16', 'resnet50', 'mobilenet')
            
        Returns:
            np.ndarray: 추출된 특성 벡터
        """
        try:
            # Keras 모델 임포트
            from torchvision.models import vgg16, resnet50, mobilenet
    
            
            # 이미지 전처리
            img = self.preprocess_image(image)
            
            # 배치 차원 추가
            img_batch = np.expand_dims(img, axis=0)
            
            # 모델 구성 및 특성 추출
            if model_name == 'vgg16':
                base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
                preprocessor = vgg16.preprocess_input
            elif model_name == 'resnet50':
                base_model = resnet50.ResNet50(weights='imagenet', include_top=False, pooling='avg')
                preprocessor = resnet50.preprocess_input
            elif model_name == 'mobilenet':
                base_model = mobilenet.MobileNet(weights='imagenet', include_top=False, pooling='avg')
                preprocessor = mobilenet.preprocess_input
            else:
                logger.warning(f"지원하지 않는 모델: {model_name}. VGG16으로 대체합니다.")
                base_model = vgg16.VGG16(weights='imagenet', include_top=False, pooling='avg')
                preprocessor = vgg16.preprocess_input
            
            # 모델 전처리 적용
            if self.normalize and img_batch.max() <= 1.0:
                # 정규화된 이미지를 다시 0-255 범위로 변환 (Keras 전처리 함수는 0-255 범위 입력 기대)
                img_batch = img_batch * 255.0
            
            x = preprocessor(img_batch)
            features = base_model.predict(x)
            
            return features.flatten()
            
        except ImportError:
            logger.warning("TensorFlow/Keras가 설치되지 않았습니다. 일반 특성 추출로 대체합니다.")
            return self.extract_features(image, method='hog')
    
    def apply_color_transform(self, image: np.ndarray, transform_type: str) -> np.ndarray:
        """
        색상 변환 적용
        
        Args:
            image: 입력 이미지
            transform_type: 변환 유형 ('grayscale', 'hsv', 'lab', 'sepia')
            
        Returns:
            np.ndarray: 변환된 이미지
        """
        # 0-255 범위로 변환 (필요한 경우)
        if self.normalize and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.copy()
        
        # RGB 변환 (필요한 경우)
        if img.ndim == 3 and img.shape[2] == 3 and self.color_mode == 'bgr':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if transform_type == 'grayscale':
            # 그레이스케일 변환
            if img.ndim == 3 and img.shape[2] == 3:
                transformed = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                transformed = np.expand_dims(transformed, axis=-1)  # 차원 추가
            else:
                transformed = img  # 이미 그레이스케일인 경우
        
        elif transform_type == 'hsv':
            # HSV 변환
            if img.ndim == 3 and img.shape[2] == 3:
                transformed = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            else:
                # 그레이스케일 이미지에 적용 불가
                logger.warning("그레이스케일 이미지에 HSV 변환을 적용할 수 없습니다.")
                transformed = img
        
        elif transform_type == 'lab':
            # LAB 변환
            if img.ndim == 3 and img.shape[2] == 3:
                transformed = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            else:
                # 그레이스케일 이미지에 적용 불가
                logger.warning("그레이스케일 이미지에 LAB 변환을 적용할 수 없습니다.")
                transformed = img
        
        elif transform_type == 'sepia':
            # 세피아 변환
            if img.ndim == 3 and img.shape[2] == 3:
                kernel = np.array([
                    [0.272, 0.534, 0.131],
                    [0.349, 0.686, 0.168],
                    [0.393, 0.769, 0.189]
                ])
                transformed = cv2.transform(img, kernel)
            else:
                # 그레이스케일 이미지에 적용 불가
                logger.warning("그레이스케일 이미지에 세피아 변환을 적용할 수 없습니다.")
                transformed = img
        
        else:
            logger.warning(f"지원하지 않는 변환 유형: {transform_type}")
            transformed = img
        
        # 정규화 (필요한 경우)
        if self.normalize:
            transformed = transformed.astype(np.float32) / 255.0
        
        return transformed
    
    def apply_edge_detection(self, image: np.ndarray, 
                           method: str = 'canny', 
                           low_threshold: int = 50, 
                           high_threshold: int = 150) -> np.ndarray:
        """
        이미지에 엣지 검출 적용
        
        Args:
            image: 입력 이미지
            method: 검출 방법 ('canny', 'sobel', 'laplacian')
            low_threshold: Canny 검출기의 하위 임계값
            high_threshold: Canny 검출기의 상위 임계값
            
        Returns:
            np.ndarray: 엣지 이미지
        """
        # 이미지 전처리
        if self.normalize and image.max() <= 1.0:
            img = (image * 255).astype(np.uint8)
        else:
            img = image.copy()
        
        # 그레이스케일 변환 (필요한 경우)
        if img.ndim == 3 and img.shape[2] > 1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if self.color_mode == 'rgb' else cv2.COLOR_BGR2GRAY)
        else:
            gray = img if img.ndim == 2 else img[:, :, 0]
        
        if method == 'canny':
            # Canny 엣지 검출
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        elif method == 'sobel':
            # Sobel 엣지 검출
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            edges = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
        
        elif method == 'laplacian':
            # Laplacian 엣지 검출
            edges = cv2.Laplacian(gray, cv2.CV_64F)
            edges = np.abs(edges).astype(np.uint8)
        
        else:
            logger.warning(f"지원하지 않는 엣지 검출 방법: {method}. Canny로 대체합니다.")
            edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # 차원 맞추기 (필요한 경우)
        if image.ndim == 3 and edges.ndim == 2:
            edges = np.expand_dims(edges, axis=-1)
            
            # 원본 이미지의 채널 수와 동일하게
            if image.shape[2] == 3:
                edges = np.repeat(edges, 3, axis=2)
        
        # 정규화 (필요한 경우)
        if self.normalize:
            edges = edges.astype(np.float32) / 255.0
        
        return edges


    
    def save_image(self, image: np.ndarray, output_path: Union[str, Path]) -> bool:
        """
        이미지 저장
        
        Args:
            image: 저장할 이미지 배열
            output_path: 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(str(output_path)), exist_ok=True)
            
            # 정규화된 이미지인 경우 0-255 범위로 변환
            if self.normalize and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            
            # 색상 모드 변환
            if self.color_mode == 'rgb':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif self.color_mode == 'grayscale' and image.ndim == 3 and image.shape[2] == 1:
                image = image.squeeze()  # (H, W, 1) -> (H, W)
            
            # 이미지 저장
            cv2.imwrite(str(output_path), image)
            return True
            
        except Exception as e:
            logger.error(f"이미지 저장 중 오류 발생: {e}")
            return False
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        단일 이미지 전처리
        
        Args:
            image: 전처리할 이미지
            
        Returns:
            np.ndarray: 전처리된 이미지
        """
        # 크기 조정
        processed_image = cv2.resize(image, (self.target_size[1], self.target_size[0]))
        
        # 색상 모드 변환
        if self.color_mode == 'rgb' and processed_image.ndim == 3 and processed_image.shape[2] == 3:
            if processed_image.dtype == np.uint8:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        elif self.color_mode == 'grayscale':
            if processed_image.ndim == 3 and processed_image.shape[2] == 3:
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
                processed_image = np.expand_dims(processed_image, axis=-1)  # 차원 추가
        
        # 정규화
        if self.normalize:
            processed_image = processed_image.astype(np.float32) / 255.0
        
        return processed_image
    
    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        이미지 배치 전처리
        
        Args:
            images: 전처리할 이미지 목록
            
        Returns:
            np.ndarray: 전처리된 이미지 배치 (N, H, W, C)
        """
        processed_images = []
        
        for image in images:
            processed_image = self.preprocess_image(image)
            processed_images.append(processed_image)
        
        return np.array(processed_images)
    
    def load_from_directory(self, directory: Union[str, Path], 
                           pattern: str = '*.jpg', recursive: bool = False) -> Tuple[List[np.ndarray], List[str]]:
        """
        디렉토리에서 이미지 로드
        
        Args:
            directory: 이미지가 있는 디렉토리
            pattern: 이미지 파일 패턴
            recursive: 하위 디렉토리 포함 여부
            
        Returns:
            Tuple[List[np.ndarray], List[str]]: (이미지 목록, 파일 경로 목록)
        """
        # 이미지 파일 경로 찾기
        if recursive:
            file_pattern = os.path.join(str(directory), '**', pattern)
            file_paths = glob.glob(file_pattern, recursive=True)
        else:
            file_pattern = os.path.join(str(directory), pattern)
            file_paths = glob.glob(file_pattern)
        
        # 파일 정렬
        file_paths.sort()
        
        # 이미지 로드
        images = []
        valid_paths = []
        
        for file_path in file_paths:
            image = self.load_image(file_path)
            if image is not None:
                images.append(image)
                valid_paths.append(file_path)
        
        logger.info(f"디렉토리에서 {len(images)}개 이미지 로드 완료: {directory}")
        
        return images, valid_paths
    
    def load_from_subdirectories(self, root_dir: Union[str, Path], 
                                pattern: str = '*.jpg') -> Dict[str, List[np.ndarray]]:
        """
        하위 디렉토리별로 이미지 로드 (클래스별 분류 구조 가정)
        
        Args:
            root_dir: 루트 디렉토리
            pattern: 이미지 파일 패턴
            
        Returns:
            Dict[str, List[np.ndarray]]: 클래스별 이미지 목록
        """
        class_images = {}
        
        # 하위 디렉토리 (클래스) 목록
        subdirs = [d for d in os.listdir(str(root_dir)) 
                   if os.path.isdir(os.path.join(str(root_dir), d))]
        
        for subdir in subdirs:
            subdir_path = os.path.join(str(root_dir), subdir)
            images, _ = self.load_from_directory(subdir_path, pattern)
            class_images[subdir] = images
            
            logger.info(f"클래스 '{subdir}'에서 {len(images)}개 이미지 로드 완료")
        
        return class_images
    
    def create_dataset(self, root_dir: Union[str, Path], pattern: str = '*.jpg', 
                      test_split: float = 0.2, shuffle: bool = True) -> Dict[str, Any]:
        """
        이미지 데이터셋 생성 (클래스별 하위 디렉토리 구조 가정)
        
        Args:
            root_dir: 루트 디렉토리
            pattern: 이미지 파일 패턴
            test_split: 테스트 데이터 비율
            shuffle: 셔플 여부
            
        Returns:
            Dict[str, Any]: 데이터셋 사전 {'x_train', 'y_train', 'x_test', 'y_test', 'class_names'}
        """
        # 클래스 목록 (하위 디렉토리)
        class_dirs = [d for d in os.listdir(str(root_dir)) 
                      if os.path.isdir(os.path.join(str(root_dir), d))]
        class_dirs.sort()  # 클래스 정렬
        
        if not class_dirs:
            logger.error(f"클래스 디렉토리를 찾을 수 없습니다: {root_dir}")
            return {}
        
        # 클래스 인덱스 매핑
        class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
        
        # 데이터 로드
        images = []
        labels = []
        
        for cls_name in class_dirs:
            cls_dir = os.path.join(str(root_dir), cls_name)
            cls_images, _ = self.load_from_directory(cls_dir, pattern)
            
            for img in cls_images:
                images.append(img)
                labels.append(class_to_idx[cls_name])
            
            logger.info(f"클래스 '{cls_name}' (인덱스 {class_to_idx[cls_name]})에서 {len(cls_images)}개 이미지 로드 완료")
        
        # 배열 변환
        X = np.array(images)
        y = np.array(labels)
        
        # 데이터 분할
        indices = np.arange(len(X))
        if shuffle:
            np.random.shuffle(indices)
        
        test_size = int(len(X) * test_split)
        test_indices = indices[:test_size]
        train_indices = indices[test_size:]
        
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test = X[test_indices]
        y_test = y[test_indices]
        
        logger.info(f"데이터셋 생성 완료: X_train={X_train.shape}, X_test={X_test.shape}")
        
        return {
            'x_train': X_train,
            'y_train': y_train,
            'x_test': X_test,
            'y_test': y_test,
            'class_names': class_dirs,
            'class_to_idx': class_to_idx
        }
