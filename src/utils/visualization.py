"""
시각화 유틸리티 모듈

이 모듈은 센서 데이터 및 모델 성능 시각화를 위한 함수들을 제공합니다.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import torch
import logging
from sklearn.metrics import confusion_matrix

from src.models.sensor.lstm_classifier import MultiSensorLSTMClassifier
from src.data.sensor.sensor_processor import INVERSE_STATE_MAPPING

# 로깅 설정
logger = logging.getLogger(__name__)

def plot_training_history(
    history: Dict[str, List[float]],
    plot_dir: str,
    filename: str = 'training_history.png'
) -> str:
    """
    학습 이력(손실 및 정확도) 시각화
    
    Args:
        history: 학습 이력 딕셔너리 ('train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy')
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 그래프 생성
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], 'b-', label='Train Loss')
    plt.plot(history['valid_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], 'b-', label='Train Accuracy')
    plt.plot(history['valid_accuracy'], 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"학습 이력 그래프 저장 완료: {filepath}")
    
    return filepath

def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    plot_dir: str,
    filename: str = 'confusion_matrix.png'
) -> str:
    """
    혼동 행렬 시각화
    
    Args:
        y_true: 실제 레이블 리스트
        y_pred: 예측 레이블 리스트
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 혼동 행렬 계산
    cm = confusion_matrix(y_true, y_pred)
    
    # 클래스 레이블 매핑
    target_names = [INVERSE_STATE_MAPPING[i] for i in range(len(INVERSE_STATE_MAPPING))]
    
    # 그래프 생성
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"혼동 행렬 그래프 저장 완료: {filepath}")
    
    return filepath

def plot_class_distribution(
    labels: List[int],
    plot_dir: str,
    filename: str = 'class_distribution.png'
) -> str:
    """
    클래스 분포 시각화
    
    Args:
        labels: 클래스 레이블 리스트
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 클래스 별 개수 계산
    unique_classes, counts = np.unique(labels, return_counts=True)
    
    # 클래스 레이블 매핑
    target_names = [INVERSE_STATE_MAPPING[i] for i in unique_classes]
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    bars = plt.bar(target_names, counts, color=sns.color_palette("husl", len(unique_classes)))
    
    # 막대 위에 개수 표시
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5, 
                str(count), ha='center', va='bottom')
    
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"클래스 분포 그래프 저장 완료: {filepath}")
    
    return filepath

def plot_sensor_data(
    data: np.ndarray,
    window_start: int = 0,
    window_size: int = 100,
    plot_dir: str = 'plots',
    filename: str = 'sensor_data.png'
) -> str:
    """
    센서 데이터 시각화
    
    Args:
        data: 센서 데이터 배열, 형태 (samples, features)
        window_start: 시각화 시작 인덱스
        window_size: 시각화할 샘플 수
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 데이터 형태 확인
    n_samples, n_features = data.shape
    
    # 시각화 윈도우 조정
    window_end = min(window_start + window_size, n_samples)
    actual_window_size = window_end - window_start
    
    # 그래프 생성
    plt.figure(figsize=(12, 8))
    
    # 각 센서별 데이터 시각화
    for i in range(n_features):
        plt.subplot(n_features, 1, i+1)
        plt.plot(range(window_start, window_end), data[window_start:window_end, i])
        plt.title(f'Sensor {i+1}')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if i == n_features - 1:  # 마지막 서브플롯에만 x축 레이블 추가
            plt.xlabel('Time Step')
    
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"센서 데이터 그래프 저장 완료: {filepath}")
    
    return filepath

def plot_attention_weights(
    model: MultiSensorLSTMClassifier,
    data: torch.Tensor,
    labels: torch.Tensor,
    plot_dir: str,
    filename: str = 'attention_weights.png',
    n_samples: int = 5
) -> str:
    """
    어텐션 가중치 시각화
    
    Args:
        model: LSTM 분류 모델
        data: 입력 데이터, 형태 (batch_size, sequence_length, input_size)
        labels: 레이블 데이터
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        n_samples: 시각화할 샘플 수
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 어텐션 가중치 계산
    model.eval()
    attention_weights = model.get_attention_weights(data[:n_samples])
    attention_weights = attention_weights.cpu().numpy()
    
    # 예측 결과 계산
    with torch.no_grad():
        outputs = model(data[:n_samples])
        _, predicted = torch.max(outputs, 1)
    
    predicted = predicted.cpu().numpy()
    labels = labels[:n_samples].cpu().numpy()
    
    # 클래스 레이블 매핑
    target_names = [INVERSE_STATE_MAPPING[i] for i in range(len(INVERSE_STATE_MAPPING))]
    
    # 그래프 생성
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        # 어텐션 가중치 시각화
        im = ax.imshow(attention_weights[i].reshape(1, -1), cmap='hot', aspect='auto')
        ax.set_title(f'Sample {i+1}: True={target_names[labels[i]]}, Pred={target_names[predicted[i]]}')
        ax.set_xlabel('Sequence Position')
        ax.set_yticks([])
        
        # 컬러바 추가
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"어텐션 가중치 그래프 저장 완료: {filepath}")
    
    return filepath

def plot_feature_importance(
    model: MultiSensorLSTMClassifier,
    data: torch.Tensor,
    labels: torch.Tensor,
    plot_dir: str,
    filename: str = 'feature_importance.png'
) -> str:
    """
    특성 중요도 시각화 (간단한 치환 기반 방법)
    
    Args:
        model: LSTM 분류 모델
        data: 입력 데이터, 형태 (batch_size, sequence_length, input_size)
        labels: 레이블 데이터
        plot_dir: 그래프를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 그래프 파일 경로
    """
    os.makedirs(plot_dir, exist_ok=True)
    
    # 기준 정확도 계산
    model.eval()
    with torch.no_grad():
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        baseline_acc = (predicted == labels).float().mean().item()
    
    # 각 특성별 중요도 계산
    importances = []
    n_features = data.shape[2]
    
    for feature_idx in range(n_features):
        # 특성 값 치환
        perturbed_data = data.clone()
        perturbed_data[:, :, feature_idx] = torch.mean(perturbed_data[:, :, feature_idx])
        
        # 정확도 변화 측정
        with torch.no_grad():
            outputs = model(perturbed_data)
            _, predicted = torch.max(outputs, 1)
            perturbed_acc = (predicted == labels).float().mean().item()
        
        # 중요도 = 기준 정확도 - 치환 후 정확도
        importance = baseline_acc - perturbed_acc
        importances.append(importance)
    
    # 그래프 생성
    plt.figure(figsize=(10, 6))
    
    # 중요도 순으로 정렬
    sorted_idx = np.argsort(importances)
    feature_names = [f'Sensor {i+1}' for i in range(n_features)]
    
    # 특성 중요도 시각화
    plt.barh([feature_names[i] for i in sorted_idx], [importances[i] for i in sorted_idx])
    plt.title('Feature Importance')
    plt.xlabel('Importance (Accuracy Drop)')
    plt.grid(True, linestyle='--', alpha=0.7, axis='x')
    plt.tight_layout()
    
    # 그래프 저장
    filepath = os.path.join(plot_dir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"특성 중요도 그래프 저장 완료: {filepath}")
    
    return filepath

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성
    test_history = {
        'train_loss': [0.8, 0.6, 0.4, 0.3, 0.2],
        'valid_loss': [0.9, 0.7, 0.5, 0.4, 0.35],
        'train_accuracy': [0.6, 0.7, 0.8, 0.85, 0.9],
        'valid_accuracy': [0.55, 0.65, 0.75, 0.8, 0.82]
    }
    
    y_true = np.random.randint(0, 4, size=100)
    y_pred = np.random.randint(0, 4, size=100)
    
    # 테스트 디렉토리
    test_dir = 'temp_plots'
    os.makedirs(test_dir, exist_ok=True)
    
    # 학습 이력 그래프
    plot_training_history(test_history, test_dir)
    
    # 혼동 행렬 그래프
    plot_confusion_matrix(y_true, y_pred, test_dir)
    
    # 클래스 분포 그래프
    plot_class_distribution(y_true, test_dir)
    
    # 센서 데이터 그래프
    sensor_data = np.random.randn(500, 4)
    plot_sensor_data(sensor_data, window_start=0, window_size=200, plot_dir=test_dir)
    
    print(f"테스트 그래프가 {test_dir} 디렉토리에 저장되었습니다.")