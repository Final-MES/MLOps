"""
모델 훈련 및 평가 유틸리티 모듈

이 모듈은 다중 센서 LSTM 분류 모델의 훈련, 검증, 테스트에 필요한 유틸리티 함수를 제공합니다.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import json
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from torch.utils.data import DataLoader, TensorDataset

from src.models.lstm_classifier import MultiSensorLSTMClassifier
from src.data.sensor_processor import INVERSE_STATE_MAPPING

# 로깅 설정
logger = logging.getLogger(__name__)

def prepare_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    device: torch.device,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader]:
    """
    학습 및 검증 데이터를 위한 PyTorch DataLoader를 준비합니다.
    
    Args:
        X_train: 학습 특성 데이터
        y_train: 학습 레이블 데이터
        X_valid: 검증 특성 데이터
        y_valid: 검증 레이블 데이터
        device: 사용할 장치 (CPU/GPU)
        batch_size: 배치 크기
        
    Returns:
        tuple: (학습 데이터 로더, 검증 데이터 로더)
    """
    # 데이터를 텐서로 변환
    X_train_tensor = torch.from_numpy(X_train).to(device)
    y_train_tensor = torch.from_numpy(y_train).to(device)
    X_valid_tensor = torch.from_numpy(X_valid).to(device)
    y_valid_tensor = torch.from_numpy(y_valid).to(device)
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"데이터 로더 준비 완료: 학습={len(train_dataset)} 샘플, 검증={len(valid_dataset)} 샘플")
    
    return train_loader, valid_loader

def train_model(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    model: MultiSensorLSTMClassifier,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    early_stopping_patience: int = 10,
    model_dir: str = "models"
) -> Tuple[MultiSensorLSTMClassifier, Dict[str, List[float]]]:
    """
    다중 센서 LSTM 분류 모델 훈련
    
    Args:
        train_loader: 학습 데이터 로더
        valid_loader: 검증 데이터 로더
        model: 훈련할 모델
        device: 사용할 장치 (CPU/GPU)
        num_epochs: 최대 에폭 수
        learning_rate: 학습률
        early_stopping_patience: 조기 종료 인내 횟수
        model_dir: 모델을 저장할 디렉토리
        
    Returns:
        tuple: (훈련된 모델, 학습 이력)
    """
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 이력 추적
    train_losses = []
    valid_losses = []
    train_accuracies = []
    valid_accuracies = []
    best_valid_loss = float('inf')
    patience_counter = 0
    
    # 모델 저장 경로
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = os.path.join(model_dir, 'best_model.pth')
    
    logger.info(f"모델 훈련 시작: 에폭={num_epochs}, 학습률={learning_rate}")
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # 정확도 계산
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_accuracy = train_correct / train_total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # 검증 모드
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                # 정확도 계산
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        
        valid_loss /= len(valid_loader.dataset)
        valid_accuracy = valid_correct / valid_total
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        # 스케줄러 업데이트
        scheduler.step(valid_loss)
        
        # 진행 상황 출력
        logger.info(f"에폭 [{epoch+1}/{num_epochs}], 학습 손실: {train_loss:.4f}, 학습 정확도: {train_accuracy:.4f}, "
                   f"검증 손실: {valid_loss:.4f}, 검증 정확도: {valid_accuracy:.4f}")
        
        # 조기 종료 확인
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # 최적 모델 저장
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"새로운 최적 모델 저장: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"조기 종료 (에폭 {epoch+1})")
                break
    
    # 최적 모델 로드
    if os.path.exists(best_model_path):
        logger.info(f"최적 모델 로드: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    
    # 학습 이력 반환
    history = {
        'train_loss': train_losses,
        'valid_loss': valid_losses,
        'train_accuracy': train_accuracies,
        'valid_accuracy': valid_accuracies
    }
    
    return model, history

def evaluate_model(
    model: MultiSensorLSTMClassifier,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, Any]:
    """
    학습된 모델 평가
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        device: 사용할 장치 (CPU/GPU)
        
    Returns:
        dict: 평가 결과 (정확도, 혼동 행렬 등)
    """
    # 평가 모드
    model.eval()
    correct = 0
    total = 0
    
    # 예측 및 실제 레이블 저장
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 레이블 저장
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    accuracy = correct / total
    logger.info(f"테스트 정확도: {accuracy:.4f}")
    
    # 혼동 행렬 및 분류 보고서 생성
    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_true, y_pred)
    
    # 클래스 레이블 매핑
    target_names = [INVERSE_STATE_MAPPING[i] for i in range(len(INVERSE_STATE_MAPPING))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    
    # 결과 반환
    result = {
        'accuracy': accuracy,
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    return result

def save_model_info(
    model: MultiSensorLSTMClassifier,
    model_dir: str,
    sequence_length: int
) -> str:
    """
    모델 정보를 JSON 파일로 저장
    
    Args:
        model: 저장할 모델
        model_dir: 모델을 저장할 디렉토리
        sequence_length: 사용된 시퀀스 길이
        
    Returns:
        str: 저장된 JSON 파일 경로
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # 모델 아키텍처 정보
    model_info = model.get_model_info()
    model_info.update({
        "sequence_length": sequence_length,
        "created_at": datetime.now().isoformat()
    })
    
    # JSON 파일 저장
    model_info_path = os.path.join(model_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    logger.info(f"모델 정보 저장 완료: {model_info_path}")
    
    return model_info_path

def save_evaluation_result(
    evaluation_result: Dict[str, Any],
    output_dir: str,
    filename: str = 'evaluation_result.json'
) -> str:
    """
    평가 결과를 JSON 파일로 저장
    
    Args:
        evaluation_result: 평가 결과 딕셔너리
        output_dir: 결과를 저장할 디렉토리
        filename: 저장할 파일 이름
        
    Returns:
        str: 저장된 JSON 파일 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON 파일 저장
    eval_path = os.path.join(output_dir, filename)
    with open(eval_path, 'w') as f:
        json.dump(evaluation_result, f, indent=4)
    
    logger.info(f"평가 결과 저장 완료: {eval_path}")
    
    return eval_path

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 테스트 데이터 생성
    batch_size, seq_length, input_size = 16, 50, 4
    test_X = np.random.randn(100, seq_length, input_size).astype(np.float32)
    test_y = np.random.randint(0, 4, size=100).astype(np.int64)
    
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 데이터 분할
    val_size = 20
    X_train, X_val = test_X[:-val_size], test_X[-val_size:]
    y_train, y_val = test_y[:-val_size], test_y[-val_size:]
    
    # 데이터 로더 준비
    train_loader, val_loader = prepare_dataloaders(
        X_train, y_train, X_val, y_val, device, batch_size=8
    )
    
    # 모델 초기화
    model = MultiSensorLSTMClassifier(
        input_size=input_size,
        hidden_size=32,
        num_layers=1,
        num_classes=4
    ).to(device)
    
    # 모델 훈련 (테스트용으로 에폭 수 적게 설정)
    model, history = train_model(
        train_loader=train_loader,
        valid_loader=val_loader,
        model=model,
        device=device,
        num_epochs=5,
        model_dir="temp_models"
    )
    
    # 모델 평가
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val).to(device), torch.from_numpy(y_val).to(device)),
        batch_size=8
    )
    
    evaluation_result = evaluate_model(model, test_loader, device)
    print(f"평가 결과: 정확도={evaluation_result['accuracy']:.4f}")
    
    # 모델 정보 저장
    save_model_info(model, "temp_models", sequence_length=seq_length)
    
    # 평가 결과 저장
    save_evaluation_result(evaluation_result, "temp_models")