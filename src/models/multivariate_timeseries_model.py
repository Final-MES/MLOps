import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from datetime import datetime
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from src.models.lstm_model import LSTMModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultivariateLSTMClassifier(nn.Module):
    """
    다변량 시계열 데이터에 대한 LSTM 기반 분류 모델
    
    Args:
        input_size (int): 입력 특성의 수 (다변량 센서 변수 개수)
        hidden_size (int): LSTM 은닉층의 크기
        num_layers (int): LSTM 레이어의 수
        num_classes (int): 분류할 클래스 수 (정상, Type1, Type2, Type3)
        dropout_rate (float, optional): 드롭아웃 비율. 기본값은 0.2
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 num_classes: int, 
                 dropout_rate: float = 0.2):
        super(MultivariateLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 어텐션 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 분류를 위한 완전 연결 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x (torch.Tensor): 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 클래스별 확률, 형태 (batch_size, num_classes)
        """
        # LSTM 출력 (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 어텐션 가중치 계산
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # 어텐션 가중치를 사용하여 컨텍스트 벡터 계산
        context = torch.sum(lstm_out * attn_weights, dim=1)
        
        # 최종 분류 결과
        out = self.fc(context)
        return out


def prepare_multivariate_data(
    data_path: str,
    state_column: str = 'state',
    time_column: str = 'time',
    feature_cols: Optional[List[str]] = None,
    sequence_length: int = 24,
    test_size: float = 0.2,
    val_size: float = 0.2,
    scaling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    다변량 시계열 데이터를 분류 모델 학습을 위해 준비합니다.
    
    Args:
        data_path (str): 보간 및 전처리된 다변량 시계열 데이터 파일 경로
        state_column (str): 상태 라벨 컬럼명 (정상, Type1, Type2, Type3)
        time_column (str): 시간 컬럼명
        feature_cols (List[str], optional): 특성 컬럼 목록. None이면 state와 time을 제외한 모든 컬럼 사용
        sequence_length (int): 시퀀스 길이
        test_size (float): 테스트 데이터 비율
        val_size (float): 검증 데이터 비율 (학습 데이터에서의 비율)
        scaling (bool): 데이터 스케일링 여부
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]: 
            (train_loader, val_loader, test_loader, data_info)
    """
    logger.info(f"다변량 시계열 데이터 로드: {data_path}")
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise
    
    # 상태 컬럼 확인
    if state_column not in df.columns:
        logger.error(f"상태 컬럼 '{state_column}'이 데이터에 없습니다.")
        raise ValueError(f"상태 컬럼 '{state_column}'이 데이터에 없습니다.")
    
    # 특성 컬럼 결정
    if feature_cols is None:
        # state와 time 컬럼을 제외한 모든 컬럼 사용
        feature_cols = [col for col in df.columns if col != state_column and col != time_column]
    
    logger.info(f"선택된 특성 변수: {len(feature_cols)} 개")
    
    # 상태 라벨 인코딩
    # 라벨 매핑 생성
    unique_states = df[state_column].unique()
    state_mapping = {state: idx for idx, state in enumerate(unique_states)}
    
    # 매핑을 사용하여 숫자로 변환
    y = df[state_column].map(state_mapping).values
    
    # 특성 추출
    X = df[feature_cols].values
    
    # 데이터 스케일링
    if scaling:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # 시퀀스 데이터 생성
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_sequences.append(X_scaled[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length - 1])  # 시퀀스의 마지막 상태를 라벨로 사용
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.int64)
    
    logger.info(f"시퀀스 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}")
    
    # 학습/검증/테스트 데이터 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences, y_sequences, test_size=test_size, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, shuffle=False
    )
    
    logger.info(f"학습 데이터: {X_train.shape}")
    logger.info(f"검증 데이터: {X_val.shape}")
    logger.info(f"테스트 데이터: {X_test.shape}")
    
    # 클래스 불균형 확인
    class_counts = np.bincount(y_train)
    class_weights = 1. / class_counts
    class_weights = class_weights / np.sum(class_weights) * len(class_counts)
    
    logger.info(f"클래스 분포: {class_counts}")
    logger.info(f"클래스 가중치: {class_weights}")
    
    # PyTorch 텐서로 변환
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 장치: {device}")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 데이터 로더 생성
    batch_size = 32
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 데이터 정보 저장
    data_info = {
        'feature_cols': feature_cols,
        'state_column': state_column,
        'sequence_length': sequence_length,
        'input_size': X_train.shape[2],
        'num_classes': len(unique_states),
        'class_mapping': state_mapping,
        'inverse_class_mapping': {idx: state for state, idx in state_mapping.items()},
        'class_weights': class_weights.tolist(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'scaler': scaler if scaling else None,
        'device': device.type
    }
    
    return train_loader, val_loader, test_loader, data_info


def train_multivariate_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    data_info: Dict[str, Any],
    hidden_size: int = 128,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    epochs: int = 100,
    patience: int = 10,
    model_dir: str = "models",
    model_name: str = "multivariate_lstm_classifier"
) -> Tuple[MultivariateLSTMClassifier, Dict[str, List[float]]]:
    """
    다변량 시계열 분류 모델을 학습합니다.
    
    Args:
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        hidden_size (int): LSTM 은닉층 크기
        num_layers (int): LSTM 레이어 수
        learning_rate (float): 학습률
        epochs (int): 학습 에폭 수
        patience (int): 조기 종료 인내 횟수
        model_dir (str): 모델 저장 디렉토리
        model_name (str): 모델 이름
    
    Returns:
        Tuple[MultivariateLSTMClassifier, Dict[str, List[float]]]: (학습된 모델, 학습 이력)
    """
    # 모델 초기화
    input_size = data_info['input_size']
    num_classes = data_info['num_classes']
    device = torch.device(data_info['device'])
    
    model = MultivariateLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        dropout_rate=0.3
    ).to(device)
    
    logger.info(f"모델 아키텍처: {model}")
    logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
    
    # 클래스 가중치 설정 (불균형 데이터 처리)
    class_weights = torch.tensor(data_info['class_weights'], dtype=torch.float32).to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 이력 추적
    history = {
        'train_loss': [], 
        'train_acc': [], 
        'val_loss': [], 
        'val_acc': []
    }
    
    # 조기 종료 설정
    best_val_loss = float('inf')
    counter = 0
    
    # 모델 저장 경로
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pth")
    model_info_path = os.path.join(model_dir, f"{model_name}_info.json")
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
        
        # 에폭당 평균 학습 손실 및 정확도
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 학습률 조정
        scheduler.step(val_loss)
        
        # 에폭 결과 출력
        logger.info(f"에폭 {epoch+1}/{epochs}, 학습 손실: {train_loss:.4f}, 학습 정확도: {train_acc:.4f}, "
                   f"검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f}")
        
        # 조기 종료 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 최상의 모델 저장
            torch.save(model.state_dict(), model_path)
            logger.info(f"새로운 최적 모델 저장: {model_path} (검증 손실: {val_loss:.4f}, 검증 정확도: {val_acc:.4f})")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"조기 종료 (에폭 {epoch+1})")
                break
    
    # 학습 완료 후 최적 모델 로드
    model.load_state_dict(torch.load(model_path))
    
    # 모델 정보 저장
    model_info = {
        "model_type": "MultivariateLSTMClassifier",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_classes": num_classes,
        "feature_cols": data_info['feature_cols'],
        "state_column": data_info['state_column'],
        "sequence_length": data_info['sequence_length'],
        "class_mapping": data_info['class_mapping'],
        "best_val_loss": best_val_loss,
        "best_val_acc": max(history['val_acc']),
        "epochs_trained": len(history['train_loss']),
        "early_stopping_used": counter >= patience,
        "created_at": datetime.now().isoformat()
    }
    
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"모델 정보 저장: {model_info_path}")
    
    return model, history