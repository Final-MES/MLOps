import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any, List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import logging
import json
from datetime import datetime
from pathlib import Path

from src.models.sensor.lstm_model import LSTMModel
from src.utils.paths import get_data_path, get_model_path, ensure_dir, get_project_paths

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def prepare_data(
    data_path: Path,
    feature_cols: Optional[List[str]] = None,
    target_col: Optional[str] = None,
    sequence_length: int = 24,
    test_size: float = 0.2,
    val_size: float = 0.2,
    scaling: bool = True
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]:
    """
    시계열 데이터를 준비하고 데이터 로더를 반환합니다.
    
    Args:
        data_path (Path): 데이터 파일 경로
        feature_cols (List[str], optional): 특성 컬럼 목록. None이면 타겟을 제외한 모든 컬럼 사용
        target_col (str, optional): 타겟 컬럼명. None이면 마지막 컬럼 사용
        sequence_length (int): 시퀀스 길이
        test_size (float): 테스트 데이터 비율
        val_size (float): 검증 데이터 비율 (학습 데이터에서의 비율)
        scaling (bool): 데이터 스케일링 여부
    
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Any]]: 
            (train_loader, val_loader, test_loader, data_info)
    """
    logger.info(f"데이터 로드 중: {data_path}")
    
    # 데이터 로드
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        logger.error(f"데이터 로드 실패: {e}")
        raise
    
    # 특성 및 타겟 컬럼 결정
    if feature_cols is None:
        # 타겟 컬럼이 지정되었으면 제외한 모든 컬럼 사용
        if target_col is not None:
            feature_cols = [col for col in df.columns if col != target_col]
        else:
            # 타겟 컬럼도 지정되지 않았으면 마지막 컬럼을 타겟으로 사용
            feature_cols = df.columns[:-1].tolist()
            target_col = df.columns[-1]
    elif target_col is None:
        # 특성 컬럼만 지정된 경우 마지막 컬럼을 타겟으로 사용
        target_col = df.columns[-1]
    
    logger.info(f"특성 변수: {feature_cols}")
    logger.info(f"타겟 변수: {target_col}")
    
    # 특성 및 타겟 분리
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # 데이터 스케일링
    scalers = {}
    if scaling:
        X_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        X_scaled = X_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y)
        
        scalers['X_scaler'] = X_scaler
        scalers['y_scaler'] = y_scaler
    else:
        X_scaled = X
        y_scaled = y
    
    # 시퀀스 데이터 생성
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_sequences.append(X_scaled[i:i + sequence_length])
        y_sequences.append(y_scaled[i + sequence_length])
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    
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
    
    # PyTorch 텐서로 변환
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"사용 장치: {device}")
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 데이터 정보 저장
    data_info = {
        'feature_cols': feature_cols,
        'target_col': target_col,
        'sequence_length': sequence_length,
        'input_size': X_train.shape[2],
        'output_size': 1,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'scalers': scalers,
        'device': device.type
    }
    
    return train_loader, val_loader, test_loader, data_info


def train_lstm_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    data_info: Dict[str, Any],
    hidden_size: int = 64,
    num_layers: int = 2,
    learning_rate: float = 0.001,
    epochs: int = 100,
    patience: int = 10,
    model_name: str = "lstm_model"
) -> Tuple[LSTMModel, Dict[str, List[float]]]:
    """
    LSTM 모델을 학습하고 학습된 모델과 손실 이력을 반환합니다.
    
    Args:
        train_loader (DataLoader): 학습 데이터 로더
        val_loader (DataLoader): 검증 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        hidden_size (int): LSTM 은닉층 크기
        num_layers (int): LSTM 레이어 수
        learning_rate (float): 학습률
        epochs (int): 학습 에폭 수
        patience (int): 조기 종료 인내 횟수
        model_name (str): 모델 이름
    
    Returns:
        Tuple[LSTMModel, Dict[str, List[float]]]: (학습된 모델, 손실 이력)
    """
    # 모델 초기화
    input_size = data_info['input_size']
    output_size = data_info['output_size']
    device = torch.device(data_info['device'])
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    
    logger.info(f"모델 아키텍처: {model}")
    logger.info(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 이력 추적
    history = {'train_loss': [], 'val_loss': []}
    
    # 조기 종료 설정
    best_val_loss = float('inf')
    counter = 0
    
    # 모델 저장 경로
    model_path = get_model_path(model_name)
    model_dir = model_path.parent
    ensure_dir(model_dir)
    
    # 모델 정보 파일 경로
    model_info_path = model_dir / f"{model_name}_info.json"
    
    # 학습 루프
    for epoch in range(epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 에폭당 평균 학습 손실
        train_loss = train_loss / len(train_loader)
        history['train_loss'].append(train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        history['val_loss'].append(val_loss)
        
        # 에폭 결과 출력
        logger.info(f"에폭 {epoch+1}/{epochs}, 학습 손실: {train_loss:.6f}, 검증 손실: {val_loss:.6f}")
        
        # 조기 종료 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # 최상의 모델 저장
            torch.save(model.state_dict(), model_path)
            logger.info(f"새로운 최적 모델 저장: {model_path}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"조기 종료 (에폭 {epoch+1})")
                break
    
    # 학습 완료 후 최적 모델 로드
    model.load_state_dict(torch.load(model_path))
    
    # 모델 정보 저장
    model_info = {
        "model_type": "LSTM",
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "output_size": output_size,
        "feature_cols": data_info['feature_cols'],
        "target_col": data_info['target_col'],
        "sequence_length": data_info['sequence_length'],
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history['train_loss']),
        "early_stopping_used": counter >= patience,
        "created_at": datetime.now().isoformat()
    }
    
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info(f"모델 정보 저장: {model_info_path}")
    
    return model, history


def evaluate_model(
    model: LSTMModel,
    test_loader: DataLoader,
    data_info: Dict[str, Any],
    plot: bool = True,
    save_plot: bool = True,
    plot_name: str = "prediction_results"
) -> Dict[str, float]:
    """
    학습된 모델을 평가합니다.
    
    Args:
        model (LSTMModel): 평가할 모델
        test_loader (DataLoader): 테스트 데이터 로더
        data_info (Dict[str, Any]): 데이터 정보
        plot (bool): 결과 시각화 여부
        save_plot (bool): 시각화 결과 저장 여부
        plot_name (str): 시각화 결과 파일명
    
    Returns:
        Dict[str, float]: 평가 지표
    """
    model.eval()
    criterion = nn.MSELoss()
    device = torch.device(data_info['device'])
    
    # 예측 수행
    y_true = []
    y_pred = []
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            # 예측 결과 저장
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())
    
    test_loss = test_loss / len(test_loader)
    
    # 데이터 역정규화
    if 'scalers' in data_info and 'y_scaler' in data_info['scalers']:
        y_scaler = data_info['scalers']['y_scaler']
        y_true = y_scaler.inverse_transform(np.array(y_true))
        y_pred = y_scaler.inverse_transform(np.array(y_pred))
    else:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
    # 평가 지표 계산
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    logger.info(f"테스트 손실: {test_loss:.6f}")
    logger.info(f"MSE: {mse:.6f}")
    logger.info(f"RMSE: {rmse:.6f}")
    logger.info(f"MAE: {mae:.6f}")
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(y_true, label='실제값')
        plt.plot(y_pred, label='예측값')
        plt.title(f'LSTM 모델 예측 결과 (RMSE: {rmse:.4f})')
        plt.xlabel('시간')
        plt.ylabel('값')
        plt.legend()
        plt.grid(True)
        
        if save_plot:
            # plots 디렉토리 경로 가져오기 및 생성
            plot_dir = get_project_paths()["root"] / "plots"
            ensure_dir(plot_dir)
            
            plot_path = plot_dir / f"{plot_name}.png"
            plt.savefig(plot_path)
            logger.info(f"예측 결과 그래프 저장: {plot_path}")
        
        plt.show()
    
    metrics = {
        'test_loss': test_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae
    }
    
    return metrics


if __name__ == "__main__":
    # 예제 사용법
    DATA_PATH = get_data_path("processed") / "sensor_data.csv"
    
    # 데이터 준비
    train_loader, val_loader, test_loader, data_info = prepare_data(
        data_path=DATA_PATH,
        sequence_length=24
    )
    
    # 모델 학습
    model, history = train_lstm_model(
        train_loader=train_loader,
        val_loader=val_loader,
        data_info=data_info,
        hidden_size=64,
        num_layers=2,
        learning_rate=0.001,
        epochs=100,
        patience=10,
        model_name="lstm_model"
    )
    
    # 모델 평가
    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        data_info=data_info,
        plot=True,
        save_plot=True,
        plot_name="prediction_results"
    )