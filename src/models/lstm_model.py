import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

# 재현성을 위한 시드 설정
torch.manual_seed(42)
np.random.seed(42)

# 장치 설정 (GPU 사용 가능시 GPU, 아니면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 시계열 데이터셋 클래스 정의
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 완전 연결 레이어
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM 출력 (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시퀀스의 출력만 사용
        out = self.fc(lstm_out[:, -1, :])
        return out

# 데이터 준비 함수
def prepare_time_series_data(df, target_col, feature_cols, sequence_length, forecast_horizon=1, test_size=0.2):
    """
    시계열 데이터를 LSTM 입력 형식으로 준비
    
    Args:
        df: pandas DataFrame, 시계열 데이터
        target_col: 예측할 대상 컬럼 이름
        feature_cols: 특성으로 사용할 컬럼 이름 리스트
        sequence_length: 입력 시퀀스 길이
        forecast_horizon: 예측 기간
        test_size: 테스트 데이터 비율
    
    Returns:
        train_loader, val_loader, test_loader, scalers 딕셔너리
    """
    # 특성(X)과 타겟(y) 분리
    data = df[feature_cols + [target_col]].values
    
    # 데이터 정규화
    scalers = {}
    scaled_data = np.zeros_like(data, dtype=np.float32)
    
    for i, col in enumerate(feature_cols + [target_col]):
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data[:, i] = scaler.fit_transform(data[:, i].reshape(-1, 1)).flatten()
        scalers[col] = scaler
    
    # 시퀀스 데이터 생성
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length - forecast_horizon + 1):
        X.append(scaled_data[i:i+sequence_length, :-1])  # 특성 컬럼
        y.append(scaled_data[i+sequence_length+forecast_horizon-1, -1])  # 타겟 컬럼
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # 학습/검증/테스트 데이터 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=test_size, shuffle=False)
    
    # PyTorch 텐서로 변환
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    # 데이터 로더 생성
    train_dataset = TimeSeriesDataset(X_train_tensor, y_train_tensor)
    val_dataset = TimeSeriesDataset(X_val_tensor, y_val_tensor)
    test_dataset = TimeSeriesDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, val_loader, test_loader, scalers

# 모델 학습 함수
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, patience=10):
    """
    LSTM 모델 학습
    
    Args:
        model: LSTM 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        num_epochs: 학습 에폭 수
        learning_rate: 학습률
        patience: Early stopping 인내 횟수
    
    Returns:
        trained model, 학습 손실 리스트, 검증 손실 리스트
    """
    # 손실 함수와 최적화 도구 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 및 검증 손실 기록
    train_losses = []
    val_losses = []
    
    # Early stopping 설정
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            
            # 순방향 전파
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # 에폭당 평균 학습 손실
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # 진행 상황 출력
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Early stopping 확인
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            early_stopping_counter += 1
            
        if early_stopping_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break
    
    # 최적의 모델 상태로 복원
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

# 모델 평가 함수
def evaluate_model(model, test_loader, scaler, plot=True):
    """
    학습된 모델을 평가
    
    Args:
        model: 학습된 LSTM 모델
        test_loader: 테스트 데이터 로더
        scaler: 타겟 변수 스케일러
        plot: 결과 시각화 여부
    
    Returns:
        MSE, MAE, 예측값, 실제값
    """
    model.eval()
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(y_batch.cpu().numpy())
    
    # NumPy 배열로 변환
    predictions = np.array(predictions)
    actual_values = np.array(actual_values)
    
    # 역정규화
    predictions_rescaled = scaler.inverse_transform(predictions)
    actual_values_rescaled = scaler.inverse_transform(actual_values)
    
    # 평가 지표 계산
    mse = np.mean((predictions_rescaled - actual_values_rescaled) ** 2)
    mae = np.mean(np.abs(predictions_rescaled - actual_values_rescaled))
    
    print(f'Test MSE: {mse:.4f}, MAE: {mae:.4f}')
    
    # 결과 시각화
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(actual_values_rescaled, label='Actual')
        plt.plot(predictions_rescaled, label='Predicted')
        plt.title('LSTM Model Predictions vs Actual Values')
        plt.xlabel('Time Steps')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return mse, mae, predictions_rescaled, actual_values_rescaled

# 예측 함수
def predict_future(model, last_sequence, scaler, n_steps=30):
    """
    학습된 모델을 사용하여 미래 값 예측
    
    Args:
        model: 학습된 LSTM 모델
        last_sequence: 마지막 입력 시퀀스
        scaler: 타겟 변수 스케일러
        n_steps: 예측할 미래 단계 수
    
    Returns:
        예측값 배열
    """
    model.eval()
    future_predictions = []
    
    # 마지막 시퀀스 복사
    current_sequence = last_sequence.clone()
    
    with torch.no_grad():
        for _ in range(n_steps):
            # 현재 시퀀스로 예측
            prediction = model(current_sequence.unsqueeze(0))
            future_predictions.append(prediction.item())
            
            # 시퀀스 업데이트 (가장 오래된 값 제거, 새 예측값 추가)
            new_sequence = torch.cat([current_sequence[:, 1:, :], 
                                      prediction.view(1, 1, -1)], dim=1)
            current_sequence = new_sequence
    
    # 역정규화
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_rescaled = scaler.inverse_transform(future_predictions)
    
    return future_predictions_rescaled

# 예제: 모델 초기화 및 사용 방법
def main():
    """
    예제 코드 - 실제 데이터셋으로 교체 필요
    """
    # 하이퍼파라미터
    input_size = 3  # 입력 특성 수 (센서 개수)
    hidden_size = 64  # LSTM 은닉층 크기
    num_layers = 2  # LSTM 레이어 수
    output_size = 1  # 출력 크기 (예측 값)
    sequence_length = 24  # 입력 시퀀스 길이 (24시간)
    learning_rate = 0.001
    num_epochs = 100
    
    # 모델 초기화
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=output_size
    ).to(device)
    
    print(model)
    
    # 1. 실제 사용 시나리오: 
    # - 데이터 준비
    # - 모델 학습
    # - 모델 평가
    # - 모델 저장 및 배포
    
    # 2. 모델 저장 예제:
    # torch.save(model.state_dict(), 'lstm_model.pth')
    
    # 3. 모델 로드 예제:
    # loaded_model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # loaded_model.load_state_dict(torch.load('lstm_model.pth'))

if __name__ == "__main__":
    main()