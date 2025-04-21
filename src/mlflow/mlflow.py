import os
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# 환경 변수 설정
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "smart_factory_lstm")
DATA_PATH = os.getenv("DATA_PATH", "/app/data/processed_data.csv")
MODEL_PATH = os.getenv("MODEL_PATH", "/app/models/lstm_model.pth")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
SEQUENCE_LENGTH = int(os.getenv("SEQUENCE_LENGTH", "24"))
HIDDEN_SIZE = int(os.getenv("HIDDEN_SIZE", "64"))
NUM_LAYERS = int(os.getenv("NUM_LAYERS", "2"))

# LSTM 모델 클래스 정의
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

# 결과를 그래프로 시각화하는 함수
def plot_results(y_test, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='실제값')
    plt.plot(y_pred, label='예측값')
    plt.title(title)
    plt.xlabel('시간')
    plt.ylabel('값')
    plt.legend()
    plt.grid(True)
    
    # 임시 파일로 저장
    temp_path = f"/tmp/{title.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
    plt.savefig(temp_path)
    plt.close()
    
    return temp_path

def main():
    # MLflow 설정
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print(f"MLflow 추적 서버: {MLFLOW_TRACKING_URI}")
    print(f"실험 이름: {EXPERIMENT_NAME}")
    
    # 데이터 로드
    print(f"데이터 로드 중: {DATA_PATH}")
    try:
        df = pd.read_csv(DATA_PATH)
        print(f"데이터 로드 완료: {df.shape}")
    except Exception as e:
        print(f"데이터 로드 오류: {str(e)}")
        return
    
    # 특성 및 타겟 컬럼 분리
    feature_cols = df.columns[1:-1]  # 첫 번째 열(타임스탬프)과 마지막 열(타겟) 제외
    target_col = df.columns[-1]
    
    print(f"특성 컬럼: {len(feature_cols)} 개")
    print(f"타겟 컬럼: {target_col}")
    
    # 특성과 타겟 분리
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # 데이터 정규화
    X_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    X_scaled = X_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)
    
    # 시퀀스 데이터 생성
    X_sequences, y_sequences = [], []
    for i in range(len(X_scaled) - SEQUENCE_LENGTH):
        X_sequences.append(X_scaled[i:i + SEQUENCE_LENGTH])
        y_sequences.append(y_scaled[i + SEQUENCE_LENGTH])
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.float32)
    
    print(f"시퀀스 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}")
    
    # 학습/검증/테스트 데이터 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_sequences, y_sequences, test_size=0.2, shuffle=False
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.2, shuffle=False
    )
    
    print(f"학습 데이터: {X_train.shape}")
    print(f"검증 데이터: {X_val.shape}")
    print(f"테스트 데이터: {X_test.shape}")
    
    # PyTorch 텐서로 변환
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
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
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 초기화
    input_size = X_train.shape[2]  # 특성 수
    output_size = 1
    
    model = LSTMModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=output_size
    ).to(device)
    
    print(f"모델 생성: 입력 크기={input_size}, 은닉층 크기={HIDDEN_SIZE}, 레이어 수={NUM_LAYERS}")
    
    # MLflow 실행 시작
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"MLflow 실행 ID: {run_id}")
        
        # 하이퍼파라미터 로깅
        mlflow.log_params({
            "sequence_length": SEQUENCE_LENGTH,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "input_size": input_size,
            "output_size": output_size,
            "optimizer": "Adam",
            "loss_function": "MSELoss"
        })
        
        # 손실 함수와 최적화 도구 정의
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # 조기 종료 설정
        best_val_loss = float('inf')
        patience = 10
        counter = 0
        
        # 학습 루프
        train_losses = []
        val_losses = []
        
        for epoch in range(EPOCHS):
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
            train_losses.append(train_loss)
            
            # 검증 모드
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            val_losses.append(val_loss)
            
            # 에폭 결과 로깅
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)
            
            print(f"에폭 {epoch+1}/{EPOCHS}, 학습 손실: {train_loss:.6f}, 검증 손실: {val_loss:.6f}")
            
            # 조기 종료 체크
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                counter = 0
                # 최상의 모델 저장
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"모델 저장됨: {MODEL_PATH}")
            else:
                counter += 1
                if counter >= patience:
                    print(f"조기 종료 (에폭 {epoch+1})")
                    break
        
        # 손실 그래프 저장 및 로깅
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='학습 손실')
        plt.plot(val_losses, label='검증 손실')
        plt.title('에폭별 손실')
        plt.xlabel('에폭')
        plt.ylabel('손실')
        plt.legend()
        plt.grid(True)
        
        loss_plot_path = f"/tmp/loss_plot_{run_id}.png"
        plt.savefig(loss_plot_path)
        plt.close()
        
        mlflow.log_artifact(loss_plot_path)
        
        # 최적의 모델 로드
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # 테스트 데이터로 모델 평가
        model.eval()
        test_loss = 0.0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                
                # 예측값과 실제값 저장
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(targets.cpu().numpy())
        
        test_loss = test_loss / len(test_loader)
        print(f"테스트 손실: {test_loss:.6f}")
        
        # 테스트 손실 로깅
        mlflow.log_metric("test_loss", test_loss)
        
        # 예측값과 실제값 역정규화
        predictions = y_scaler.inverse_transform(np.array(predictions))
        actuals = y_scaler.inverse_transform(np.array(actuals))
        
        # 성능 지표 계산
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mae = np.mean(np.abs(predictions - actuals))
        
        print(f"RMSE: {rmse:.6f}")
        print(f"MAE: {mae:.6f}")
        
        # 성능 지표 로깅
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae
        })
        
        # 예측 결과 시각화 및 로깅
        pred_plot_path = plot_results(actuals, predictions, "테스트 데이터 예측 결과")
        mlflow.log_artifact(pred_plot_path)
        
        # 스케일러 저장
        np.save("/app/models/X_scaler.npy", X_scaler)
        np.save("/app/models/y_scaler.npy", y_scaler)
        
        # 메타데이터 저장
        model_info = {
            "input_size": input_size,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "output_size": output_size,
            "sequence_length": SEQUENCE_LENGTH,
            "feature_cols": feature_cols.tolist(),
            "target_col": target_col,
            "test_loss": test_loss,
            "rmse": rmse,
            "mae": mae,
            "train_samples": X_train.shape[0],
            "created_at": datetime.now().isoformat()
        }
        
        import json
        with open("/app/models/model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
        
        # 모델 등록
        mlflow.pytorch.log_model(model, "lstm_model")
        
        print(f"학습 완료! MLflow 실행 ID: {run_id}")
        print(f"모델 저장 경로: {MODEL_PATH}")

if __name__ == "__main__":
    main()