#!/usr/bin/env python
"""
다중 센서 데이터를 활용한 이상 상태 분류 모델
- 각 센서(1,2,3,4)의 데이터를 시간에 맞춰 보간
- 보간된 다중 센서 데이터를 이용해 상태(normal, type1, type2, type3) 분류 수행
"""
import os
import glob
import logging
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from scipy.interpolate import interp1d
import json
import traceback

# 로깅 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 상태 매핑 (전역 정의)
STATE_MAPPING = {
    "normal": 0,
    "type1": 1,
    "type2": 2,
    "type3": 3
}
INVERSE_STATE_MAPPING = {v: k for k, v in STATE_MAPPING.items()}

# 필요한 디렉토리 생성
def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성"""
    os.makedirs(directory, exist_ok=True)
    return directory

class SensorDataProcessor:
    """
    다중 센서 데이터 처리를 위한 클래스
    - 센서 데이터 로드
    - 상태 결정
    - 시간 기준 보간
    - 센서 데이터 결합
    """
    
    def __init__(self, interpolation_step=0.001, window_size=15):
        """
        Args:
            interpolation_step: 보간에 사용할 시간 간격 (초 단위)
            window_size: 이동평균 윈도우 크기
        """
        self.interpolation_step = interpolation_step
        self.window_size = window_size
        self.scaler = MinMaxScaler()
        
        # 상태 매핑은 전역 변수 사용
        self.state_mapping = STATE_MAPPING
        self.inverse_state_mapping = INVERSE_STATE_MAPPING
    
    def load_and_interpolate_sensor_data(self, data_dir, prefix="g1"):
        """
        센서 데이터를 로드하고 시간 기준으로 보간
        
        Args:
            data_dir: 데이터 디렉토리
            prefix: 파일 접두사
            
        Returns:
            dict: 센서 ID를 키로 갖는 보간된 데이터프레임 딕셔너리
        """
        sensor_data = {}
        
        # 센서 번호별 파일 목록
        sensor_files = [
            f"{prefix}_sensor1.csv",
            f"{prefix}_sensor2.csv",
            f"{prefix}_sensor3.csv",
            f"{prefix}_sensor4.csv"
        ]
        
        for i, sensor_file in enumerate(sensor_files, start=1):
            file_path = os.path.join(data_dir, sensor_file)
            
            if not os.path.exists(file_path):
                logger.warning(f"센서 {i}의 파일이 없습니다: {file_path}")
                continue
            
            # 데이터 로드
            df = pd.read_csv(file_path, names=['time', 'normal', 'type1', 'type2', 'type3'])
            
            # 선형 보간
            x_new = np.arange(0, 140, self.interpolation_step)
            interpolated_data = []
            
            for state in ['normal', 'type1', 'type2', 'type3']:
                f_linear = interp1d(df['time'], df[state], kind='linear')
                interpolated_data.append(f_linear(x_new))
            
            # 보간된 데이터를 데이터프레임으로 변환
            sensor_data[f'sensor{i}'] = pd.DataFrame(
                np.array(interpolated_data).T,
                columns=['normal', 'type1', 'type2', 'type3']
            )
        
        return sensor_data
    
    def combine_and_preprocess_sensor_data(self, interpolated_data):
        """
        보간된 센서 데이터를 상태별로 결합하고 전처리
        
        Args:
            interpolated_data: 보간된 센서 데이터 딕셔너리
            
        Returns:
            tuple: 상태별 전처리된 데이터 (normal, type1, type2, type3)
        """
        # 상태별로 센서 데이터 결합
        combined_data = {}
        for state in ['normal', 'type1', 'type2', 'type3']:
            combined_data[state] = pd.concat([
                interpolated_data[f'sensor{i}'][state] for i in range(1, 5)
            ], axis=1)
            combined_data[state].columns = ['s1', 's2', 's3', 's4']
        
        # 이동평균 필터 적용
        filtered_data = {}
        for state in ['normal', 'type1', 'type2', 'type3']:
            filtered_columns = []
            for col in combined_data[state].columns:
                filtered_signal = np.convolve(combined_data[state][col], np.ones(self.window_size), 'valid') / self.window_size
                filtered_columns.append(filtered_signal.reshape(-1, 1))
            filtered_data[state] = np.concatenate(filtered_columns, axis=1)
        
        # MinMaxScaler를 사용하여 정규화
        self.scaler.fit(combined_data['normal'])
        
        processed_data = {}
        for state in ['normal', 'type1', 'type2', 'type3']:
            processed_data[state] = self.scaler.transform(filtered_data[state])
        
        return processed_data
    
    def split_and_combine_data(self, processed_data, train_ratio=0.6, valid_ratio=0.2, test_ratio=0.2):
        """
        전처리된 데이터를 학습, 검증, 테스트 세트로 분할하고 결합
        
        Args:
            processed_data: 전처리된 상태별 데이터 딕셔너리
            train_ratio: 학습 데이터 비율
            valid_ratio: 검증 데이터 비율
            test_ratio: 테스트 데이터 비율
            
        Returns:
            tuple: (학습 데이터, 검증 데이터, 테스트 데이터)
        """
        train_data = {}
        valid_data = {}
        test_data = {}
        
        # 상태별로 데이터 분할
        for state in ['normal', 'type1', 'type2', 'type3']:
            data_size = len(processed_data[state])
            train_size = int(data_size * train_ratio)
            valid_size = int(data_size * valid_ratio)
            
            train_data[state] = processed_data[state][:train_size]
            valid_data[state] = processed_data[state][train_size:train_size+valid_size]
            test_data[state] = processed_data[state][train_size+valid_size:]
        
        # 분할된 데이터 결합
        train_combined = np.concatenate([train_data[state] for state in ['normal', 'type1', 'type2', 'type3']])
        valid_combined = np.concatenate([valid_data[state] for state in ['normal', 'type1', 'type2', 'type3']])
        test_combined = np.concatenate([test_data[state] for state in ['normal', 'type1', 'type2', 'type3']])
        
        return train_combined, valid_combined, test_combined

class MultiSensorLSTMClassifier(nn.Module):
    """
    다중 센서 LSTM 분류 모델
    """
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.3):
        super(MultiSensorLSTMClassifier, self).__init__()
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
        
        # 어텐션 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        """
        순방향 전파
        
        Args:
            x: 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
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

def prepare_sequence_data(data, sequence_length=50):
    """
    결합된 센서 데이터를 시퀀스 데이터로 변환
    
    Args:
        data: 결합된 센서 데이터 (학습/검증/테스트)
        sequence_length: 시퀀스 길이
        
    Returns:
        tuple: (시퀀스 데이터, 레이블 데이터)
    """
    X_sequences = []
    y_sequences = []
    
    # 데이터 길이 확인
    data_size = len(data)
    num_features = data.shape[1]
    
    # 시퀀스 생성
    for i in range(data_size - sequence_length):
        # 시퀀스 추출
        sequence = data[i:i+sequence_length]
        X_sequences.append(sequence)
        
        # 레이블 생성 (데이터의 1/4씩 나누어 각 상태로 매핑)
        state_idx = (i // (data_size // 4)) % 4  # 0, 1, 2, 3 순서로 반복
        y_sequences.append(state_idx)
    
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.int64)

    return X_sequences, y_sequences

def train_model(train_data, valid_data, args):
    """
    모델 학습
    
    Args:
        train_data: 학습 데이터
        valid_data: 검증 데이터
        args: 학습 관련 인자
        
    Returns:
        tuple: (학습된 모델, 학습 이력)
    """
    # PyTorch 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"학습에 사용할 장치: {device}")
    
    # 학습 데이터 준비
    X_train, y_train = prepare_sequence_data(train_data, sequence_length=args.sequence_length)
    X_valid, y_valid = prepare_sequence_data(valid_data, sequence_length=args.sequence_length)
    # 데이터 크기 및 분포 출력
    logger.info(f"학습 데이터 크기: X_train={X_train.shape}, y_train={y_train.shape}")
    logger.info(f"검증 데이터 크기: X_valid={X_valid.shape}, y_valid={y_valid.shape}")
    logger.info(f"학습 데이터 레이블 분포: {np.bincount(y_train)}")
    logger.info(f"검증 데이터 레이블 분포: {np.bincount(y_valid)}")

    # 데이터 로더 생성
    train_dataset = TensorDataset(
        torch.from_numpy(X_train[:50000]).to(device), 
        torch.from_numpy(y_train[:50000]).to(device)
    )
    valid_dataset = TensorDataset(
        torch.from_numpy(X_valid).to(device), 
        torch.from_numpy(y_valid).to(device)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=4, shuffle=False)
    
    # 모델 초기화
    input_size = X_train.shape[2]
    num_classes = len(np.unique(y_train))
    
    model = MultiSensorLSTMClassifier(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes = num_classes,
        dropout_rate=0.3
    ).to(device)
    
    # 손실 함수 및 옵티마이저
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 루프
    num_epochs = args.epochs
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10
    
    logger.info(f"모델 학습 시작: 에폭 {num_epochs}, 은닉층 크기 {args.hidden_size}, 레이어 수 {args.num_layers}")
    
    for epoch in range(num_epochs):
        logger.info("1")
        # 학습
        model.train()
        train_loss = 0.0
        logger.info("2")
        for inputs, labels in train_loader:
            logger.info("3")
            optimizer.zero_grad()
            logger.info("4")
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 검증
        model.eval()
        valid_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in valid_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
        
        valid_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)
        
        # 스케줄러 업데이트
        scheduler.step(valid_loss)
        
        # 진행 상황 출력
        logger.info(f"에폭 [{epoch+1}/{num_epochs}], 학습 손실: {train_loss:.4f}, 검증 손실: {valid_loss:.4f}")
        
        # 조기 종료 확인
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
            # 최적 모델 저장
            best_model_path = os.path.join(args.model_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"새로운 최적 모델 저장: {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"조기 종료 (에폭 {epoch+1})")
                break
    
    # 최적 모델 로드
    best_model_path = os.path.join(args.model_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        logger.info(f"최적 모델 로드: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
    else:
        logger.warning(f"최적 모델 파일을 찾을 수 없습니다: {best_model_path}")
    return model, (train_losses, valid_losses)

def evaluate_model(model, test_data, args):
    """
    모델 평가
    
    Args:
        model: 학습된 모델
        test_data: 테스트 데이터
        args: 평가 관련 인자
        
    Returns:
        dict: 평가 결과 (정확도, 혼동 행렬 등)
    """
    # PyTorch 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 테스트 데이터 준비
    X_test, y_test = prepare_sequence_data(test_data, sequence_length=args.sequence_length)
    
    # 데이터 로더 생성
    test_dataset = TensorDataset(
        torch.from_numpy(X_test).to(device), 
        torch.from_numpy(y_test).to(device)
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 평가
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

    # 혼동 행렬 시각화 및 저장
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    confusion_matrix_path = os.path.join(args.plot_dir, 'confusion_matrix.png')
    plt.savefig(confusion_matrix_path)
    logger.info(f"혼동 행렬 시각화 저장: {confusion_matrix_path}")
    plt.close()

    # JSON으로 직렬화 가능하도록 NumPy 배열을 리스트로 변환
    cm_list = cm.tolist()

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm_list,
        'classification_report': report
    }

def preprocess_sensor_data(args):
    """
    센서 데이터 전처리
    
    Args:
        args: 명령줄 인자
        
    Returns:
        tuple: (학습 데이터, 검증 데이터, 테스트 데이터)
    """
    logger.info("센서 데이터 전처리 시작")
    
    # 데이터 디렉토리 확인
    if not os.path.isdir(args.data_dir):
        logger.error(f"데이터 디렉토리 {args.data_dir}이(가) 존재하지 않습니다.")
        return None, None, None

    try:
        # 데이터 처리기 초기화
        processor = SensorDataProcessor(interpolation_step=args.interp_step, window_size=15)
    
        # 센서 데이터 로드 및 보간
        logger.info("센서 데이터 로드 및 보간 중...")
        interpolated_data = processor.load_and_interpolate_sensor_data(args.data_dir)
    
        # 센서 데이터 결합 및 전처리
        logger.info("센서 데이터 결합 및 전처리 중...")
        processed_data = processor.combine_and_preprocess_sensor_data(interpolated_data)
    
        # 데이터 분할 및 결합
        logger.info("데이터 분할 및 결합 중...")
        train_data, valid_data, test_data = processor.split_and_combine_data(
            processed_data,
            train_ratio=0.6,
            valid_ratio=0.2,
            test_ratio=0.2
        )
    
        logger.info(f"학습 데이터 형태: {train_data.shape}")
        logger.info(f"검증 데이터 형태: {valid_data.shape}")
        logger.info(f" 테스트 데이터 형태: {test_data.shape}")
    
        return train_data, valid_data, test_data
    
    except Exception as e:
        logger.error(f"데이터 전처리 중 오류 발생: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='다중 센서 데이터를 이용한 상태 분류 모델')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--data_dir', type=str, default=os.path.join(base_dir, 'data', 'raw'), help='원시 데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default=os.path.join(base_dir, 'data', 'processed'), help='처리된 데이터 저장 디렉토리')
    parser.add_argument('--model_dir', type=str, default=os.path.join(base_dir, 'models'), help='모델 저장 디렉토리')
    parser.add_argument('--plot_dir', type=str, default=os.path.join(base_dir, 'plots'), help='결과 시각화 저장 디렉토리')

    # 나머지 인자들은 기존과 동일
    parser.add_argument('--sequence_length', type=int, default=15, help='시퀀스 길이')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--hidden_size', type=int, default=32, help='LSTM 은닉층 크기')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM 레이어 수')
    parser.add_argument('--save_data', action='store_true', help='처리된 데이터 저장 여부')
    parser.add_argument('--interp_step', type=float, default=0.001, help='보간 간격 (초 단위)')

    args = parser.parse_args()

    # 필요한 디렉토리 생성
    ensure_dir(args.data_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.plot_dir)

    # 데이터 전처리
    logger.info("===== 센서 데이터 전처리 =====")
    train_data, valid_data, test_data = preprocess_sensor_data(args)

    if train_data is None or valid_data is None or test_data is None:
        logger.error("데이터 전처리에 실패했습니다.")
        return

    # 처리된 데이터 저장 (선택적)
    if args.save_data:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        train_path = os.path.join(args.output_dir, f"train_data_{current_time}.npy")
        valid_path = os.path.join(args.output_dir, f"valid_data_{current_time}.npy")
        test_path = os.path.join(args.output_dir, f"test_data_{current_time}.npy")
    
        np.save(train_path, train_data)
        np.save(valid_path, valid_data)
        np.save(test_path, test_data)
    
        logger.info(f"처리된 데이터 저장 완료:")
        logger.info(f"- 학습 데이터: {train_path}")
        logger.info(f"- 검증 데이터: {valid_path}")
        logger.info(f"- 테스트 데이터: {test_path}")

    # 모델 학습
    logger.info("===== 모델 학습 =====")
    model, history = train_model(train_data, valid_data, args)

    # 학습 이력 시각화
    train_loss, valid_loss = history
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()

    history_plot_path = os.path.join(args.plot_dir, 'training_history.png')
    plt.savefig(history_plot_path)
    logger.info(f"학습 이력 그래프 저장: {history_plot_path}")
    plt.close()

    # 모델 평가
    logger.info("===== 모델 평가 =====")
    evaluation_result = evaluate_model(model, test_data, args)

    logger.info(f"테스트 데이터 평가 결과:")
    logger.info(f"- 정확도: {evaluation_result['accuracy']:.4f}")
    logger.info(f"- 혼동 행렬:\n{evaluation_result['confusion_matrix']}")
    
    # 모델 저장
    model_path = os.path.join(args.model_dir, f"sensor_classifier_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_path)
    logger.info(f"최종 모델 저장 완료: {model_path}")

    # 모델 아키텍처 정보 저장
    model_info = {
        "model_type": "MultiSensorLSTMClassifier",
        "input_size": model.lstm.input_size,
        "hidden_size": model.hidden_size,
        "num_layers": model.num_layers,
        "num_classes": model.fc.out_features,
        "sequence_length": args.sequence_length,
        "created_at": datetime.now().isoformat()
    }
    
    model_info_path = os.path.join(args.model_dir, 'model_info.json')
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    
    # 평가 결과 저장
    evaluation_path = os.path.join(args.output_dir, 'evaluation_result.json')
    with open(evaluation_path, 'w') as f:
        json.dump(evaluation_result, f, indent=4)

    logger.info(f"평가 결과 저장 완료: {evaluation_path}")

if __name__ == "__main__":
    main()