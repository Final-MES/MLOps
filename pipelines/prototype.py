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
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.interpolate import interp1d
import json

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    
    def __init__(self, interpolation_step=0.001):
        """
        Args:
            interpolation_step: 보간에 사용할 시간 간격 (초 단위)
        """
        self.interpolation_step = interpolation_step
        # 상태 매핑
        self.state_mapping = {
            "normal": 0,
            "type1": 1,
            "type2": 2,
            "type3": 3
        }
        self.inverse_state_mapping = {v: k for k, v in self.state_mapping.items()}
    
    def load_sensor_files(self, data_dir, prefix="g1"):
        """
        센서 파일 로드 및 상태 결정
        
        Args:
            data_dir: 데이터 디렉토리
            prefix: 파일 접두사
            
        Returns:
            dict: 센서 ID를 키로 갖는 데이터프레임 딕셔너리
        """
        sensor_data = {}
        
        # 센서 번호별 파일 검색 패턴
        patterns = [
            f"{prefix}*sensor1*.csv",
            f"{prefix}*sensor2*.csv",
            f"{prefix}*sensor3*.csv",
            f"{prefix}*sensor4*.csv"
        ]
        
        for i, pattern in enumerate(patterns):
            sensor_id = i + 1
            files = glob.glob(os.path.join(data_dir, pattern))
            
            if not files:
                logger.warning(f"센서 {sensor_id}에 해당하는 파일이 없습니다: {pattern}")
                continue
                
            # 해당 센서의 모든 파일 데이터 통합
            all_sensor_data = []
            for file in files:
                try:
                    # 컬럼명 지정하여 로드
                    df = pd.read_csv(file, names=["time", "normal", "type1", "type2", "type3"])
                    
                    # 시간 컬럼이 숫자가 아니면 변환 시도
                    if not pd.api.types.is_numeric_dtype(df["time"]):
                        try:
                            df["time"] = pd.to_datetime(df["time"]).astype(np.int64) // 10**9  # 초 단위 타임스탬프로 변환
                        except:
                            logger.error(f"파일 {file}의 시간 컬럼을 숫자로 변환할 수 없습니다.")
                            continue
                    
                    # 각 행에서 가장 큰 값을 가진 컬럼을 상태로 결정
                    df['state'] = df[["normal", "type1", "type2", "type3"]].idxmax(axis=1)
                    
                    # 상태를 숫자로 인코딩
                    df['state_encoded'] = df['state'].map(self.state_mapping)
                    
                    all_sensor_data.append(df)
                    logger.info(f"파일 로드 완료: {file} ({len(df)} 행)")
                    
                except Exception as e:
                    logger.error(f"파일 {file} 로드 중 오류 발생: {str(e)}")
            
            if all_sensor_data:
                # 데이터 병합
                sensor_data[f"sensor{sensor_id}"] = pd.concat(all_sensor_data, ignore_index=True)
                
                # 시간 순으로 정렬
                sensor_data[f"sensor{sensor_id}"] = sensor_data[f"sensor{sensor_id}"].sort_values("time")
                
                # 상태 분포 확인
                state_counts = sensor_data[f"sensor{sensor_id}"]['state'].value_counts()
                logger.info(f"센서 {sensor_id} 상태 분포: {state_counts.to_dict()}")
                
                logger.info(f"센서 {sensor_id} 데이터 로드 완료: {len(sensor_data[f'sensor{sensor_id}'])} 행")
        
        return sensor_data
    
    def interpolate_sensor_data(self, sensor_data):
        """
        모든 센서 데이터를 균일한 시간 간격으로 보간
        
        Args:
            sensor_data: 센서 ID를 키로 갖는 데이터프레임 딕셔너리
            
        Returns:
            dict: 보간된 센서 데이터
        """
        total_states = {
        'normal': 0, 'type1': 0, 'type2': 0, 'type3': 0
    }
        sensor_count = len(sensor_data)
    
        for sensor_id, df in sensor_data.items():
            state_counts = df[['normal', 'type1', 'type2', 'type3']].sum()
        for state, count in state_counts.items():
            total_states[state] += count
    
    # 평균 상태 분포 계산
        for state in total_states:
            total_states[state] /= sensor_count
    
        logger.info("통합 상태 분포:")
        logger.info(total_states)
        if not sensor_data:
            logger.error("보간할 센서 데이터가 없습니다.")
            return {}
        
        # 모든 센서의 시간 범위 결정
        min_time = float('inf')
        max_time = float('-inf')
        
        for sensor_id, df in sensor_data.items():
            if len(df) > 0:
                min_time = min(min_time, df["time"].min())
                max_time = max(max_time, df["time"].max())
        
        if min_time == float('inf') or max_time == float('-inf'):
            logger.error("유효한 시간 범위를 결정할 수 없습니다.")
            return {}
        
        # 균일한 시간 간격 생성
        uniform_time = np.arange(min_time, max_time + self.interpolation_step, self.interpolation_step)
        
        # 각 센서 데이터 보간
        interpolated_data = {}
        
        for sensor_id, df in sensor_data.items():
            if len(df) < 2:  # 보간에는 최소 2개 포인트 필요
                logger.warning(f"센서 {sensor_id}의 데이터가 너무 적어 보간을 건너뜁니다.")
                continue
            
            # 센서별 데이터프레임 생성 (시간 컬럼 포함)
            interp_df = pd.DataFrame({"time": uniform_time})
            
            # 원본 측정값 보간 (normal, type1, type2, type3)
            for column in ["normal", "type1", "type2", "type3"]:
                # 시간과 특성 추출
                times = df["time"].values
                values = df[column].values
                
                # 중복된 시간 제거 (보간 함수를 위해)
                unique_times, unique_indices = np.unique(times, return_index=True)
                unique_values = values[unique_indices]
                
                if len(unique_times) < 2:
                    logger.warning(f"센서 {sensor_id}의 {column} 데이터에 중복 제거 후 포인트가 부족합니다.")
                    interp_df[column] = np.nan
                    continue
                
                try:
                    # 선형 보간 함수 생성
                    f = interp1d(
                        unique_times, 
                        unique_values, 
                        kind='linear', 
                        bounds_error=False, 
                        fill_value=(unique_values[0], unique_values[-1])  # 범위 외 값은 끝단 값으로 채움
                    )
                    
                    # 보간 적용
                    interp_df[column] = f(uniform_time)
                except Exception as e:
                    logger.error(f"센서 {sensor_id}의 {column} 보간 중 오류: {str(e)}")
                    interp_df[column] = np.nan
            
            # 결측치 처리 (전방 채우기 후 후방 채우기)
            interp_df = interp_df.fillna(method='ffill').fillna(method='bfill')
            
            # 보간된 값으로 상태 다시 결정
            interp_df['state'] = interp_df[["normal", "type1", "type2", "type3"]].idxmax(axis=1)
            interp_df['state_encoded'] = interp_df['state'].map(self.state_mapping)
            
            # 보간된 데이터 저장
            interpolated_data[sensor_id] = interp_df
            
            logger.info(f"센서 {sensor_id} 데이터 보간 완료: {len(interp_df)} 행")
            
            # 보간 후 상태 분포 확인
            state_counts = interp_df['state'].value_counts()
            logger.info(f"센서 {sensor_id} 보간 후 상태 분포: {state_counts.to_dict()}")
        
        return interpolated_data
    
    def combine_sensor_data(self, interpolated_data):
        """
        보간된 센서 데이터를 단일 데이터프레임으로 결합
        
        Args:
            interpolated_data: 보간된 센서 데이터 딕셔너리
            
        Returns:
            pd.DataFrame: 결합된 데이터프레임
        """
        if not interpolated_data:
            logger.error("결합할 센서 데이터가 없습니다.")
            return None
        
        # 시간 컬럼을 기준으로 병합 시작
        combined_df = None
        
        for sensor_id, df in interpolated_data.items():
            # 센서 측정값 컬럼에 센서 ID 접두사 추가
            df_renamed = df.copy()
            measurement_cols = ["normal", "type1", "type2", "type3"]
            for col in measurement_cols:
                df_renamed = df_renamed.rename(columns={col: f"{sensor_id}_{col}"})
            
            # 필요한 컬럼만 유지 (시간, 센서 측정값)
            columns_to_keep = ["time"] + [f"{sensor_id}_{col}" for col in measurement_cols] + ["state", "state_encoded"]
            df_renamed = df_renamed[columns_to_keep]
            
            if combined_df is None:
                combined_df = df_renamed
            else:
                # 시간을 기준으로 병합
                # 내부 조인 사용 - 모든 센서에 공통된 시간만 유지
                combined_df = pd.merge(
                    combined_df, 
                    df_renamed, 
                    on="time", 
                    how="inner",
                    suffixes=('', f'_{sensor_id}')
                )
        
        if combined_df is None or len(combined_df) == 0:
            logger.error("센서 데이터 결합 후 유효한 데이터가 없습니다.")
            return None
        
        # 원본 상태와 상태 인코딩 컬럼이 중복되었을 경우 처리
        state_columns = [col for col in combined_df.columns if col.endswith('_state') or col == 'state']
        encoded_columns = [col for col in combined_df.columns if col.endswith('_state_encoded') or col == 'state_encoded']
        
        # 첫 번째 상태 컬럼과 상태 인코딩 컬럼만 유지하고 나머지는 삭제
        if state_columns:
            first_state_col = state_columns[0]
            columns_to_drop = [col for col in state_columns if col != first_state_col]
            combined_df = combined_df.drop(columns=columns_to_drop)
            combined_df = combined_df.rename(columns={first_state_col: 'state'})
        
        if encoded_columns:
            first_encoded_col = encoded_columns[0]
            columns_to_drop = [col for col in encoded_columns if col != first_encoded_col]
            combined_df = combined_df.drop(columns=columns_to_drop)
            combined_df = combined_df.rename(columns={first_encoded_col: 'state_encoded'})
        
        logger.info(f"센서 데이터 결합 완료: {len(combined_df)} 행, {len(combined_df.columns)} 열")
        
        # 결합된 데이터의 상태 분포 확인
        state_counts = combined_df['state'].value_counts()
        logger.info(f"결합된 데이터 상태 분포: {state_counts.to_dict()}")
        
        return combined_df

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

def prepare_sequence_data(combined_df, sequence_length=50, test_size=0.2, val_size=0.2):
    # 각 센서의 측정값 컬럼 정의
    sensor_cols = {
        'sensor1': ['sensor1_normal', 'sensor1_type1', 'sensor1_type2', 'sensor1_type3'],
        'sensor2': ['sensor2_normal', 'sensor2_type1', 'sensor2_type2', 'sensor2_type3'],
        'sensor3': ['sensor3_normal', 'sensor3_type1', 'sensor3_type2', 'sensor3_type3'],
        'sensor4': ['sensor4_normal', 'sensor4_type1', 'sensor4_type2', 'sensor4_type3']
    }

    # 상태 컬럼 선택
    state_col = 'state_encoded' if 'state_encoded' in combined_df.columns else 'state'

    # 데이터 타입 변환 및 NaN 처리
    for sensor_measurements in sensor_cols.values():
        combined_df[sensor_measurements] = combined_df[sensor_measurements].apply(pd.to_numeric, errors='coerce').fillna(0)

    # 시퀀스 데이터 생성
    X_sequences = []
    y_sequences = []

    for i in range(len(combined_df) - sequence_length + 1):
        # 각 센서의 시퀀스 데이터 추출
        sensor_sequences = {}
        for sensor, cols in sensor_cols.items():
            # 해당 센서의 시퀀스 데이터
            sensor_sequence = combined_df[cols].iloc[i:i+sequence_length].values
            sensor_sequences[sensor] = sensor_sequence

        # 각 센서의 마지막 시점의 측정값을 하나의 입력으로 결합
        X_sequence = np.concatenate([
            sensor_sequences['sensor1'][-1],
            sensor_sequences['sensor2'][-1],
            sensor_sequences['sensor3'][-1],
            sensor_sequences['sensor4'][-1]
        ])

        # 상태 레이블 (원-핫 인코딩)
        state_label = int(combined_df[state_col].iloc[i + sequence_length - 1])
        
        X_sequences.append(X_sequence)
        y_sequences.append(state_label)

    # NumPy 배열로 변환
    X_sequences = np.array(X_sequences, dtype=np.float32)
    y_sequences = np.array(y_sequences, dtype=np.int64)

    logger.info(f"시퀀스 데이터 형태: X={X_sequences.shape}, y={y_sequences.shape}")

    # 클래스 분포 확인
    unique_classes, class_counts = np.unique(y_sequences, return_counts=True)
    class_distribution = dict(zip(unique_classes, class_counts))
    logger.info(f"시퀀스 데이터 클래스 분포: {class_distribution}")

    # 데이터 스케일링
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_sequences)

    # 학습/검증/테스트 데이터 분할
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_scaled, y_sequences, test_size=test_size, stratify=y_sequences
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, stratify=y_train_val
    )

    # 클래스 가중치 계산
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights)

    # 데이터 로더 생성
    device = torch.device('cpu')

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # 데이터 로더 생성
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 상태 매핑 정보
    state_mapping = {
        "normal": 0,
        "type1": 1,
        "type2": 2,
        "type3": 3
    }
    inverse_state_mapping = {v: k for k, v in state_mapping.items()}

    # 데이터 정보 저장
    data_info = {
        'feature_cols': [col for sensor_cols_list in sensor_cols.values() for col in sensor_cols_list],
        'sequence_length': sequence_length,
        'input_size': X_train.shape[1],
        'num_classes': len(np.unique(y_sequences)),
        'class_mapping': state_mapping,
        'inverse_class_mapping': inverse_state_mapping,
        'class_weights': class_weights.tolist(),
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'device': device.type,
        'scaler': scaler
    }

    return train_loader, val_loader, test_loader, data_info

def evaluate_model(model, test_loader, data_info, plot_dir="plots"):
    """
    모델 평가
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        data_info: 데이터 정보
        plot_dir: 결과 시각화 저장 디렉토리
        
    Returns:
        dict: 평가 결과
    """
    model.eval()
    device = torch.device(data_info['device'])
    
    # 결과 저장을 위한 리스트
    all_preds = []
    all_targets = []
    
    # 클래스 매핑 정보
    inverse_class_mapping = data_info['inverse_class_mapping']
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            # CPU로 이동하여 NumPy 배열로 변환
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # NumPy 배열로 변환
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 정확도 계산
    accuracy = np.mean(all_preds == all_targets)
    
    # 혼동 행렬 계산
    from sklearn.metrics import confusion_matrix, classification_report
    conf_matrix = confusion_matrix(all_targets, all_preds)
    
    # 클래스 레이블 변환 (숫자 -> 원래 레이블)
    class_labels = [inverse_class_mapping[i] for i in range(len(inverse_class_mapping)) if i in inverse_class_mapping]
    
    # 분류 보고서
    report = classification_report(all_targets, all_preds, target_names=class_labels, output_dict=True)
    
    logger.info(f"테스트 정확도: {accuracy:.4f}")
    logger.info("분류 보고서:")
    for label in class_labels:
        if label in report:
            precision = report[label]['precision']
            recall = report[label]['recall']
            f1 = report[label]['f1-score']
            logger.info(f"{label}: 정밀도={precision:.4f}, 재현율={recall:.4f}, F1={f1:.4f}")
    
    # 결과 시각화
    ensure_dir(plot_dir)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(10, 8))
    import seaborn as sns
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title('상태 분류 혼동 행렬')
    plt.ylabel('실제 상태')
    plt.xlabel('예측 상태')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 클래스별 성능 시각화
    plt.figure(figsize=(12, 6))
    metrics = {
        'Precision': [report[label]['precision'] for label in class_labels if label in report],
        'Recall': [report[label]['recall'] for label in class_labels if label in report],
        'F1-Score': [report[label]['f1-score'] for label in class_labels if label in report]
    }
    
    x = np.arange(len(class_labels))
    width = 0.2
    
    plt.bar(x - width, metrics['Precision'], width=width, label='Precision')
    plt.bar(x, metrics['Recall'], width=width, label='Recall')
    plt.bar(x + width, metrics['F1-Score'], width=width, label='F1-Score')
    
    plt.xlabel('상태 클래스')
    plt.ylabel('점수')
    plt.title('클래스별 성능 메트릭')
    plt.xticks(x, class_labels)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'class_performance.png'))
    plt.close()
    
    # 평가 결과 저장
    evaluation_result = {
        'accuracy': float(accuracy),
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'class_labels': class_labels
    }
    
    # 평가 결과 저장
    with open(os.path.join(plot_dir, 'evaluation_result.json'), 'w') as f:
        json.dump(evaluation_result, f, indent=2)
    
    logger.info(f"평가 결과 저장: {os.path.join(plot_dir, 'evaluation_result.json')}")
    
    return evaluation_result

def analyze_feature_importance(model, test_loader, data_info, plot_dir="plots"):
    """
    특성 중요도 분석
    
    Args:
        model: 평가할 모델
        test_loader: 테스트 데이터 로더
        data_info: 데이터 정보
        plot_dir: 결과 시각화 저장 디렉토리
        
    Returns:
        dict: 특성 중요도 분석 결과
    """
    device = torch.device(data_info['device'])
    model.eval()
    
    # 기본 정확도 계산
    original_accuracy = 0
    total_samples = 0
    
    with torch.no_grad():
        all_targets = []
        all_preds = []
        
        for inputs, targets in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            total_samples += targets.size(0)
        
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)
        original_accuracy = np.mean(all_preds == all_targets)
    
    logger.info(f"원본 모델 정확도: {original_accuracy:.4f}")
    
    # 특성 중요도 계산 (순열 중요도 방법)
    feature_cols = data_info['feature_cols']
    importance_scores = {}
    
    # 특성 그룹 (센서별) 중요도 계산
    sensor_groups = {}
    for col in feature_cols:
        # 센서 ID 추출 (예: 'sensor1_normal' -> 'sensor1')
        sensor_id = col.split('_')[0]
        if sensor_id not in sensor_groups:
            sensor_groups[sensor_id] = []
        sensor_groups[sensor_id].append(col)
    
    logger.info("센서별 중요도 계산 중...")
    
    # 각 센서 그룹별 중요도 계산
    for sensor_id, sensor_features in sensor_groups.items():
        logger.info(f"{sensor_id} 중요도 계산 중...")
        
        # 특성 인덱스 가져오기
        feature_indices = []
        for feature in sensor_features:
            feature_idx = feature_cols.index(feature)
            feature_indices.append(feature_idx)
        
        # 순열 중요도 계산
        perturbed_accuracy = 0
        n_repeats = 3  # 여러 번 반복하여 안정성 향상
        
        for _ in range(n_repeats):
            with torch.no_grad():
                perturbed_correct = 0
                
                for inputs, targets in test_loader:
                    # 특성값 순열화
                    perturbed_inputs = inputs.clone()
                    perm_idx = torch.randperm(perturbed_inputs.size(0))
                    
                    # 해당 센서의 모든 특성에 대해 순열화 적용
                    for idx in feature_indices:
                        perturbed_inputs[:, :, idx] = perturbed_inputs[perm_idx, :, idx]
                    
                    outputs = model(perturbed_inputs)
                    _, preds = torch.max(outputs, 1)
                    perturbed_correct += (preds == targets).sum().item()
                
                # 순열화 후 정확도 계산
                iter_accuracy = perturbed_correct / total_samples
                perturbed_accuracy += iter_accuracy / n_repeats
        
        # 중요도 = 원본 정확도 - 순열화 후 정확도
        importance = original_accuracy - perturbed_accuracy
        importance_scores[sensor_id] = importance
        logger.info(f"{sensor_id} 중요도: {importance:.4f}")
    
    # 측정값 유형별(normal, type1, type2, type3) 중요도 계산
    measurement_groups = {}
    for col in feature_cols:
        # 측정값 유형 추출 (예: 'sensor1_normal' -> 'normal')
        measurement_type = col.split('_', 1)[1] if '_' in col else col
        if measurement_type not in measurement_groups:
            measurement_groups[measurement_type] = []
        measurement_groups[measurement_type].append(col)
    
    logger.info("측정값 유형별 중요도 계산 중...")
    
    # 각 측정값 유형별 중요도 계산
    for m_type, m_features in measurement_groups.items():
        logger.info(f"{m_type} 중요도 계산 중...")
        
        # 특성 인덱스 가져오기
        feature_indices = []
        for feature in m_features:
            feature_idx = feature_cols.index(feature)
            feature_indices.append(feature_idx)
        
        # 순열 중요도 계산
        perturbed_accuracy = 0
        n_repeats = 3  # 여러 번 반복하여 안정성 향상
        
        for _ in range(n_repeats):
            with torch.no_grad():
                perturbed_correct = 0
                
                for inputs, targets in test_loader:
                    # 특성값 순열화
                    perturbed_inputs = inputs.clone()
                    perm_idx = torch.randperm(perturbed_inputs.size(0))
                    
                    # 해당 측정값 유형의 모든 특성에 대해 순열화 적용
                    for idx in feature_indices:
                        perturbed_inputs[:, :, idx] = perturbed_inputs[perm_idx, :, idx]
                    
                    outputs = model(perturbed_inputs)
                    _, preds = torch.max(outputs, 1)
                    perturbed_correct += (preds == targets).sum().item()
                
                # 순열화 후 정확도 계산
                iter_accuracy = perturbed_correct / total_samples
                perturbed_accuracy += iter_accuracy / n_repeats
        
        # 중요도 = 원본 정확도 - 순열화 후 정확도
        importance = original_accuracy - perturbed_accuracy
        importance_scores[m_type] = importance
        logger.info(f"{m_type} 중요도: {importance:.4f}")
    
    # 결과 시각화
    ensure_dir(plot_dir)
    
    # 센서별 중요도 시각화
    plt.figure(figsize=(10, 6))
    sensor_ids = list(sensor_groups.keys())
    sensor_importance = [importance_scores[s_id] for s_id in sensor_ids]
    
    # 중요도에 따라 정렬
    sorted_idx = np.argsort(sensor_importance)
    sorted_sensor_ids = [sensor_ids[i] for i in sorted_idx]
    sorted_importance = [sensor_importance[i] for i in sorted_idx]
    
    plt.barh(sorted_sensor_ids, sorted_importance)
    plt.xlabel('중요도')
    plt.ylabel('센서')
    plt.title('센서별 중요도')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'sensor_importance.png'))
    plt.show()
    plt.close()
    
    # 측정값 유형별 중요도 시각화
    plt.figure(figsize=(10, 6))
    m_types = list(measurement_groups.keys())
    m_importance = [importance_scores[m_type] for m_type in m_types]
    
    # 중요도에 따라 정렬
    sorted_idx = np.argsort(m_importance)
    sorted_m_types = [m_types[i] for i in sorted_idx]
    sorted_importance = [m_importance[i] for i in sorted_idx]
    
    plt.barh(sorted_m_types, sorted_importance)
    plt.xlabel('중요도')
    plt.ylabel('측정값 유형')
    plt.title('측정값 유형별 중요도')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'measurement_importance.png'))
    plt.show()
    plt.close()
    
    # 특성 중요도 결과 저장
    importance_result = {
        'sensor_importance': {k: float(v) for k, v in importance_scores.items() if k in sensor_ids},
        'measurement_importance': {k: float(v) for k, v in importance_scores.items() if k in m_types}
    }
    
    # 결과 저장
    with open(os.path.join(plot_dir, 'feature_importance.json'), 'w') as f:
        json.dump(importance_result, f, indent=2)
    
    logger.info(f"특성 중요도 분석 결과 저장: {os.path.join(plot_dir, 'feature_importance.json')}")
    
    return importance_result

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='다중 센서 데이터를 이용한 상태 분류 모델')
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    parser.add_argument('--data_dir', type=str, default=os.path.join(base_dir, 'data', 'raw'), help='원시 데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default=os.path.join(base_dir, 'data', 'processed'), help='처리된 데이터 저장 디렉토리')
    parser.add_argument('--model_dir', type=str, default=os.path.join(base_dir, 'models'), help='모델 저장 디렉토리')
    parser.add_argument('--plot_dir', type=str, default=os.path.join(base_dir, 'plots'), help='결과 시각화 저장 디렉토리')
    
    # 나머지 인자들은 기존과 동일
    parser.add_argument('--sequence_length', type=int, default=50, help='시퀀스 길이')
    parser.add_argument('--epochs', type=int, default=100, help='학습 에폭 수')
    parser.add_argument('--hidden_size', type=int, default=128, help='LSTM 은닉층 크기')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'analyze'], default='train', help='실행 모드')
    parser.add_argument('--save_data', action='store_true', help='처리된 데이터 저장 여부')
    parser.add_argument('--interp_step', type=float, default=0.001, help='보간 간격 (초 단위)')
    
    args = parser.parse_args()
    
    # 필요한 디렉토리 생성
    ensure_dir(args.data_dir)
    ensure_dir(args.output_dir)
    ensure_dir(args.model_dir)
    ensure_dir(args.plot_dir)
    
    # 데이터 전처리 및 로드
    logger.info("===== 센서 데이터 로드 및 처리 =====")
    processor = SensorDataProcessor(interpolation_step=args.interp_step)
    
    # 데이터 로드
    sensor_data = processor.load_sensor_files(args.data_dir)
    
    # 데이터가 충분한지 확인
    if not sensor_data or len(sensor_data) == 0:
        logger.error("로드된 센서 데이터가 없습니다.")
        return
    
    # 데이터 보간
    logger.info("===== 센서 데이터 보간 =====")
    interpolated_data = processor.interpolate_sensor_data(sensor_data)
    
    # 데이터 결합
    logger.info("===== 센서 데이터 결합 =====")
    combined_df = processor.combine_sensor_data(interpolated_data)
    
    if combined_df is None or len(combined_df) == 0:
        logger.error("유효한 결합 데이터가 없습니다.")
        return
    
    # 처리된 데이터 저장 (선택적)
    if args.save_data:
        processed_data_path = os.path.join(args.output_dir, f"multi_sensor_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        combined_df.to_csv(processed_data_path, index=False)
        logger.info(f"처리된 데이터 저장 완료: {processed_data_path}")
    
    # 시퀀스 데이터 준비
    logger.info("===== 시퀀스 데이터 준비 =====")
    train_loader, val_loader, test_loader, data_info = prepare_sequence_data(
        combined_df,
        sequence_length=args.sequence_length
    )
    
    if args.mode == 'train':
        # 모델 학습
        logger.info("===== 모델 학습 시작 =====")
        model, history = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            data_info=data_info,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            model_dir=args.model_dir
        )
        
        # 학습 결과 시각화
        plt.figure(figsize=(12, 5))
        
        # 손실 그래프
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 정확도 그래프
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Accuracy History')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        history_plot_path = os.path.join(args.plot_dir, 'training_history.png')
        plt.savefig(history_plot_path)
        logger.info(f"학습 이력 시각화 저장: {history_plot_path}")
        plt.close()
        
        # 모델 평가
        logger.info("===== 모델 평가 시작 =====")
        evaluation_result = evaluate_model(
            model=model,
            test_loader=test_loader,
            data_info=data_info,
            plot_dir=args.plot_dir
        )
        
        # 특성 중요도 분석
        logger.info("===== 특성 중요도 분석 시작 =====")
        importance_result = analyze_feature_importance(
            model=model,
            test_loader=test_loader,
            data_info=data_info,
            plot_dir=args.plot_dir
        )
        
        logger.info("===== 모델 학습 및 평가 완료 =====")
        logger.info(f"최종 테스트 정확도: {evaluation_result['accuracy']:.4f}")
    
    elif args.mode == 'evaluate':
        # 저장된 모델 로드
        logger.info("===== 저장된 모델 평가 =====")
        model_path = os.path.join(args.model_dir, "multi_sensor_model.pth")
        model_info_path = os.path.join(args.model_dir, "multi_sensor_model_info.json")
        
        if not os.path.exists(model_path) or not os.path.exists(model_info_path):
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
            return
        
        # 모델 정보 로드
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            
        # 모델 초기화
        device = torch.device(data_info['device'])
        model = MultiSensorLSTMClassifier(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes']
        ).to(device)
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"모델 로드 완료: {model_path}")
        
        # 모델 평가
        evaluation_result = evaluate_model(
            model=model,
            test_loader=test_loader,
            data_info=data_info,
            plot_dir=args.plot_dir
        )
        
        logger.info("===== 모델 평가 완료 =====")
        logger.info(f"테스트 정확도: {evaluation_result['accuracy']:.4f}")
    
    elif args.mode == 'analyze':
        # 저장된 모델 로드
        logger.info("===== 저장된 모델 분석 =====")
        model_path = os.path.join(args.model_dir, "multi_sensor_model.pth")
        model_info_path = os.path.join(args.model_dir, "multi_sensor_model_info.json")
        
        if not os.path.exists(model_path) or not os.path.exists(model_info_path):
            logger.error(f"모델 파일이 존재하지 않습니다: {model_path}")
            return
        
        # 모델 정보 로드
        with open(model_info_path, 'r') as f:
            model_info = json.load(f)
            
        # 모델 초기화
        device = torch.device(data_info['device'])
        model = MultiSensorLSTMClassifier(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_layers=model_info['num_layers'],
            num_classes=model_info['num_classes']
        ).to(device)
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        logger.info(f"모델 로드 완료: {model_path}")
        
        # 특성 중요도 분석
        importance_result = analyze_feature_importance(
            model=model,
            test_loader=test_loader,
            data_info=data_info,
            plot_dir=args.plot_dir
        )
        
        logger.info("===== 모델 분석 완료 =====")

if __name__ == "__main__":
    main()