"""
센서 데이터 처리 모듈

이 모듈은 다중 센서 데이터의 로드, 보간, 전처리를 위한 기능을 제공합니다.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Any, Optional
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler

# 로깅 설정
logger = logging.getLogger(__name__)

# 상태 매핑 (전역 정의)
STATE_MAPPING = {
    "normal": 0,
    "type1": 1,
    "type2": 2,
    "type3": 3
}
INVERSE_STATE_MAPPING = {v: k for k, v in STATE_MAPPING.items()}

class SensorDataProcessor:
    """
    다중 센서 데이터 처리를 위한 클래스
    - 센서 데이터 로드
    - 상태 결정
    - 시간 기준 보간
    - 센서 데이터 결합
    """
    
    def __init__(self, interpolation_step: float = 0.001, window_size: int = 15):
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
        
        logger.info(f"센서 데이터 처리기 초기화: 보간 간격={interpolation_step}, 윈도우 크기={window_size}")
    
    def load_and_interpolate_sensor_data(self, data_dir: str, prefix: str = "g1") -> Dict[str, pd.DataFrame]:
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
            
            logger.info(f"센서 {i} 데이터 로드 및 보간 완료: {len(sensor_data[f'sensor{i}'])} 샘플")
        
        return sensor_data
    
    def combine_and_preprocess_sensor_data(self, interpolated_data: Dict[str, pd.DataFrame]) -> Dict[str, np.ndarray]:
        """
        보간된 센서 데이터를 상태별로 결합하고 전처리
        
        Args:
            interpolated_data: 보간된 센서 데이터 딕셔너리
            
        Returns:
            dict: 상태별 전처리된 데이터 (normal, type1, type2, type3)
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
                filtered_signal = np.convolve(
                    combined_data[state][col], 
                    np.ones(self.window_size), 
                    'valid'
                ) / self.window_size
                filtered_columns.append(filtered_signal.reshape(-1, 1))
            filtered_data[state] = np.concatenate(filtered_columns, axis=1)
        
        # 정규화를 위해 normal 데이터 기준으로 스케일러 학습
        self.scaler.fit(combined_data['normal'])
        
        # 각 상태별 데이터 정규화
        processed_data = {}
        for state in ['normal', 'type1', 'type2', 'type3']:
            processed_data[state] = self.scaler.transform(filtered_data[state])
            logger.info(f"{state} 데이터 전처리 완료: {processed_data[state].shape} 형태")
        
        return processed_data
    
    def split_and_combine_data(self, 
                              processed_data: Dict[str, np.ndarray], 
                              train_ratio: float = 0.6, 
                              valid_ratio: float = 0.2, 
                              test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        
        # 데이터 세트 크기 로깅
        logger.info(f"데이터 분할 완료:")
        logger.info(f"- 학습 데이터: {train_combined.shape}")
        logger.info(f"- 검증 데이터: {valid_combined.shape}")
        logger.info(f"- 테스트 데이터: {test_combined.shape}")
        
        return train_combined, valid_combined, test_combined
    def process_realtime_data(self, file_path: str, sequence_length: int = 100) -> np.ndarray:
        """
        실시간 센서 데이터를 모델 입력에 적합한 형태로 전처리합니다.
    
        이  함수는 'time', 'data' 형식의 CSV 파일을 읽어서
        모델 입력에 적합한 시퀀스 데이터로 변환합니다.
    
        Args:
            file_path (str): 데이터 파일 경로
            sequence_length (int): 시퀀스 길이 (기본값: 100)
        
        Returns:
            np.ndarray: 전처리된 시퀀스 데이터 (형태: [1, sequence_length, features])
        """
        try:
           # 데이터 로드
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            df = pd.read_csv(file_path)
        
            # 필요한 컬럼 확인
            required_columns = ['time', 'data']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"필요한 컬럼 {required_columns}이 데이터프레임에 없습니다.")
                return None
        
            # 데이터 정규화 또는 스케일링 (필요한 경우)
            # 이 부분은 기존 모델 학습에 사용된 스케일링 방식과 일치해야 합니다
            data_array = df['data'].values
        
            # 시퀀스 데이터 생성
            if len(data_array) < sequence_length:
                logger.warning(f"데이터 길이({len(data_array)})가 시퀀스 길이({sequence_length})보다 작습니다.")
                # 부족한 부분을 0으로 패딩
                padded_data = np.zeros(sequence_length)
                padded_data[:len(data_array)] = data_array
                data_array = padded_data
            elif len(data_array) > sequence_length:
                # 최신 데이터 위주로 시퀀스 길이만큼 자르기
                data_array = data_array[-sequence_length:]
        
            # 모델 입력 형태로 변환 (단일 특성이면 [1, sequence_length, 1] 형태)
            # 여러 특성이 있는 경우 각 특성을 별도 열로 추가
            model_input = data_array.reshape(1, sequence_length, 1)
        
            logger.info(f"실시간 데이터 전처리 완료: 형태={model_input.shape}")
            return model_input
        
        except Exception as e:
            logger.error(f"실시간 데이터 전처리 중 오류 발생: {str(e)}")
            return None

    def process_g2_realtime_data(self, file_path: str, sequence_length: int = 100) -> np.ndarray:
        """
        g2 실시간 센서 데이터를 모델 입력에 적합한 형태로 전처리합니다.
    
        이 함수는 g2 센서 데이터('time', 'data' 형식)를 읽어서
        다중 센서 LSTM 모델 입력에 적합한 시퀀스 데이터로 변환합니다.
    
        Args:
            file_path (str): 데이터 파일 경로
            sequence_length (int): 시퀀스 길이 (기본값: 100)
        
        Returns:
            np.ndarray: 전처리된 시퀀스 데이터 (형태: [1, sequence_length, features])
        """
        try:
            # 데이터 로드
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
            
            df = pd.read_csv(file_path)
        
            # 필요한 컬럼 확인
            if 'time' not in df.columns or 'data' not in df.columns:
                logger.error("필요한 컬럼(time, data)이 데이터프레임에 없습니다.")
                return None
        
            # 데이터 정규화 (기존 모델과 동일한 방식으로)
            data_values = df['data'].values
        
            # MinMax 스케일링 적용 (기존 모델 학습에 사용된 방식과 동일해야 함)
            # 여기서는 -1 ~ 1 범위로 가정합니다. 모델에 맞게 조정 필요
            min_val = data_values.min()
            max_val = data_values.max()
            if max_val > min_val:
                normalized_data = -1 + 2 * (data_values - min_val) / (max_val - min_val)
            else:
                normalized_data = np.zeros_like(data_values)
        
            # 시퀀스 데이터 생성 
            if len(normalized_data) < sequence_length:
                logger.warning(f"데이터 길이({len(normalized_data)})가 시퀀스 길이({sequence_length})보다 작습니다.")
                # 부족한 부분을 0으로 패딩
                padded_data = np.zeros(sequence_length)
                padded_data[:len(normalized_data)] = normalized_data
                sequence_data = padded_data
            else:
                # 최신 데이터를 시퀀스 길이만큼 사용
                sequence_data = normalized_data[-sequence_length:]
        
            # 단일 샘플, 4개 특성을 가진 모델 입력 형태로 변환
            # 현재는 단일 특성(data)만 있으므로 해당 특성을 4개의 입력 자리에 복사
            # 실제 상황에 맞게 조정해야 할 수 있습니다
            features = 4  # 모델의 입력 특성 수
            model_input = np.zeros((1, sequence_length, features))
        
            for i in range(features):
                model_input[0, :, i] = sequence_data
            
            logger.info(f"g2 실시간 데이터 전처리 완료: 형태={model_input.shape}")
            return model_input
        
        except Exception as e:
            logger.error(f"g2 실시간 데이터 전처리 중 오류 발생: {str(e)}")
            return None

def prepare_sequence_data(data: np.ndarray, sequence_length: int = 50) -> Tuple[np.ndarray, np.ndarray]:
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

    logger.info(f"시퀀스 데이터 생성 완료: X={X_sequences.shape}, y={y_sequences.shape}")
    
    return X_sequences, y_sequences
