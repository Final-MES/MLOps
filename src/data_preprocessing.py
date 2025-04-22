"""
센서 데이터 전처리 모듈

이 모듈은 센서 데이터에 대한 전처리 기능을 제공합니다:
- 보간법(interpolation)을 사용한 균일한 시간 간격의 데이터 생성
- 이동 평균(moving average)을 이용한 노이즈 감소
- 다양한 센서 유형별 데이터 변환 및 통합
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Dict, List, Tuple, Union, Optional
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class SensorDataPreprocessor:
    """센서 데이터 전처리를 위한 클래스"""
    
    def __init__(self, window_size: int = 15):
        """
        전처리기 초기화
        
        Args:
            window_size (int): 이동 평균의 윈도우 크기. 기본값은 15.
        """
        self.window_size = window_size
        logger.info(f"센서 데이터 전처리기 초기화: 윈도우 크기 = {window_size}")
    
    def interpolate_sensor_data(
        self, 
        sensor_data: Dict[str, pd.DataFrame], 
        time_range: np.ndarray = None,
        step: float = 0.001,
        kind: str = 'linear'
    ) -> Dict[str, pd.DataFrame]:
        """
        센서 데이터를 균일한 시간 간격으로 보간합니다.
        
        Args:
            sensor_data (Dict[str, pd.DataFrame]): 각 센서별 데이터프레임을 포함하는 사전.
                                                'time' 컬럼과 'normal', 'type1', 'type2', 'type3' 등의 컬럼 필요.
            time_range (np.ndarray, optional): 보간에 사용할 시간 범위. None이면 자동 생성.
            step (float): 시간 간격 단위. 기본값은 0.001.
            kind (str): 보간 방법. 기본값은 'linear'. 'cubic', 'quadratic' 등도 가능.
            
        Returns:
            Dict[str, pd.DataFrame]: 보간된 센서 데이터
        """
        logger.info(f"센서 데이터 보간 시작: 간격={step}, 방법={kind}")
        
        # 시간 범위가 제공되지 않은 경우 자동 생성
        if time_range is None:
            # 모든 센서의 시간 범위 확인
            min_time = float('inf')
            max_time = float('-inf')
            
            for sensor_name, df in sensor_data.items():
                if 'time' in df.columns:
                    min_time = min(min_time, df['time'].min())
                    max_time = max(max_time, df['time'].max())
            
            # 시간 범위 생성
            time_range = np.arange(min_time, max_time + step, step)
            logger.info(f"시간 범위 자동 생성: {min_time} ~ {max_time}, {len(time_range)} 포인트")
        
        interpolated_data = {}
        
        for sensor_name, df in sensor_data.items():
            if 'time' not in df.columns:
                logger.warning(f"센서 {sensor_name}에 'time' 컬럼이 없습니다. 건너뜁니다.")
                continue
            
            # 각 상태(normal, type1 등)에 대한 보간 수행
            columns = [col for col in df.columns if col != 'time']
            interpolated_values = []
            
            for column in columns:
                try:
                    # 보간 함수 생성
                    f_interp = interpolate.interp1d(
                        df['time'], 
                        df[column], 
                        kind=kind,
                        bounds_error=False,  # 범위를 벗어나는 경우 NaN 처리
                        fill_value="extrapolate"  # 필요한 경우 외삽
                    )
                    
                    # 보간 적용
                    interpolated_values.append(f_interp(time_range))
                    
                except Exception as e:
                    logger.error(f"센서 {sensor_name}의 {column} 보간 중 오류 발생: {e}")
                    # 오류 발생 시 NaN으로 채운 배열 생성
                    interpolated_values.append(np.full_like(time_range, np.nan, dtype=float))
            
            # 보간된 값으로 데이터프레임 생성
            interpolated_df = pd.DataFrame(
                np.array(interpolated_values).T, 
                columns=columns
            )
            interpolated_df['time'] = time_range
            
            interpolated_data[sensor_name] = interpolated_df
            logger.info(f"센서 {sensor_name} 보간 완료: {len(interpolated_df)} 행")
        
        return interpolated_data
    
    def apply_moving_average(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None, 
        window_size: int = None
    ) -> pd.DataFrame:
        """
        지정된 컬럼에 이동 평균을 적용합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 이동 평균을 적용할 컬럼 목록. None이면 숫자형 컬럼 모두 적용.
            window_size (int, optional): 이동 평균 윈도우 크기. None이면 초기화 시 지정한 값 사용.
            
        Returns:
            pd.DataFrame: 이동 평균이 적용된 데이터프레임
        """
        if window_size is None:
            window_size = self.window_size
        
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼은 제외
            if 'time' in columns:
                columns.remove('time')
        
        logger.info(f"이동 평균 적용: 윈도우 크기={window_size}, 컬럼={columns}")
        
        result_df = df.copy()
        
        for column in columns:
            if column in df.columns:
                try:
                    # 이동 평균 계산 (NumPy convolve 사용)
                    values = df[column].values
                    ma_values = np.convolve(values, np.ones(window_size), 'valid') / window_size
                    
                    # 결과 길이가 원본보다 짧아짐 (window_size-1만큼)
                    # 시작 부분에 NaN을 채워서 길이 맞추기
                    padding = np.full(window_size-1, np.nan)
                    padded_ma = np.concatenate([padding, ma_values])
                    
                    # 결과 저장
                    result_df[f"{column}_ma{window_size}"] = padded_ma
                    
                except Exception as e:
                    logger.error(f"컬럼 {column}에 이동 평균 적용 중 오류 발생: {e}")
            else:
                logger.warning(f"컬럼 {column}이 데이터프레임에 존재하지 않습니다.")
        
        return result_df
    
    def get_ma_columns(self, df: pd.DataFrame, prefix: str, window_size: int = None) -> np.ndarray:
        """
        이동 평균이 적용된 컬럼을 추출하여 NumPy 배열로 반환합니다.
        
        Args:
            df (pd.DataFrame): 이동 평균이 적용된 데이터프레임
            prefix (str): 추출할 컬럼의 접두사 (예: 's1', 's2')
            window_size (int, optional): 이동 평균 윈도우 크기. None이면 초기화 시 지정한 값 사용.
            
        Returns:
            np.ndarray: 추출된 데이터를 포함하는 형태가 변환된(reshaped) 배열
        """
        if window_size is None:
            window_size = self.window_size
        
        column_name = f"{prefix}_ma{window_size}" if f"{prefix}_ma{window_size}" in df.columns else prefix
        
        if column_name in df.columns:
            values = df[column_name].values
            # NaN 값 제거 (이동 평균 적용으로 인한 시작 부분의 NaN)
            values = values[~np.isnan(values)]
            # 배열 형태 변환 (n,1) 형태로
            return values.reshape(len(values), 1)
        else:
            logger.warning(f"컬럼 {column_name}이 데이터프레임에 존재하지 않습니다.")
            return np.array([]).reshape(0, 1)
    
    def preprocess_sensor_data(
        self, 
        sensor_data: Dict[str, pd.DataFrame],
        sensor_columns: List[str] = None,
        fault_types: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        센서 데이터를 완전히 전처리하여 분석 준비가 된 형태로 변환합니다.
        
        Args:
            sensor_data (Dict[str, pd.DataFrame]): 각 센서별 데이터프레임을 포함하는 사전
            sensor_columns (List[str], optional): 처리할 센서 컬럼 목록 (예: ['s1', 's2', 's3', 's4'])
            fault_types (List[str], optional): 처리할 고장 유형 목록 (예: ['normal', 'type1', 'type2', 'type3'])
            
        Returns:
            Dict[str, np.ndarray]: 전처리된 데이터를 포함하는 사전
            (각 고장 유형별로 모든 센서 데이터가 통합된 배열)
        """
        if sensor_columns is None:
            sensor_columns = ['s1', 's2', 's3', 's4']
        
        if fault_types is None:
            fault_types = ['normal', 'type1', 'type2', 'type3']
        
        logger.info(f"센서 데이터 전처리 시작: 센서={sensor_columns}, 고장 유형={fault_types}")
        
        # 결과를 저장할 사전
        result = {}
        
        # 각 고장 유형별로 처리
        for fault_type in fault_types:
            if fault_type not in sensor_data:
                logger.warning(f"고장 유형 {fault_type}에 대한 데이터가 없습니다.")
                continue
            
            df = sensor_data[fault_type]
            
            # 이동 평균 적용
            processed_df = self.apply_moving_average(df, sensor_columns)
            
            # 각 센서 컬럼에 대한 이동 평균 배열 추출
            sensor_arrays = []
            for sensor_col in sensor_columns:
                sensor_array = self.get_ma_columns(processed_df, sensor_col)
                if len(sensor_array) > 0:
                    sensor_arrays.append(sensor_array)
            
            # 모든 센서 배열 수평 연결 (n행 x m센서)
            if sensor_arrays:
                combined_array = np.concatenate(sensor_arrays, axis=1)
                result[fault_type] = combined_array
                logger.info(f"고장 유형 {fault_type} 처리 완료: 형태={combined_array.shape}")
            else:
                logger.warning(f"고장 유형 {fault_type}에 대한 처리된 센서 데이터가 없습니다.")
        
        return result
    
    def preprocess_data_from_external_api(
        self, 
        api_data: pd.DataFrame, 
        time_column: str = 'timestamp',
        sensor_columns: List[str] = None,
        window_size: int = None
    ) -> Dict[str, np.ndarray]:
        """
        외부 API에서 수집한 데이터를 전처리합니다.
        
        Args:
            api_data (pd.DataFrame): API에서 수집한 원본 데이터
            time_column (str): 시간 정보가 포함된 컬럼명
            sensor_columns (List[str], optional): 처리할 센서 컬럼 목록
            window_size (int, optional): 이동 평균 윈도우 크기
            
        Returns:
            Dict[str, np.ndarray]: 전처리된 데이터를 포함하는 사전
        """
        if window_size is None:
            window_size = self.window_size
        
        if sensor_columns is None:
            # 숫자형 컬럼 중 시간 컬럼 제외
            sensor_columns = api_data.select_dtypes(include=[np.number]).columns.tolist()
            if time_column in sensor_columns:
                sensor_columns.remove(time_column)
        
        logger.info(f"외부 API 데이터 전처리 시작: 센서={sensor_columns}")
        
        # 시간 컬럼이 datetime 형식이면 숫자로 변환 (단위: 초)
        if pd.api.types.is_datetime64_any_dtype(api_data[time_column]):
            # 기준 시간 설정 (첫 번째 행의 시간)
            reference_time = api_data[time_column].min()
            api_data['time'] = (api_data[time_column] - reference_time).dt.total_seconds()
        else:
            # 이미 숫자 형식이면 그대로 사용
            api_data['time'] = api_data[time_column]
        
        # 균일한 시간 간격으로 보간
        time_range = np.arange(
            api_data['time'].min(), 
            api_data['time'].max() + 0.001, 
            0.001
        )
        
        interpolated_values = []
        for column in sensor_columns:
            try:
                # 보간 함수 생성
                f_interp = interpolate.interp1d(
                    api_data['time'], 
                    api_data[column], 
                    kind='linear',
                    bounds_error=False,
                    fill_value="extrapolate"
                )
                
                # 보간 적용
                interpolated_values.append(f_interp(time_range))
                
            except Exception as e:
                logger.error(f"센서 {column} 보간 중 오류 발생: {e}")
                # 오류 발생 시 NaN으로 채운 배열 생성
                interpolated_values.append(np.full_like(time_range, np.nan, dtype=float))
        
        # 보간된 값으로 데이터프레임 생성
        interpolated_df = pd.DataFrame(
            np.array(interpolated_values).T, 
            columns=sensor_columns
        )
        interpolated_df['time'] = time_range
        
        # 이동 평균 적용
        processed_df = self.apply_moving_average(interpolated_df, sensor_columns, window_size)
        
        # 각 센서 컬럼에 대한 이동 평균 배열 추출
        sensor_arrays = []
        for sensor_col in sensor_columns:
            sensor_array = self.get_ma_columns(processed_df, sensor_col, window_size)
            if len(sensor_array) > 0:
                sensor_arrays.append(sensor_array)
        
        # 모든 센서 배열 수평 연결 (n행 x m센서)
        result = {}
        if sensor_arrays:
            combined_array = np.concatenate(sensor_arrays, axis=1)
            result['processed_data'] = combined_array
            logger.info(f"외부 API 데이터 처리 완료: 형태={combined_array.shape}")
        
        return result


# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 예시 데이터 생성 (실제 사용 시에는 실제 데이터로 대체)
    time = np.linspace(0, 100, 1000)
    normal_data = np.sin(time / 5) + np.random.normal(0, 0.1, 1000)
    type1_data = np.sin(time / 5) + 0.5 + np.random.normal(0, 0.1, 1000)
    type2_data = np.sin(time / 5) * 0.5 + np.random.normal(0, 0.1, 1000)
    type3_data = np.sin(time / 5 + 1) + np.random.normal(0, 0.1, 1000)
    
    sensor1 = pd.DataFrame({
        'time': time,
        'normal': normal_data,
        'type1': type1_data,
        'type2': type2_data,
        'type3': type3_data
    })
    
    sensor2 = pd.DataFrame({
        'time': time,
        'normal': normal_data * 0.8,
        'type1': type1_data * 0.8,
        'type2': type2_data * 0.8,
        'type3': type3_data * 0.8
    })
    
    # 전처리기 초기화
    preprocessor = SensorDataPreprocessor(window_size=15)
    
    # 보간 적용
    sensor_data = {'sensor1': sensor1, 'sensor2': sensor2}
    interpolated_data = preprocessor.interpolate_sensor_data(
        sensor_data,
        time_range=np.arange(0, 100, 0.001),
        step=0.001
    )
    
    # 이동 평균 적용
    processed_sensor1 = preprocessor.apply_moving_average(
        interpolated_data['sensor1'], 
        columns=['normal', 'type1', 'type2', 'type3']
    )
    
    # 특정 컬럼 추출
    normal_s1 = preprocessor.get_ma_columns(processed_sensor1, 'normal')
    
    print(f"처리된 센서 데이터 형태: {normal_s1.shape}")