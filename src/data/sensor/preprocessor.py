"""
센서 데이터 전처리 모듈

이 모듈은 센서 데이터에 대한 전처리 기능을 제공합니다:
- 데이터 로드 및 기본 전처리
- 보간법(interpolation)을 사용한 균일한 시간 간격의 데이터 생성
- 이동 평균(moving average)을 이용한 노이즈 감소
- 스케일링 및 정규화
- 통계적 특성 추출
"""

import numpy as np
import pandas as pd
from scipy import interpolate
from typing import Dict, List, Tuple, Union, Optional
import logging
import os
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        데이터 파일 로드
        
        Args:
            file_path (str): 로드할 파일 경로
            
        Returns:
            pd.DataFrame: 로드된 데이터프레임
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")
                
            df = pd.read_csv(file_path, names =["time","normal","type1","type2","type3"],header = None)
            
            # 시간 컬럼 확인 및 처리
            if 'timestamp' in df.columns:
                # timestamp를 time으로 변환 (필요한 경우)
                df['time'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
                logger.info(f"'timestamp' 컬럼을 'time' 컬럼으로 변환했습니다")
            elif 'time' not in df.columns:
                # time 컬럼이 없는 경우 인덱스 기반으로 생성
                df['time'] = np.arange(len(df))
                logger.info(f"'time' 컬럼이 없어 인덱스 기반으로 생성했습니다")
                
            logger.info(f"파일 로드 성공: {file_path}, {len(df)} 행, {len(df.columns)} 열")
            return df
        except Exception as e:
            logger.error(f"파일 로드 중 오류 발생: {e}")
            raise
    
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
    
    def clean_data(
        self, 
        df: pd.DataFrame,
        drop_columns: List[str] = None,
        fill_na: bool = True,
        na_strategy: str = 'both'  # 'ffill', 'bfill', 'mean', 'both'
    ) -> pd.DataFrame:
        """
        데이터 정제: 불필요한 컬럼 제거, 결측치 처리
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            drop_columns (List[str], optional): 제거할 컬럼 목록
            fill_na (bool): 결측치 처리 여부
            na_strategy (str): 결측치 처리 전략 ('ffill': 앞 값으로, 'bfill': 뒤 값으로, 
                              'mean': 평균으로, 'both': 앞/뒤 값으로 먼저 채우고 평균으로)
            
        Returns:
            pd.DataFrame: 정제된 데이터프레임
        """
        result_df = df.copy()
        
        # 불필요한 컬럼 제거
        if drop_columns:
            cols_to_drop = [col for col in drop_columns if col in result_df.columns]
            if cols_to_drop:
                result_df = result_df.drop(columns=cols_to_drop)
                logger.info(f"컬럼 제거: {cols_to_drop}")
        
        # 결측치 처리
        if fill_na:
            original_na_count = result_df.isna().sum().sum()
            
            if na_strategy == 'ffill' or na_strategy == 'both':
                result_df = result_df.fillna(method='ffill')
                
            if na_strategy == 'bfill' or na_strategy == 'both':
                result_df = result_df.fillna(method='bfill')
            
            # 남은 결측치는 컬럼별 평균으로 대체
            if na_strategy == 'mean' or na_strategy == 'both':
                for col in result_df.columns:
                    if result_df[col].dtype.kind in 'ifc':  # 숫자형 컬럼만
                        if result_df[col].isna().any():  # 결측치가 있는 경우만 처리
                            col_mean = result_df[col].mean()
                            result_df[col] = result_df[col].fillna(col_mean)
            
            remaining_na_count = result_df.isna().sum().sum()
            logger.info(f"결측치 처리 완료: {original_na_count}개 중 {original_na_count - remaining_na_count}개 처리됨")
            
            if remaining_na_count > 0:
                logger.warning(f"처리되지 않은 결측치가 {remaining_na_count}개 남아있습니다")
        
        return result_df
    
    def apply_moving_average(
        self, 
        df: pd.DataFrame, 
        columns: List[str] = None, 
        window_size: int = None,
        suffix: str = None
    ) -> pd.DataFrame:
        """
        지정된 컬럼에 이동 평균을 적용합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 이동 평균을 적용할 컬럼 목록. None이면 숫자형 컬럼 모두 적용.
            window_size (int, optional): 이동 평균 윈도우 크기. None이면 초기화 시 지정한 값 사용.
            suffix (str, optional): 결과 컬럼에 추가할 접미사. None이면 "_ma{window_size}" 사용.
            
        Returns:
            pd.DataFrame: 이동 평균이 적용된 데이터프레임
        """
        if window_size is None:
            window_size = self.window_size
        
        # 접미사 설정
        if suffix is None:
            suffix = f"_ma{window_size}"
        
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼과 이미 처리된 컬럼은 제외
            exclude_cols = ['time']
            exclude_cols.extend([col for col in columns if col.endswith(suffix)])
            columns = [col for col in columns if col not in exclude_cols]
        
        logger.info(f"이동 평균 적용: 윈도우 크기={window_size}, 컬럼 수={len(columns)}")
        
        result_df = df.copy()
        processed_columns = []
        
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
                    new_column = f"{column}{suffix}"
                    result_df[new_column] = padded_ma
                    processed_columns.append(new_column)
                    
                except Exception as e:
                    logger.error(f"컬럼 {column}에 이동 평균 적용 중 오류 발생: {e}")
            else:
                logger.warning(f"컬럼 {column}이 데이터프레임에 존재하지 않습니다.")
        
        logger.info(f"이동 평균 적용 완료: {len(processed_columns)}개 컬럼 처리됨")
        return result_df
    
    def scale_data(
        self,
        df: pd.DataFrame,
        target_column: str = None,
        scaling_method: str = 'minmax',
        feature_range: Tuple[int, int] = (-1, 1)
    ) -> Dict[str, any]:
        """
        데이터 스케일링 수행
        
        Args:
            df (pd.DataFrame): 스케일링할 데이터프레임
            target_column (str, optional): 타겟 컬럼명 (스케일링에서 제외)
            scaling_method (str): 스케일링 방법 ('minmax' 또는 'standard')
            feature_range (Tuple[int, int]): MinMaxScaler의 경우 스케일링 범위
            
        Returns:
            Dict[str, any]: 스케일링된 특성, 타겟, 관련 정보를 포함하는 딕셔너리
        """
        # 'time' 컬럼도 스케일링에서 제외
        exclude_columns = ['time']
        if target_column:
            exclude_columns.append(target_column)
        
        # 타겟 컬럼과 특성 컬럼 분리
        if target_column and target_column in df.columns:
            target = df[target_column].values
            feature_cols = [col for col in df.columns if col not in exclude_columns and df[col].dtype.kind in 'ifc']
            features = df[feature_cols].values
        else:
            target = None
            feature_cols = [col for col in df.columns if col not in exclude_columns and df[col].dtype.kind in 'ifc']
            features = df[feature_cols].values
        
        # 스케일링 방법 선택
        if scaling_method.lower() == 'minmax':
            scaler = MinMaxScaler(feature_range=feature_range)
        else:  # 'standard' 기본값
            scaler = StandardScaler()
        
        # 특성 스케일링
        scaled_features = scaler.fit_transform(features)
        
        logger.info(f"데이터 스케일링 완료: 방법={scaling_method}, 특성 수={features.shape[1]}")
        
        # 스케일러 저장 경로 구성 (선택적)
        # scaler_path = os.path.join(os.getcwd(), 'models', f"{scaling_method}_scaler.pkl")
        # joblib.dump(scaler, scaler_path)
        # logger.info(f"스케일러 저장: {scaler_path}")
        
        return {
            'features': scaled_features,
            'target': target,
            'feature_columns': feature_cols,
            'scaler': scaler
        }
    
    def extract_statistical_moments(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        window_size: int = 100,
        stats_to_extract: List[str] = None
    ) -> pd.DataFrame:
        """
        시계열 데이터에서 통계적 모멘트를 추출합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 처리할 컬럼 목록. None이면 숫자형 컬럼 모두 사용.
            window_size (int): 통계량 계산을 위한 윈도우 크기
            stats_to_extract (List[str], optional): 추출할 통계량 목록. 
                기본값은 ['std', 'skew', 'kurt', 'mean'].
            
        Returns:
            pd.DataFrame: 통계적 모멘트가 추가된 데이터프레임
        """
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼은 제외
            if 'time' in columns:
                columns.remove('time')
                
        # 통계량 설정
        if stats_to_extract is None:
            stats_to_extract = ['std', 'skew', 'kurt', 'mean']
        
        logger.info(f"통계적 모멘트 추출: 윈도우 크기={window_size}, 컬럼 수={len(columns)}, 통계량={stats_to_extract}")
        
        result_df = df.copy()
        processed_columns = []
        
        for column in columns:
            if column in df.columns:
                try:
                    # 롤링 윈도우를 사용한 통계량 계산
                    if 'std' in stats_to_extract:
                        # 2차 모멘트 - 표준편차 (변동성)
                        result_df[f"{column}_std{window_size}"] = df[column].rolling(window=window_size, min_periods=1).std()
                        processed_columns.append(f"{column}_std{window_size}")
                    
                    if 'skew' in stats_to_extract:
                        # 3차 모멘트 - 왜도 (비대칭성)
                        result_df[f"{column}_skew{window_size}"] = df[column].rolling(window=window_size, min_periods=1).skew()
                        processed_columns.append(f"{column}_skew{window_size}")
                    
                    if 'kurt' in stats_to_extract:
                        # 4차 모멘트 - 첨도 (뾰족함)
                        result_df[f"{column}_kurt{window_size}"] = df[column].rolling(window=window_size, min_periods=1).kurt()
                        processed_columns.append(f"{column}_kurt{window_size}")
                        
                    if 'mean' in stats_to_extract:
                        # 1차 모멘트 - 평균
                        result_df[f"{column}_mean{window_size}"] = df[column].rolling(window=window_size, min_periods=1).mean()
                        processed_columns.append(f"{column}_mean{window_size}")
                    
                except Exception as e:
                    logger.error(f"컬럼 {column}에 통계적 모멘트 추출 중 오류 발생: {e}")
            else:
                logger.warning(f"컬럼 {column}이 데이터프레임에 존재하지 않습니다.")
        
        logger.info(f"통계적 모멘트 추출 완료: {len(processed_columns)}개 컬럼 생성됨")
        return result_df

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
    
    print(f"처리된 센서 데이터 형태: {processed_sensor1.shape}")