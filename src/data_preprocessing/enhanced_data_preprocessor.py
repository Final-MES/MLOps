import numpy as np
import pandas as pd
from scipy import interpolate, signal
from typing import Dict, List, Tuple, Union, Optional
import logging
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 로깅 설정
logger = logging.getLogger(__name__)

class SensorDataPreprocessor:
    """센서 데이터 전처리를 위한 확장된 클래스"""
    
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
                                                'time' 컬럼과 센서 값 컬럼들이 필요함.
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
                    logger.info(f"센서 {sensor_name} 시간 범위: {min_time} ~ {max_time}")
            
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
    
    def extract_statistical_moments(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        window_size: int = 100
    ) -> pd.DataFrame:
        """
        시계열 데이터에서 통계적 모멘트를 추출합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 처리할 컬럼 목록. None이면 숫자형 컬럼 모두 사용.
            window_size (int): 통계량 계산을 위한 윈도우 크기
            
        Returns:
            pd.DataFrame: 통계적 모멘트가 추가된 데이터프레임
        """
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼은 제외
            if 'time' in columns:
                columns.remove('time')
        
        logger.info(f"통계적 모멘트 추출: 윈도우 크기={window_size}, 컬럼={columns}")
        
        result_df = df.copy()
        
        for column in columns:
            if column in df.columns:
                try:
                    # 롤링 윈도우를 사용한 통계량 계산
                    # 1차 모멘트 - 평균 (이미 이동 평균으로 계산됨)
                    
                    # 2차 모멘트 - 표준편차 (변동성)
                    result_df[f"{column}_std{window_size}"] = df[column].rolling(window=window_size, min_periods=1).std()
                    
                    # 3차 모멘트 - 왜도 (비대칭성)
                    result_df[f"{column}_skew{window_size}"] = df[column].rolling(window=window_size, min_periods=1).skew()
                    
                    # 4차 모멘트 - 첨도 (뾰족함)
                    result_df[f"{column}_kurt{window_size}"] = df[column].rolling(window=window_size, min_periods=1).kurt()
                    
                    # 추가 통계량 - RMS (Root Mean Square)
                    rms = np.sqrt(np.mean(np.square(df[column].rolling(window=window_size, min_periods=1).apply(
                        lambda x: np.array(x), raw=True))))
                    result_df[f"{column}_rms{window_size}"] = rms
                    
                except Exception as e:
                    logger.error(f"컬럼 {column}에 통계적 모멘트 추출 중 오류 발생: {e}")
            else:
                logger.warning(f"컬럼 {column}이 데이터프레임에 존재하지 않습니다.")
        
        return result_df
    
    def extract_frequency_features(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        window_size: int = 256,
        sampling_rate: float = 1.0
    ) -> pd.DataFrame:
        """
        시계열 데이터에서 주파수 도메인 특성을 추출합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 처리할 컬럼 목록. None이면 숫자형 컬럼 모두 사용.
            window_size (int): FFT 윈도우 크기 (2의 거듭제곱이 효율적)
            sampling_rate (float): 센서 데이터의 샘플링 주기(Hz)
            
        Returns:
            pd.DataFrame: 주파수 특성이 추가된 데이터프레임
        """
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼은 제외
            if 'time' in columns:
                columns.remove('time')
        
        logger.info(f"주파수 도메인 특성 추출: 윈도우 크기={window_size}, 컬럼={columns}")
        
        result_df = df.copy()
        
        # 결과 저장을 위한 빈 리스트 초기화
        dominant_freqs = {col: [] for col in columns}
        spectral_energy = {col: [] for col in columns}
        spectral_entropy = {col: [] for col in columns}
        
        # 주파수 도메인 특성 추출
        for i in range(0, len(df), window_size//2):  # 50% 오버랩
            end_idx = min(i + window_size, len(df))
            if end_idx - i < window_size // 2:  # 너무 작은 윈도우는 건너뜀
                continue
                
            for column in columns:
                if column in df.columns:
                    try:
                        # 윈도우 데이터 추출
                        window_data = df[column].iloc[i:end_idx].values
                        
                        # 윈도우 크기가 목표보다 작으면 패딩
                        if len(window_data) < window_size:
                            window_data = np.pad(window_data, (0, window_size - len(window_data)), 'constant', constant_values=0)
                        
                        # 윈도잉 함수 적용 (신호 누출 방지)
                        window_func = np.hamming(len(window_data))
                        windowed_data = window_data * window_func
                        
                        # FFT 수행
                        fft_result = fft(windowed_data)
                        # 주파수 축 계산
                        freqs = fftfreq(len(windowed_data), 1/sampling_rate)
                        
                        # 양의 주파수만 선택 (절반)
                        pos_mask = freqs >= 0
                        freqs = freqs[pos_mask]
                        fft_mag = np.abs(fft_result[pos_mask])
                        
                        # 파워 스펙트럼 계산 (정규화)
                        power_spectrum = fft_mag**2 / len(windowed_data)
                        
                        # 주요 주파수 특성 추출
                        if np.sum(power_spectrum) > 0:
                            # 1. 우세 주파수 (최대 에너지를 가진 주파수)
                            dominant_freq = freqs[np.argmax(power_spectrum)]
                            
                            # 2. 스펙트럼 에너지 (전체 에너지)
                            energy = np.sum(power_spectrum)
                            
                            # 3. 스펙트럼 엔트로피 (주파수 분포의 무질서도)
                            norm_power = power_spectrum / energy
                            entropy = -np.sum(norm_power * np.log2(norm_power + 1e-10))
                        else:
                            dominant_freq = 0
                            energy = 0
                            entropy = 0
                        
                        # 결과 저장
                        dominant_freqs[column].append(dominant_freq)
                        spectral_energy[column].append(energy)
                        spectral_entropy[column].append(entropy)
                        
                    except Exception as e:
                        logger.error(f"컬럼 {column}에 주파수 특성 추출 중 오류 발생: {e}")
                        dominant_freqs[column].append(0)
                        spectral_energy[column].append(0)
                        spectral_entropy[column].append(0)
        
        # 주파수 특성을 원본 데이터프레임의 각 행에 맵핑 (반복)
        for column in columns:
            if len(dominant_freqs[column]) > 0:
                # 주파수 특성 배열을 원본 데이터프레임 길이에 맞게 확장
                dom_freq_expanded = np.repeat(dominant_freqs[column], window_size//2)[:len(df)]
                energy_expanded = np.repeat(spectral_energy[column], window_size//2)[:len(df)]
                entropy_expanded = np.repeat(spectral_entropy[column], window_size//2)[:len(df)]
                
                # 데이터프레임에 추가
                result_df[f"{column}_dom_freq"] = dom_freq_expanded
                result_df[f"{column}_energy"] = energy_expanded
                result_df[f"{column}_entropy"] = entropy_expanded
        
        return result_df
    
    def extract_cross_correlations(
        self,
        df: pd.DataFrame,
        columns: List[str] = None,
        window_size: int = 100,
        max_lag: int = 10
    ) -> pd.DataFrame:
        """
        센서 간 교차상관관계를 계산합니다.
        
        Args:
            df (pd.DataFrame): 원본 데이터프레임
            columns (List[str], optional): 처리할 컬럼 목록. None이면 숫자형 컬럼 모두 사용.
            window_size (int): 상관관계 계산을 위한 윈도우 크기
            max_lag (int): 최대 지연 단계
            
        Returns:
            pd.DataFrame: 교차상관관계가 추가된 데이터프레임
        """
        # 컬럼이 지정되지 않은 경우 숫자형 컬럼 모두 선택
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
            # 'time' 컬럼은 제외
            if 'time' in columns:
                columns.remove('time')
        
        logger.info(f"교차상관관계 계산: 윈도우 크기={window_size}, 컬럼={columns}")
        
        result_df = df.copy()
        
        # 모든 센서 쌍에 대해 교차상관관계 계산
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i >= j:  # 중복 방지
                    continue
                
                try:
                    # 롤링 윈도우 방식으로 교차상관관계 계산
                    xcorr_values = []
                    
                    for k in range(len(df) - window_size + 1):
                        # 현재 윈도우 데이터
                        window1 = df[col1].iloc[k:k+window_size].values
                        window2 = df[col2].iloc[k:k+window_size].values
                        
                        # 교차상관관계 계산
                        xcorr = np.correlate(window1 - np.mean(window1), window2 - np.mean(window2), mode='full')
                        # 정규화
                        xcorr = xcorr / (np.std(window1) * np.std(window2) * len(window1))
                        
                        # 관심 있는 지연(lag) 범위만 선택
                        mid_point = len(xcorr) // 2
                        xcorr_trimmed = xcorr[mid_point-max_lag:mid_point+max_lag+1]
                        
                        # 최대 상관계수와 해당 지연 저장
                        max_xcorr_idx = np.argmax(np.abs(xcorr_trimmed))
                        max_xcorr = xcorr_trimmed[max_xcorr_idx]
                        lag = max_xcorr_idx - max_lag
                        
                        xcorr_values.append((max_xcorr, lag))
                    
                    # 계산된 값을 데이터프레임에 추가
                    # 패딩 (윈도우 크기 - 1개의 NaN으로 시작)
                    padding = [(np.nan, np.nan)] * (window_size - 1)
                    xcorr_values = padding + xcorr_values
                    
                    # 최대 상관계수와 지연 분리
                    max_xcorr_series, lag_series = zip(*xcorr_values)
                    
                    # 데이터프레임에 추가
                    col_name = f"{col1}_{col2}"
                    result_df[f"{col_name}_xcorr"] = max_xcorr_series
                    result_df[f"{col_name}_lag"] = lag_series
                    
                except Exception as e:
                    logger.error(f"{col1}와 {col2} 간 교차상관관계 계산 중 오류 발생: {e}")
        
        return result_df
    
    def preprocess_for_anomaly_detection(
        self,
        sensor_data: Dict[str, pd.DataFrame],
        extract_features: bool = True,
        window_size: int = 100,
        sampling_rate: float = 1.0
    ) -> pd.DataFrame:
        """
        이상 감지를 위한 센서 데이터 전처리를 수행합니다.
        
        Args:
            sensor_data (Dict[str, pd.DataFrame]): 각 센서별 데이터프레임을 포함하는 사전
            extract_features (bool): 추가 특성 추출 여부
            window_size (int): 특성 추출을 위한 윈도우 크기
            sampling_rate (float): 센서 데이터의 샘플링 주기(Hz)
            
        Returns:
            pd.DataFrame: 이상 감지를 위해 전처리된 데이터프레임
        """
        # 먼저 센서 데이터 보간
        interpolated_data = self.interpolate_sensor_data(sensor_data)
        
        # 모든 센서 데이터를 통합
        combined_df = None
        
        for sensor_id, df in interpolated_data.items():
            if combined_df is None:
                combined_df = df.copy()
                # 센서 ID를 컬럼명에 추가
                combined_df.columns = [f"{sensor_id}_{col}" if col != 'time' else col for col in combined_df.columns]
            else:
                # 'time' 컬럼을 제외한 나머지 컬럼만 추가
                temp_df = df.copy()
                temp_df.columns = [f"{sensor_id}_{col}" if col != 'time' else col for col in temp_df.columns]
                # 'time' 컬럼을 기준으로 결합
                combined_df = pd.merge(combined_df, temp_df, on='time', how='outer')
        
        if combined_df is None:
            logger.error("보간된 센서 데이터가 없습니다.")
            return pd.DataFrame()
        
        # 이동 평균 적용
        columns = [col for col in combined_df.columns if col != 'time']
        smoothed_df = self.apply_moving_average(combined_df, columns)
        
        # 추가 특성 추출
        if extract_features:
            # 통계적 모멘트 추출
            feature_df = self.extract_statistical_moments(smoothed_df, columns, window_size)
            
            # 주파수 도메인 특성 추출
            feature_df = self.extract_frequency_features(feature_df, columns, window_size, sampling_rate)
            
            # 교차상관관계 추출 (센서 수가 많으면 계산량이 많아질 수 있음)
            if len(columns) <= 10:  # 센서 수가 적을 때만 수행
                feature_df = self.extract_cross_correlations(feature_df, columns, window_size)
            
            return feature_df
        else:
            return smoothed_df
    
    def visualize_preprocessing_results(
        self,
        original_data: Dict[str, pd.DataFrame],
        processed_data: pd.DataFrame,
        sensor_id: str = None,
        column: str = None,
        save_path: str = None
    ):
        """
        전처리 결과를 시각화합니다.
        
        Args:
            original_data (Dict[str, pd.DataFrame]): 원본 센서 데이터
            processed_data (pd.DataFrame): 전처리된 데이터
            sensor_id (str, optional): 시각화할 센서 ID
            column (str, optional): 시각화할 컬럼
            save_path (str, optional): 시각화 결과 저장 경로
        """
        if sensor_id is None:
            # 첫 번째 센서 선택
            sensor_id = list(original_data.keys())[0]
        
        original_df = original_data[sensor_id]
        
        if column is None:
            # 'time'을 제외한 첫 번째 컬럼 선택
            column = [col for col in original_df.columns if col != 'time'][0]
        
        # 원본 데이터와 전처리된 데이터 시각화
        plt.figure(figsize=(15, 10))
        
        # 원본 데이터
        plt.subplot(3, 1, 1)
        plt.plot(original_df['time'], original_df[column], 'b-', alpha=0.7, label='원본 데이터')
        plt.title(f'센서 {sensor_id}의 {column} - 원본 데이터')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # 보간된 데이터
        plt.subplot(3, 1, 2)
        processed_col = f"{sensor_id}_{column}"
        if processed_col in processed_data.columns:
            plt.plot(processed_data['time'], processed_data[processed_col], 'g-', label='보간된 데이터')
            plt.title(f'센서 {sensor_id}의 {column} - 보간된 데이터')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        # 이동 평균 적용 데이터
        plt.subplot(3, 1, 3)
        ma_col = f"{processed_col}_ma{self.window_size}"
        if ma_col in processed_data.columns:
            plt.plot(processed_data['time'], processed_data[ma_col], 'r-', label=f'이동 평균 (윈도우={self.window_size})')
            plt.title(f'센서 {sensor_id}의 {column} - 이동 평균 데이터')
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path)
            logger.info(f"시각화 결과 저장: {save_path}")
        else:
            plt.show()