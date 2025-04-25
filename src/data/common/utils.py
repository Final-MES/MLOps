"""
공통 데이터 처리 유틸리티

이 모듈은 다양한 데이터 유형에서 공통적으로 사용할 수 있는 
데이터 처리 유틸리티 함수들을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import json

# 로깅 설정
logger = logging.getLogger(__name__)

def check_file_exists(file_path: Union[str, Path]) -> bool:
    """
    파일 존재 여부 확인
    
    Args:
        file_path: 확인할 파일 경로
        
    Returns:
        bool: 파일 존재 여부
    """
    return os.path.isfile(file_path)

def save_to_json(data: Any, file_path: Union[str, Path], indent: int = 4) -> bool:
    """
    데이터를 JSON 파일로 저장
    
    Args:
        data: 저장할 데이터
        file_path: 저장할 파일 경로
        indent: JSON 들여쓰기 크기
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        with open(file_path, 'w') as f:
            if hasattr(data, 'tolist'):  # NumPy 배열인 경우
                data = data.tolist()
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        logger.error(f"JSON 저장 중 오류 발생: {e}")
        return False

def load_from_json(file_path: Union[str, Path]) -> Optional[Any]:
    """
    JSON 파일에서 데이터 로드
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        Optional[Any]: 로드된 데이터 또는 오류 시 None
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON 로드 중 오류 발생: {e}")
        return None

def save_numpy_array(array: np.ndarray, file_path: Union[str, Path]) -> bool:
    """
    NumPy 배열을 파일로 저장
    
    Args:
        array: 저장할 NumPy 배열
        file_path: 저장할 파일 경로
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        np.save(file_path, array)
        return True
    except Exception as e:
        logger.error(f"NumPy 배열 저장 중 오류 발생: {e}")
        return False

def load_numpy_array(file_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    파일에서 NumPy 배열 로드
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        Optional[np.ndarray]: 로드된 NumPy 배열 또는 오류 시 None
    """
    try:
        return np.load(file_path)
    except Exception as e:
        logger.error(f"NumPy 배열 로드 중 오류 발생: {e}")
        return None

def save_pandas_dataframe(df: pd.DataFrame, file_path: Union[str, Path], format: str = 'csv') -> bool:
    """
    Pandas DataFrame을 파일로 저장
    
    Args:
        df: 저장할 DataFrame
        file_path: 저장할 파일 경로
        format: 저장 형식 ('csv', 'parquet', 'excel')
        
    Returns:
        bool: 저장 성공 여부
    """
    try:
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        else:
            logger.error(f"지원하지 않는 형식: {format}")
            return False
        return True
    except Exception as e:
        logger.error(f"DataFrame 저장 중 오류 발생: {e}")
        return False

def load_pandas_dataframe(file_path: Union[str, Path], format: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    파일에서 Pandas DataFrame 로드
    
    Args:
        file_path: 로드할 파일 경로
        format: 파일 형식 (None이면 확장자에서 유추)
        
    Returns:
        Optional[pd.DataFrame]: 로드된 DataFrame 또는 오류 시 None
    """
    try:
        if format is None:
            # 확장자에서 형식 유추
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.csv':
                format = 'csv'
            elif ext == '.parquet':
                format = 'parquet'
            elif ext in ['.xls', '.xlsx']:
                format = 'excel'
            else:
                logger.error(f"지원하지 않는 파일 확장자: {ext}")
                return None
        
        if format.lower() == 'csv':
            return pd.read_csv(file_path)
        elif format.lower() == 'parquet':
            return pd.read_parquet(file_path)
        elif format.lower() == 'excel':
            return pd.read_excel(file_path)
        else:
            logger.error(f"지원하지 않는 형식: {format}")
            return None
    except Exception as e:
        logger.error(f"DataFrame 로드 중 오류 발생: {e}")
        return None

def normalize_data(data: np.ndarray, method: str = 'minmax', axis: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    데이터 정규화
    
    Args:
        data: 정규화할 데이터
        method: 정규화 방법 ('minmax', 'standard', 'robust')
        axis: 정규화 축 (0: 각 특성별로, 1: 각 샘플별로)
        
    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: (정규화된 데이터, 정규화 파라미터)
    """
    params = {}
    
    if method == 'minmax':
        # Min-Max 정규화 (0~1)
        data_min = np.min(data, axis=axis, keepdims=True)
        data_max = np.max(data, axis=axis, keepdims=True)
        normalized_data = (data - data_min) / (data_max - data_min + 1e-8)
        params = {'min': data_min, 'max': data_max}
        
    elif method == 'standard':
        # 표준화 (평균 0, 표준편차 1)
        data_mean = np.mean(data, axis=axis, keepdims=True)
        data_std = np.std(data, axis=axis, keepdims=True)
        normalized_data = (data - data_mean) / (data_std + 1e-8)
        params = {'mean': data_mean, 'std': data_std}
        
    elif method == 'robust':
        # 로버스트 정규화 (중앙값 0, IQR 1)
        data_median = np.median(data, axis=axis, keepdims=True)
        q1 = np.percentile(data, 25, axis=axis, keepdims=True)
        q3 = np.percentile(data, 75, axis=axis, keepdims=True)
        iqr = q3 - q1
        normalized_data = (data - data_median) / (iqr + 1e-8)
        params = {'median': data_median, 'iqr': iqr}
        
    else:
        logger.error(f"지원하지 않는 정규화 방법: {method}")
        return data, {}
    
    return normalized_data, params

def denormalize_data(data: np.ndarray, params: Dict[str, Any], method: str = 'minmax') -> np.ndarray:
    """
    정규화된 데이터 원복
    
    Args:
        data: 정규화된 데이터
        params: 정규화 파라미터
        method: 정규화 방법 ('minmax', 'standard', 'robust')
        
    Returns:
        np.ndarray: 원본 스케일로 복원된 데이터
    """
    if method == 'minmax':
        # Min-Max 역정규화
        data_min = params.get('min')
        data_max = params.get('max')
        
        if data_min is None or data_max is None:
            logger.error("Min-Max 정규화 파라미터 없음")
            return data
            
        return data * (data_max - data_min) + data_min
        
    elif method == 'standard':
        # 표준화 역변환
        data_mean = params.get('mean')
        data_std = params.get('std')
        
        if data_mean is None or data_std is None:
            logger.error("표준화 파라미터 없음")
            return data
            
        return data * data_std + data_mean
        
    elif method == 'robust':
        # 로버스트 정규화 역변환
        data_median = params.get('median')
        iqr = params.get('iqr')
        
        if data_median is None or iqr is None:
            logger.error("로버스트 정규화 파라미터 없음")
            return data
            
        return data * iqr + data_median
        
    else:
        logger.error(f"지원하지 않는 정규화 방법: {method}")
        return data

def create_directory_if_not_exists(directory: Union[str, Path]) -> bool:
    """
    디렉토리가 존재하지 않으면 생성
    
    Args:
        directory: 생성할 디렉토리 경로
        
    Returns:
        bool: 생성 성공 여부
    """
    try:
        os.makedirs(directory, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"디렉토리 생성 중 오류 발생: {e}")
        return False

def split_dataset(data: np.ndarray, labels: Optional[np.ndarray] = None, 
                train_ratio: float = 0.7, valid_ratio: float = 0.15, test_ratio: float = 0.15, 
                shuffle: bool = True, random_seed: int = 42) -> Dict[str, np.ndarray]:
    """
    데이터셋 분할
    
    Args:
        data: 분할할 데이터
        labels: 분할할 레이블 (없으면 None)
        train_ratio: 학습 데이터 비율
        valid_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        shuffle: 셔플 여부
        random_seed: 랜덤 시드
        
    Returns:
        Dict[str, np.ndarray]: 분할된 데이터셋 사전
    """
    # 비율의 합이 1인지 확인
    if abs(train_ratio + valid_ratio + test_ratio - 1.0) > 1e-10:
        logger.warning("비율의 합이 1이 아닙니다. 자동으로 조정합니다.")
        total = train_ratio + valid_ratio + test_ratio
        train_ratio /= total
        valid_ratio /= total
        test_ratio /= total
    
    n_samples = len(data)
    indices = np.arange(n_samples)
    
    if shuffle:
        # 셔플
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # 인덱스 분할
    train_end = int(train_ratio * n_samples)
    valid_end = train_end + int(valid_ratio * n_samples)
    
    train_indices = indices[:train_end]
    valid_indices = indices[train_end:valid_end]
    test_indices = indices[valid_end:]
    
    # 데이터 분할
    train_data = data[train_indices]
    valid_data = data[valid_indices]
    test_data = data[test_indices]
    
    result = {
        'train_data': train_data,
        'valid_data': valid_data,
        'test_data': test_data
    }
    
    # 레이블이 있으면 분할
    if labels is not None:
        train_labels = labels[train_indices]
        valid_labels = labels[valid_indices]
        test_labels = labels[test_indices]
        
        result.update({
            'train_labels': train_labels,
            'valid_labels': valid_labels,
            'test_labels': test_labels
        })
    
    logger.info(f"데이터셋 분할 완료: 학습={len(train_data)}, 검증={len(valid_data)}, 테스트={len(test_data)}")
    return result