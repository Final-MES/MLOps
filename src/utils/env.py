"""
환경 변수 로드 및 관리 모듈

이 모듈은 .env 파일에서 환경 변수를 로드하고, 
애플리케이션 전체에서 사용할 수 있는 설정을 제공합니다.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_env_path() -> Path:
    """
    환경에 적합한 .env 파일 경로를 반환합니다.
    
    Returns:
        Path: .env 파일 경로
    """
    # 프로젝트 루트 디렉토리 확인
    root_dir = Path(__file__).parent.parent.parent
    
    # 환경에 따른 파일 선택
    env = os.getenv('APP_ENV', 'development')
    
    # 환경별 .env 파일 경로
    env_file_map = {
        'development': root_dir / 'config' / '.env.development',
        'testing': root_dir / 'config' / '.env.testing',
        'production': root_dir / 'config' / '.env.production',
    }
    
    # 환경별 파일이 있으면 사용, 없으면 기본 .env 파일 사용
    env_file = env_file_map.get(env, root_dir / 'config' / '.env')
    
    # 파일이 존재하는지 확인
    if not env_file.exists():
        logger.warning(f"환경 파일이 존재하지 않습니다: {env_file}")
        logger.info(f"기본 .env 파일을 사용합니다: {root_dir / 'config' / '.env'}")
        env_file = root_dir / 'config' / '.env'
    
    return env_file

def load_env_vars() -> Dict[str, str]:
    """
    환경 변수를 로드하고 딕셔너리로 반환합니다.
    
    Returns:
        Dict[str, str]: 환경 변수 딕셔너리
    """
    env_path = get_env_path()
    logger.info(f"환경 파일 로드 중: {env_path}")
    
    # .env 파일 로드
    load_dotenv(dotenv_path=env_path)
    
    # 모든 환경 변수를 딕셔너리로 수집
    env_vars = {key: value for key, value in os.environ.items()}
    
    return env_vars

def get_env(key: str, default: Optional[Any] = None) -> Any:
    """
    환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default (Any, optional): 기본값
        
    Returns:
        Any: 환경 변수 값 또는 기본값
    """
    return os.getenv(key, default)

def get_boolean_env(key: str, default: bool = False) -> bool:
    """
    불리언 환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default (bool, optional): 기본값
        
    Returns:
        bool: 환경 변수 값 또는 기본값
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 't', 'y', 'yes')

def get_int_env(key: str, default: int = 0) -> int:
    """
    정수 환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default (int, optional): 기본값
        
    Returns:
        int: 환경 변수 값 또는 기본값
    """
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"환경 변수 {key}를 정수로 변환할 수 없습니다. 기본값 {default}를 사용합니다.")
        return default

def get_float_env(key: str, default: float = 0.0) -> float:
    """
    부동 소수점 환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default (float, optional): 기본값
        
    Returns:
        float: 환경 변수 값 또는 기본값
    """
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        logger.warning(f"환경 변수 {key}를 실수로 변환할 수 없습니다. 기본값 {default}를 사용합니다.")
        return default

def get_list_env(key: str, default: Optional[list] = None, separator: str = ',') -> list:
    """
    리스트 환경 변수 값을 가져옵니다.
    
    Args:
        key (str): 환경 변수 키
        default (list, optional): 기본값
        separator (str, optional): 구분자
        
    Returns:
        list: 환경 변수 값 또는 기본값
    """
    if default is None:
        default = []
        
    value = os.getenv(key)
    if not value:
        return default
        
    return [item.strip() for item in value.split(separator)]

# 환경 변수 자동 로드
load_env_vars()

# 현재 환경 정보
ENV = get_env('APP_ENV', 'development')
DEBUG = get_boolean_env('DEBUG', False)
LOG_LEVEL = get_env('LOG_LEVEL', 'INFO')

# 애플리케이션 경로 정보
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(get_env('DATA_DIR', str(BASE_DIR / 'data')))
MODEL_DIR = Path(get_env('MODEL_DIR', str(BASE_DIR / 'models/trained')))
LOG_DIR = Path(get_env('LOG_DIR', str(BASE_DIR / 'logs')))

# 애플리케이션 설정 정보
APP_NAME = get_env('APP_NAME', 'smart-factory-mlops')
API_HOST = get_env('API_HOST', '0.0.0.0')
API_PORT = get_int_env('API_PORT', 8000)
METRICS_PORT = get_int_env('METRICS_PORT', 8001)

# MLflow 설정
MLFLOW_TRACKING_URI = get_env('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
MLFLOW_EXPERIMENT_NAME = get_env('MLFLOW_EXPERIMENT_NAME', 'smart_factory_lstm')
USE_MLFLOW = get_boolean_env('USE_MLFLOW', True)

# 모델 학습 설정
BATCH_SIZE = get_int_env('BATCH_SIZE', 32)
LEARNING_RATE = get_float_env('LEARNING_RATE', 0.001)
EPOCHS = get_int_env('EPOCHS', 100)
PATIENCE = get_int_env('PATIENCE', 10)