"""
프로젝트 전역 설정 및 유틸리티 함수
"""
import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트 디렉토리 설정
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(PROJECT_ROOT, 'logs', 'app.log'))
    ]
)

# 전역 로거
logger = logging.getLogger(__name__)

# 디렉토리 경로 상수
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 디렉토리 자동 생성
def create_directories():
    """필요한 디렉토리 자동 생성"""
    for directory in [MODELS_DIR, DATA_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)

# 모듈 임포트 시 디렉토리 생성
create_directories()