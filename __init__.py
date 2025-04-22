# project_root/__init__.py

"""
프로젝트 루트 디렉토리를 위한 초기화 스크립트
모듈 경로 및 기본 설정 제공
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 디렉토리 절대 경로 설정
PROJECT_ROOT = Path(__file__).parent.absolute()

# Python 경로에 프로젝트 루트 추가
sys.path.insert(0, str(PROJECT_ROOT))

# 기본 디렉토리 경로 상수
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# 필요한 디렉토리 자동 생성
def create_project_directories():
    """프로젝트에 필요한 기본 디렉토리 생성"""
    for directory in [MODELS_DIR, DATA_DIR, CONFIG_DIR, LOGS_DIR]:
        os.makedirs(directory, exist_ok=True)

# 모듈 로드 시 디렉토리 생성
create_project_directories()