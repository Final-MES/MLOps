# src/utils/paths.py - 경로 처리 유틸리티
from pathlib import Path
import os

# 프로젝트 루트 디렉토리
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()

def get_config_path() -> Path:
    """설정 파일 경로"""
    return PROJECT_ROOT / 'config'

def get_data_path(subdir: str = '') -> Path:
    """데이터 디렉토리 경로"""
    path = PROJECT_ROOT / 'data'
    if subdir:
        path = path / subdir
    return path

def get_model_path(model_name: str = '') -> Path:
    """모델 파일 경로"""
    path = PROJECT_ROOT / 'models'
    if model_name:
        path = path / f"{model_name}.pth"
    return path

def ensure_dir(path: Path) -> Path:
    """디렉토리 존재 확인 및 생성"""
    path.mkdir(parents=True, exist_ok=True)
    return path