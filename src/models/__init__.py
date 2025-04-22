"""
모델 관련 유틸리티 및 전역 설정
"""
import os
import logging
from pathlib import Path

# 모델 관련 로거
logger = logging.getLogger(__name__)

# 모델 유틸리티 함수
def get_latest_model(models_dir):
    """
    최신 모델 파일 찾기
    
    Args:
        models_dir (str): 모델 디렉토리 경로
    
    Returns:
        str: 최신 모델 파일 경로
    """
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        logger.warning("No model files found")
        return None
    
    latest_model = max(model_files, key=lambda f: os.path.getctime(os.path.join(models_dir, f)))
    return os.path.join(models_dir, latest_model)

def validate_model_file(model_path):
    """
    모델 파일 유효성 검사
    
    Args:
        model_path (str): 모델 파일 경로
    
    Returns:
        bool: 모델 파일 유효성
    """
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return False
    
    # 추가 검증 로직 (파일 크기, 형식 등)
    file_size = os.path.getsize(model_path)
    if file_size == 0:
        logger.error(f"Model file is empty: {model_path}")
        return False
    
    return True