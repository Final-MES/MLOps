# src/utils/logging.py - 로깅 설정 유틸리티
import logging
from pathlib import Path

def setup_logging(
    level: int = logging.INFO,
    log_file: str = None,
    log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
) -> logging.Logger:
    """애플리케이션 로깅 설정"""
    
    # 로그 디렉토리 생성
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # 로거 설정
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # 포맷터 설정
    formatter = logging.Formatter(log_format)
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (지정된 경우)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger