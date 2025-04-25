"""
파이프라인 모듈 패키지

센서 데이터 처리 및 모델 학습을 위한 파이프라인 모듈을 포함합니다.
"""

# 파이프라인 모듈 등록
from src.pipelines.sensor_classification import run_training_pipeline

__all__ = ['run_training_pipeline']