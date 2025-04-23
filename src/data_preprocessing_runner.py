# src/data_preprocessing_runner.py

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import SensorDataPreprocessor

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_training_data(
    input_path: str, 
    output_path: str, 
    target_column: str = 'target_quality'
):
    """
    모델 학습을 위한 데이터 준비 및 변환
    
    Args:
        input_path (str): 원본 데이터 경로
        output_path (str): 변환된 데이터 저장 경로
        target_column (str): 타겟 컬럼명
    """
    try:
        # 데이터 전처리기 초기화
        preprocessor = SensorDataPreprocessor(
            window_size=15  # 적절한 윈도우 크기 설정
        )
        
        # 데이터 로드
        logger.info(f"데이터 로드: {input_path}")
        # 원본 데이터를 적절한 포맷으로 로드
        import pandas as pd
        raw_data = pd.read_csv(input_path)
        
        # 센서 데이터 형식으로 변환 (단일 센서 데이터 형태로 가정)
        sensor_data = {'sensor1': raw_data}
        
        # 데이터 보간 및 정제
        logger.info("데이터 보간 및 정제 시작")
        interpolated_data = preprocessor.interpolate_sensor_data(
            sensor_data,
            step=0.001,  # 적절한 시간 간격 설정
            kind='linear'
        )
        
        # 이동 평균 및 특성 추출
        cleaned_data = preprocessor.apply_moving_average(
            interpolated_data['sensor1'],
            columns=[col for col in interpolated_data['sensor1'].columns if col != 'time'],
            window_size=15
        )
        
        # 필요하다면 추가 특성 추출
        enhanced_data = preprocessor.extract_statistical_moments(
            cleaned_data, 
            columns=[col for col in cleaned_data.columns if col != 'time' and col != target_column],
            window_size=100
        )
        
        # 변환된 데이터 저장
        logger.info(f"변환된 데이터 저장: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 데이터를 CSV로 저장
        enhanced_data.to_csv(output_path, index=False)
        
        logger.info("데이터 변환 완료")
        return enhanced_data
    
    except Exception as e:
        logger.error(f"데이터 변환 중 오류 발생: {e}")
        raise

def main():
    # 기본 경로 설정
    input_data_path = os.path.join(PROJECT_ROOT, 'data', 'raw', 'sensor_data.csv')
    output_data_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'training_data.csv')
    
    # 데이터 변환 실행
    prepare_training_data(
        input_path=input_data_path, 
        output_path=output_data_path
    )

if __name__ == "__main__":
    main()