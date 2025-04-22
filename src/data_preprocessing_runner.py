# src/data_preprocessing_runner.py

import os
import sys
import logging
from pathlib import Path

# 프로젝트 루트 경로 설정
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import DataPreprocessor

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
        preprocessor = DataPreprocessor(
            scaling_method='minmax', 
            imputation_strategy='mean'
        )
        
        # 데이터 로드
        logger.info(f"데이터 로드: {input_path}")
        raw_data = preprocessor.load_data(input_path)
        
        # 데이터 정제
        logger.info("데이터 정제 시작")
        cleaned_data = preprocessor.clean_data(
            raw_data, 
            drop_columns=['id', 'timestamp'],  # 필요에 따라 수정
            fill_na=True
        )
        
        # 데이터 스케일링
        logger.info("데이터 스케일링")
        scaled_data = preprocessor.scale_data(
            cleaned_data, 
            target_column=target_column
        )
        
        # 변환된 데이터 저장
        logger.info(f"변환된 데이터 저장: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 스케일링된 데이터를 CSV로 저장
        import pandas as pd
        
        scaled_df = pd.DataFrame(
            scaled_data['features'], 
            columns=scaled_data['feature_columns']
        )
        scaled_df[target_column] = scaled_data['target']
        
        scaled_df.to_csv(output_path, index=False)
        
        logger.info("데이터 변환 완료")
        return scaled_df
    
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