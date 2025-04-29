"""
CSV 데이터 블럭 생성 유틸리티

이 모듈은 CSV 파일에서 여러 컬럼의 데이터를 읽어 블럭 형태로 구성합니다.
특정 개수의 연속된 데이터를 각 컬럼에서 추출하여 하나의 데이터 블럭으로 만듭니다.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import List, Optional, Union, Tuple, Dict, Any

# 로깅 설정
logger = logging.getLogger(__name__)

def generate_column_blocks(
    csv_path: str, 
    block_size: int = 100, 
    exclude_columns: Optional[List[int]] = None,
    column_names: Optional[List[str]] = None
) -> np.ndarray:
    """
    CSV 파일에서 데이터를 읽고 각 컬럼별로 블럭 형태로 구성합니다.
    
    Args:
        csv_path (str): CSV 파일 경로
        block_size (int): 각 컬럼에서 가져올 연속 데이터 개수
        exclude_columns (List[int], optional): 제외할 컬럼 인덱스 목록
        column_names (List[str], optional): 사용할 컬럼 이름 목록
        
    Returns:
        np.ndarray: 컬럼 블럭 형태로 구성된 데이터 배열
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_path):
            logger.error(f"파일이 존재하지 않습니다: {csv_path}")
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {csv_path}")
        
        # 기본값 설정
        if exclude_columns is None:
            exclude_columns = [0]  # 기본적으로 첫 번째 컬럼 제외
        
        # CSV 파일 읽기
        logger.info(f"CSV 파일 로드 중: {csv_path}")
        
        # column_names가 제공된 경우 이름으로 읽기, 아니면 헤더 없이 읽기
        if column_names:
            df = pd.read_csv(csv_path, names=column_names)
        else:
            df = pd.read_csv(csv_path, header=None)
        
        # 제외할 컬럼 제거
        included_columns = [i for i in range(len(df.columns)) if i not in exclude_columns]
        df_selected = df.iloc[:, included_columns]
        
        logger.info(f"데이터 로드 완료: {len(df)} 행, 선택된 컬럼 수: {len(df_selected.columns)}")
        
        # block_size 확인
        if block_size <= 0:
            logger.warning(f"block_size는 양수여야 합니다. 기본값 100으로 설정합니다.")
            block_size = 100
        
        # 컬럼별 블럭 생성
        columns_data = []
        column_count = len(df_selected.columns)
        
        for col_idx in range(column_count):
            col_data = df_selected.iloc[:block_size, col_idx].values
            columns_data.append(col_data)
        
        # 데이터를 연속적으로 붙여서 블럭 생성
        block_data = np.concatenate(columns_data)
        
        logger.info(f"데이터 블럭 생성 완료: 블럭 크기 {block_size}, 총 데이터 포인트 {len(block_data)}")
        return block_data
        
    except Exception as e:
        logger.error(f"데이터 블럭 생성 중 오류 발생: {str(e)}")
        raise
    
def generate_sequential_column_blocks(
    csv_path: str, 
    block_size: int = 100, 
    exclude_columns: Optional[List[int]] = None,
    column_names: Optional[List[str]] = None,
    num_blocks: int = 1
) -> List[np.ndarray]:
    """
    CSV 파일에서 여러 개의 순차적인 데이터 블럭을 생성합니다.
    
    Args:
        csv_path (str): CSV 파일 경로
        block_size (int): 각 컬럼에서 가져올 연속 데이터 개수
        exclude_columns (List[int], optional): 제외할 컬럼 인덱스 목록
        column_names (List[str], optional): 사용할 컬럼 이름 목록
        num_blocks (int): 생성할 블럭 수
        
    Returns:
        List[np.ndarray]: 생성된 데이터 블럭 목록
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(csv_path):
            logger.error(f"파일이 존재하지 않습니다: {csv_path}")
            raise FileNotFoundError(f"파일이 존재하지 않습니다: {csv_path}")
        
        # 기본값 설정
        if exclude_columns is None:
            exclude_columns = [0]  # 기본적으로 첫 번째 컬럼 제외
        
        # CSV 파일 읽기
        logger.info(f"CSV 파일 로드 중: {csv_path}")
        
        # column_names가 제공된 경우 이름으로 읽기, 아니면 헤더 없이 읽기
        if column_names:
            df = pd.read_csv(csv_path, names=column_names)
        else:
            df = pd.read_csv(csv_path, header=None)
        
        # 제외할 컬럼 제거
        included_columns = [i for i in range(len(df.columns)) if i not in exclude_columns]
        df_selected = df.iloc[:, included_columns]
        
        logger.info(f"데이터 로드 완료: {len(df)} 행, 선택된 컬럼 수: {len(df_selected.columns)}")
        
        # block_size 확인
        if block_size <= 0:
            logger.warning(f"block_size는 양수여야 합니다. 기본값 100으로 설정합니다.")
            block_size = 100
        
        # 여러 블럭 생성
        blocks = []
        column_count = len(df_selected.columns)
        
        for block_idx in range(num_blocks):
            start_row = block_idx * block_size
            
            # 데이터가 충분한지 확인
            if start_row + block_size > len(df):
                logger.warning(f"데이터가 부족하여 블럭 {block_idx+1}/{num_blocks}을 생성할 수 없습니다.")
                break
                
            # 블럭 생성
            block_columns_data = []
            for col_idx in range(column_count):
                col_data = df_selected.iloc[start_row:start_row+block_size, col_idx].values
                block_columns_data.append(col_data)
            
            # 데이터를 연속적으로 붙여서 블럭 생성
            block_data = np.concatenate(block_columns_data)
            blocks.append(block_data)
            
            logger.debug(f"블럭 {block_idx+1}/{num_blocks} 생성 완료")
        
        logger.info(f"총 {len(blocks)}개 데이터 블럭 생성 완료")
        return blocks
        
    except Exception as e:
        logger.error(f"데이터 블럭 생성 중 오류 발생: {str(e)}")
        raise

# 사용 예시
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 파일 경로 설정
    csv_path = "app/data/raw/g2_sensor1.csv"
    
    try:
        # 데이터 블럭 생성
        block_data = generate_column_blocks(
            csv_path=csv_path,
            block_size=100,
            exclude_columns=[0]  # 첫 번째 컬럼(시간) 제외
        )
        
        print(f"생성된 데이터 블럭의 크기: {block_data.shape}")
        print(f"첫 10개 데이터 포인트: {block_data[:10]}")
        
        # 여러 블럭 생성 예시
        blocks = generate_sequential_column_blocks(
            csv_path=csv_path,
            block_size=100,
            exclude_columns=[0],
            num_blocks=5
        )
        
        print(f"생성된 블럭 수: {len(blocks)}")
        for i, block in enumerate(blocks):
            print(f"블럭 {i+1} 크기: {block.shape}")
            
    except Exception as e:
        print(f"오류 발생: {str(e)}")