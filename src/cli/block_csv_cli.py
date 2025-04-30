#!/usr/bin/env python
"""
CSV 데이터 블럭 생성 CLI 모듈

이 모듈은 CSV 파일에서 데이터를 읽어 컬럼별로 블럭 형태로 가공하는
대화형 명령줄 인터페이스를 제공합니다.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import csv
from pathlib import Path


# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'block_csv.log'))
    ]
)
logger = logging.getLogger(__name__)

def clear_screen():
    """화면 지우기"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header(title):
    """헤더 출력"""
    clear_screen()
    print("=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print()

def get_input(prompt, default=None):
    """사용자 입력 받기 (기본값 지원)"""
    if default is not None:
        result = input(f"{prompt} [{default}]: ")
        return result if result.strip() else default
    else:
        return input(f"{prompt}: ")

def get_numeric_input(prompt, default, min_val=None, max_val=None):
    """숫자 입력 받기 (범위 검사 포함)"""
    while True:
        try:
            result = input(f"{prompt} [{default}]: ")
            if not result.strip():
                result = default
            else:
                result = float(result)
                
            # 정수형인 경우 변환
            if result == int(result):
                result = int(result)
            
            # 범위 검사
            if min_val is not None and result < min_val:
                print(f"값이 너무 작습니다. 최소값: {min_val}")
                continue
            if max_val is not None and result > max_val:
                print(f"값이 너무 큽니다. 최대값: {max_val}")
                continue
                
            return result
        except ValueError:
            print("유효한 숫자를 입력해주세요.")

def get_yes_no_input(prompt, default=True):
    """예/아니오 입력 받기"""
    default_str = "Y/n" if default else "y/N"
    while True:
        result = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not result:
            return default
        elif result in ['y', 'yes']:
            return True
        elif result in ['n', 'no']:
            return False
        else:
            print("'y' 또는 'n'을 입력해주세요.")

import os
import pandas as pd
import csv

def block_csv_menu():
    print_header("CSV 데이터 블럭 생성")

    default_input_path = os.path.join(project_root, "data", "vibrate", "g2_sensor1.csv")
    default_output_dir = os.path.join(project_root, "data", "blocks")

    input_path = get_input("CSV 파일 경로", default_input_path)
    if not os.path.exists(input_path):
        print(f"⚠️ 경고: 파일 '{input_path}'이(가) 존재하지 않습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return

    output_dir = get_input("출력 디렉토리 경로", default_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    block_size = get_numeric_input("블럭 크기", 100, min_val=1)
    num_blocks = get_numeric_input("블럭 수", 25, min_val=1)

    # 원본 데이터 로드
    df = pd.read_csv(input_path, header=None)
    print(f"원본 데이터의 첫 번째 값: {df.iloc[0, 1]}")  # 첫 번째 값 확인
    value_columns = df.iloc[:, 1:]  # 센서값만 (B~F열)

    blocks = []

    for col_index in range(value_columns.shape[1]):
        col = value_columns.iloc[:, col_index]
        for block_num in range(num_blocks):
            start_idx = block_num * block_size
            end_idx = start_idx + block_size

            # 데이터가 범위를 초과하면 종료
            if end_idx > len(col):
                break

            # 각 블록 데이터를 제대로 추가
            block = col[start_idx:end_idx].tolist()
            blocks.extend(block)  # 블록을 하나의 리스트로 추가

    # 시간 생성: 0.000736씩 증가
    total_rows = len(blocks)
    timestamps = [i * 0.000736 for i in range(total_rows)]

    # 출력 경로 설정
    output_path = os.path.join(output_dir, f"g2_sensor1_blocks_{block_size}x{num_blocks}.csv")

    # CSV 파일로 저장
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        # 각 값에 대해 타임스탬프와 데이터를 기록
        for i in range(total_rows):
            writer.writerow([timestamps[i], blocks[i]])

    print(f"✅ '{output_path}'에 저장되었습니다.")
    input("\n계속하려면 Enter 키를 누르세요...")

def main():
    """메인 함수: CLI 실행"""
    # 대화형 메뉴 실행
    block_csv_menu()
    return 0

if __name__ == "__main__":
    sys.exit(main())