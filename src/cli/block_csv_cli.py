#!/usr/bin/env python
"""
CSV 데이터 블럭 생성 CLI 모듈

이 모듈은 CSV 파일에서 데이터를 읽어 컬럼별로 블럭 형태로 가공하는
대화형 명령줄 인터페이스를 제공합니다.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import csv
from pathlib import Path


# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 데이터 블럭 생성 유틸리티 임포트
from src.utils.csv_block_generator import generate_column_blocks, generate_sequential_column_blocks

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

def block_csv_menu():
    """CSV 데이터 블럭 생성 메뉴"""
    print_header("CSV 데이터 블럭 생성")
    
    print("CSV 파일에서 데이터를 읽어 블럭 형태로 가공합니다.")
    print("각 컬럼별로 지정된 개수의 데이터를 연속적으로 붙여 데이터 블럭을 생성합니다.\n")
    
    # 기본 경로 설정
    default_input_path = os.path.join(project_root, "data", "vibrate", "g2_sensor1.csv")
    default_output_dir = os.path.join(project_root, "data", "blocks")
    
    # 입력 파일 경로 설정
    input_path = get_input("CSV 파일 경로", default_input_path)
    
    # 파일 존재 확인
    if not os.path.exists(input_path):
        print(f"⚠️ 경고: 파일 '{input_path}'이(가) 존재하지 않습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 출력 디렉토리 설정
    output_dir = get_input("출력 디렉토리 경로", default_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 블럭 크기 설정
    block_size = get_numeric_input("블럭 크기 (각 컬럼에서 가져올 데이터 개수)", 100, min_val=1)
    
    # 제외할 컬럼 설정
    exclude_first_column = get_yes_no_input("첫 번째 컬럼(시간)을 제외하시겠습니까?", default=False)
    exclude_columns = [0] if exclude_first_column else []
    
    # 추가로 제외할 컬럼 설정
    additional_exclude = get_input("추가로 제외할 컬럼 번호 (쉼표로 구분, 없으면 비워두세요)")
    if additional_exclude.strip():
        try:
            additional_cols = [int(col.strip()) for col in additional_exclude.split(',')]
            exclude_columns.extend(additional_cols)
            exclude_columns = sorted(list(set(exclude_columns)))  # 중복 제거 및 정렬
        except ValueError:
            print("⚠️ 경고: 잘못된 컬럼 번호 형식입니다. 추가 제외 컬럼을 무시합니다.")

    # 여러 블럭 생성 여부
    create_multiple = get_yes_no_input("여러 개의 순차적 블럭을 생성하시겠습니까?", default=True)
    num_blocks = 1
    
    if create_multiple:
        num_blocks = get_numeric_input("생성할 블럭 수", 25, min_val=1, max_val=100)
    
    # 처리 시작 확인
    print("\n입력 설정 요약:")
    print(f"- 입력 파일: {input_path}")
    print(f"- 출력 디렉토리: {output_dir}")
    print(f"- 블럭 크기: {block_size}")
    print(f"- 제외 컬럼: {exclude_columns}")
    print(f"- 블럭 수: {num_blocks}")
    
    proceed = get_yes_no_input("\n위 설정으로 블럭을 생성하시겠습니까?", default=True)
    
    if not proceed:
        print("작업을 취소합니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    try:
        # 파일 이름 추출 (경로와 확장자 제외)
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        
        # 단일 블럭 또는 다중 블럭 생성
        if create_multiple:
            print(f"\n{num_blocks}개의 순차적 블럭 생성 중...")
            
            blocks = generate_sequential_column_blocks(
                csv_path=input_path,
                block_size=block_size,
                exclude_columns=exclude_columns,
                num_blocks=num_blocks
            )
            
            # 결과 저장 (CSV 형식)
            output_path = os.path.join(output_dir, f"{file_name}_blocks_{block_size}x{num_blocks}.csv")
            
            # CSV 파일로 저장
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 헤더 추가 (블록 번호와 인덱스)
                header = ['block_id', 'index', 'value']
                
                # 각 블록의 데이터를 행으로 저장 (값만 저장)
                for block in blocks:
                    for value in block:
                        # 값만 저장 (리스트로 전달하면 CSV 형식으로 저장됨)
                        writer.writerow([value])
            
            print(f"✅ {len(blocks)}개의 블럭이 생성되어 '{output_path}'에 저장되었습니다.")

        else:
            print("\n데이터 블럭 생성 중...")
            
            block_data = generate_column_blocks(
                csv_path=input_path,
                block_size=block_size,
                exclude_columns=exclude_columns
            )
            
            # 결과 저장 (CSV 형식)
            output_path = os.path.join(output_dir, f"{file_name}_block_{block_size}.csv")
            
            # CSV 파일로 저장
            with open(output_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # 헤더 추가 (데이터 인덱스와 값)
                header = ['index', 'value']
                writer.writerow(header)
                
                # 데이터를 행으로 저장
                for i, value in enumerate(block_data):
                    writer.writerow([i, value])
            
            print(f"✅ 데이터 블럭이 생성되어 '{output_path}'에 저장되었습니다.")
            print(f"- 블럭 크기: {len(block_data)}")

            # CSV 파일 내용 확인
            print("CSV 파일이 생성되었습니다. 처음 몇 줄의 내용:")
            try:
                with open(output_path, 'r') as f:
                    for i, line in enumerate(f):
                        if i >= 5:  # 처음 5줄만 출력
                            break
                        print(line.strip())
                print("...")
            except Exception as e:
                print(f"CSV 파일 읽기 실패: {str(e)}")
                
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        logger.exception("데이터 블럭 생성 중 오류 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def main():
    """메인 함수: CLI 실행"""
    
    try:    
        # 대화형 메뉴 실행
        block_csv_menu()
        return 0
        
    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        print("로그 파일을 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())