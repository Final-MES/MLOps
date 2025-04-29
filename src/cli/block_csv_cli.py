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
        create_example = get_yes_no_input("예제 파일을 생성하시겠습니까?", default=True)
        
        if create_example:
            # 예제 파일이 있는 디렉토리 생성
            os.makedirs(os.path.dirname(input_path), exist_ok=True)
            
            # 예제 데이터 생성
            time_col = np.arange(0, 1000, 0.1)  # 시간 데이터
            col_b = np.sin(time_col * 0.1)  # B 컬럼 (사인 파형)
            col_c = np.cos(time_col * 0.1)  # C 컬럼 (코사인 파형)
            col_d = np.sin(time_col * 0.05)  # D 컬럼 (저주파 사인 파형)
            col_e = np.random.normal(0, 0.5, size=len(time_col))  # E 컬럼 (랜덤 노이즈)
            
            # 데이터프레임 생성 및 저장
            df = pd.DataFrame({
                0: time_col, 
                1: col_b, 
                2: col_c, 
                3: col_d, 
                4: col_e
            })
            df.to_csv(input_path, index=False, header=False)
            
            print(f"✅ 예제 파일 '{input_path}'을(를) 생성했습니다.")
        else:
            print("❌ 파일이 없어 처리를 중단합니다.")
            input("\n계속하려면 Enter 키를 누르세요...")
            return
    
    # 출력 디렉토리 설정
    output_dir = get_input("출력 디렉토리 경로", default_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # 블럭 크기 설정
    block_size = get_numeric_input("블럭 크기 (각 컬럼에서 가져올 데이터 개수)", 100, min_val=1)
    
    # 제외할 컬럼 설정
    exclude_first_column = get_yes_no_input("첫 번째 컬럼(시간)을 제외하시겠습니까?", default=True)
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
    
    # 시각화 여부
    visualize = get_yes_no_input("생성된 블럭을 시각화하시겠습니까?", default=True)
    
    # 여러 블럭 생성 여부
    create_multiple = get_yes_no_input("여러 개의 순차적 블럭을 생성하시겠습니까?", default=False)
    num_blocks = 1
    
    if create_multiple:
        num_blocks = get_numeric_input("생성할 블럭 수", 5, min_val=1, max_val=100)
    
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
            
            # 결과 저장
            output_path = os.path.join(output_dir, f"{file_name}_blocks_{block_size}x{num_blocks}.npy")
            np.save(output_path, blocks)
            
            print(f"✅ {len(blocks)}개의 블럭이 생성되어 '{output_path}'에 저장되었습니다.")
            
            if visualize and len(blocks) > 0:
                visualize_blocks(blocks, exclude_columns, block_size, file_name)
                
        else:
            print("\n데이터 블럭 생성 중...")
            
            block_data = generate_column_blocks(
                csv_path=input_path,
                block_size=block_size,
                exclude_columns=exclude_columns
            )
            
            # 결과 저장
            output_path = os.path.join(output_dir, f"{file_name}_block_{block_size}.npy")
            np.save(output_path, block_data)
            
            print(f"✅ 데이터 블럭이 생성되어 '{output_path}'에 저장되었습니다.")
            print(f"- 블럭 크기: {len(block_data)}")
            
            if visualize:
                visualize_block(block_data, len(block_data) // block_size, block_size, file_name)
            
            # npy 파일 로드
            npy_data = np.load(output_path)

            # npy 파일 내용 출력
            print("npy 파일 내용:")
            print(npy_data)
                
    except Exception as e:
        print(f"❌ 오류 발생: {str(e)}")
        logger.exception("데이터 블럭 생성 중 오류 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def visualize_block(block_data, num_columns, block_size, file_name):
    """데이터 블럭 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        # 각 컬럼 데이터 시각화
        for i in range(num_columns):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            
            plt.subplot(num_columns, 1, i+1)
            plt.plot(block_data[start_idx:end_idx])
            plt.ylabel(f'Column {i+1}')
            plt.grid(True)
            
            if i == 0:
                plt.title(f'Block Visualization: {file_name}')
                
            if i == num_columns - 1:
                plt.xlabel('Sample Index')
        
        plt.tight_layout()
        
        # 시각화 결과 저장
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, f"{file_name}_block_visualization.png")
        plt.savefig(plot_path)
        
        print(f"📊 블럭 시각화 저장 완료: {plot_path}")
        
    except Exception as e:
        print(f"⚠️ 시각화 중 오류 발생: {str(e)}")
        logger.error(f"시각화 중 오류 발생: {str(e)}")

def visualize_blocks(blocks, exclude_columns, block_size, file_name):
    """여러 데이터 블럭 시각화"""
    try:
        import matplotlib.pyplot as plt
        
        # 첫 번째 블럭만 시각화
        first_block = blocks[0]
        num_columns = len(first_block) // block_size
        
        plt.figure(figsize=(12, 8))
        
        # 각 컬럼 데이터 시각화
        for i in range(num_columns):
            start_idx = i * block_size
            end_idx = start_idx + block_size
            
            plt.subplot(num_columns, 1, i+1)
            plt.plot(first_block[start_idx:end_idx])
            plt.ylabel(f'Column {i+1}')
            plt.grid(True)
            
            if i == 0:
                plt.title(f'First Block Visualization: {file_name} (Total: {len(blocks)} blocks)')
                
            if i == num_columns - 1:
                plt.xlabel('Sample Index')
        
        plt.tight_layout()
        
        # 시각화 결과 저장
        plots_dir = os.path.join(project_root, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_path = os.path.join(plots_dir, f"{file_name}_blocks_visualization.png")
        plt.savefig(plot_path)
        
        print(f"📊 블럭 시각화 저장 완료: {plot_path} (첫 번째 블럭만 시각화)")
        
    except Exception as e:
        print(f"⚠️ 시각화 중 오류 발생: {str(e)}")
        logger.error(f"시각화 중 오류 발생: {str(e)}")

def main():
    """메인 함수: CLI 실행"""
    try:
        # 명령줄 인자 파싱
        parser = argparse.ArgumentParser(description='CSV 데이터를 블럭 형태로 가공하는 도구')
        parser.add_argument('--file', type=str, default=None,
                          help='CSV 파일 경로 (기본값: data/vibrate/g2_sensor1.csv)')
        parser.add_argument('--output', type=str, default=None,
                          help='출력 디렉토리 (기본값: data/blocks)')
        parser.add_argument('--block-size', type=int, default=100,
                          help='블럭 크기 (기본값: 100)')
        parser.add_argument('--num-blocks', type=int, default=1,
                          help='생성할 블럭 수 (기본값: 1)')
        parser.add_argument('--exclude-first', action='store_true',
                          help='첫 번째 컬럼 제외 여부 (기본값: True)')
        parser.add_argument('--no-visualize', action='store_true',
                          help='시각화 생성하지 않음')
        
        args = parser.parse_args()
        
        # 명령줄 인자가 제공된 경우 직접 처리
        if args.file is not None:
            input_path = args.file
            output_dir = args.output or os.path.join(project_root, "data", "blocks")
            os.makedirs(output_dir, exist_ok=True)
            
            exclude_columns = [0] if args.exclude_first else []
            visualize = not args.no_visualize
            
            try:
                # 파일 이름 추출
                file_name = os.path.splitext(os.path.basename(input_path))[0]
                
                # 단일/다중 블럭 생성
                if args.num_blocks > 1:
                    blocks = generate_sequential_column_blocks(
                        csv_path=input_path,
                        block_size=args.block_size,
                        exclude_columns=exclude_columns,
                        num_blocks=args.num_blocks
                    )
                    
                    # 결과 저장
                    output_path = os.path.join(output_dir, f"{file_name}_blocks_{args.block_size}x{args.num_blocks}.npy")
                    np.save(output_path, blocks)
                    print(f"✅ {len(blocks)}개의 블럭이 생성되어 '{output_path}'에 저장되었습니다.")
                    
                    if visualize and len(blocks) > 0:
                        visualize_blocks(blocks, exclude_columns, args.block_size, file_name)
                    
                else:
                    block_data = generate_column_blocks(
                        csv_path=input_path,
                        block_size=args.block_size,
                        exclude_columns=exclude_columns
                    )
                    
                    # 결과 저장
                    output_path = os.path.join(output_dir, f"{file_name}_block_{args.block_size}.npy")
                    np.save(output_path, block_data)
                    print(f"✅ 데이터 블럭이 생성되어 '{output_path}'에 저장되었습니다.")
                    
                    if visualize:
                        num_columns = len(block_data) // args.block_size
                        visualize_block(block_data, num_columns, args.block_size, file_name)
                
                return 0
                
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")
                logger.exception("데이터 블럭 생성 중 오류 발생")
                return 1
        
        # 명령줄 인자가 없는 경우 대화형 메뉴 실행
        block_csv_menu()
        return 0
        
    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        print("로그 파일을 확인하세요.")
        return 1

if __name__ == "__main__":
    sys.exit(main())