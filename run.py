#!/usr/bin/env python
"""
센서 데이터 분류 시스템 실행 스크립트

이 스크립트는 다중 센서 데이터 분류 시스템의 CLI 인터페이스를 실행합니다.
"""

import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 필요한 디렉토리 생성
for dir_name in ['data/raw', 'data/processed', 'models', 'plots', 'logs', 'deployment']:
    os.makedirs(os.path.join(project_root, dir_name), exist_ok=True)

# CLI 모듈 실행
from src.cli.cli import main
def main():
    print("MLOps 도구 선택:")
    print("1. 센서 데이터 분석")
    print("2. 이미지 데이터 분석")
    print("3. 텍스트 데이터 분석")
    print("4. 데이터베이스에서 CSV 추출")
    
    choice = input("옵션을 선택하세요 (1-4): ")
    
    if choice == "1":
        from src.cli.sensor_cli import main as sensor_main
        sensor_main()
    elif choice == "2":
        from src.cli.image_cli import main as image_main
        image_main()
    elif choice == "3":
        from src.cli.text_cli import main as text_main
        text_main()
    elif choice == "4":
        from src.cli.db_export_cli import main as db_export_main
        db_export_main()
    else:
        print("잘못된 선택입니다.")
if __name__ == "__main__":
    main()