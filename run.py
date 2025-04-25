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
from src.cli import main

if __name__ == "__main__":
    main()