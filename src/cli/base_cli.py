#!/usr/bin/env python
"""
기본 CLI 클래스 모듈

이 모듈은 모든 CLI 모듈의 기본 클래스를 제공합니다.
공통 기능과 인터페이스를 정의하여 코드 중복을 줄이고
일관된 사용자 경험을 보장합니다.
"""

import os
import sys
import logging
import argparse
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from pathlib import Path

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils.paths import get_project_paths, ensure_dir

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'cli.log'))
    ]
)
logger = logging.getLogger(__name__)

class BaseCLI(ABC):
    """
    모든 CLI 모듈을 위한 기본 클래스
    
    이 클래스는 다음과 같은 공통 기능을 제공합니다:
    - 화면 지우기 및 헤더 출력
    - 사용자 입력 받기 (문자열, 숫자, 예/아니오)
    - 상태 관리 및 출력
    - 디렉토리 관리
    """
    
    def __init__(self, title: str = "MLOps CLI"):
        """
        CLI 초기화
        
        Args:
            title: CLI 제목
        """
        self.title = title
        self.state = {}  # 상태 정보 저장
        self.paths = get_project_paths()  # 프로젝트 주요 경로
        
        # 필요한 디렉토리 생성
        for path in self.paths.values():
            ensure_dir(path)
        
        logger.info(f"{self.__class__.__name__} 초기화됨")
    
    def clear_screen(self) -> None:
        """화면 지우기"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self, subtitle: Optional[str] = None) -> None:
        """
        헤더 출력
        
        Args:
            subtitle: 부제목 (선택적)
        """
        self.clear_screen()
        print("=" * 60)
        print(f"{self.title:^60}")
        if subtitle:
            print(f"{subtitle:^60}")
        print("=" * 60)
        print()
    
    def get_input(self, prompt: str, default: Any = None) -> str:
        """
        사용자 입력 받기 (기본값 지원)
        
        Args:
            prompt: 사용자에게 표시할 프롬프트
            default: 기본값 (Enter 키만 누른 경우 사용)
            
        Returns:
            str: 사용자 입력값 또는 기본값
        """
        if default is not None:
            result = input(f"{prompt} [{default}]: ")
            return result if result.strip() else str(default)
        else:
            return input(f"{prompt}: ")
    
    def get_numeric_input(self, prompt: str, default: float, min_val: float = None, max_val: float = None) -> float:
        """
        숫자 입력 받기 (범위 검사 포함)
        
        Args:
            prompt: 사용자에게 표시할 프롬프트
            default: 기본값
            min_val: 최소값
            max_val: 최대값
            
        Returns:
            float: 사용자 입력값 또는 기본값
        """
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
    
    def get_yes_no_input(self, prompt: str, default: bool = True) -> bool:
        """
        예/아니오 입력 받기
        
        Args:
            prompt: 사용자에게 표시할 프롬프트
            default: 기본값
            
        Returns:
            bool: 사용자 선택 (True/False)
        """
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
    
    def print_status(self) -> None:
        """현재 상태 출력"""
        print("\n현재 상태:")
        print("-" * 40)
        
        # 상태 항목이 없는 경우
        if not self.state:
            print("아직 상태 정보가 없습니다.")
        else:
            # 상태 정보 출력
            for key, value in self.state.items():
                if value is not None:
                    print(f"✅ {key}: {value}")
                else:
                    print(f"❌ {key}: 없음")
        
        print("-" * 40)
    
    def update_state(self, key: str, value: Any) -> None:
        """
        상태 업데이트
        
        Args:
            key: 상태 키
            value: 상태 값
        """
        self.state[key] = value
        logger.debug(f"상태 업데이트: {key}={value}")
    
    def wait_for_user(self) -> None:
        """사용자가 계속하기를 기다림"""
        input("\n계속하려면 Enter 키를 누르세요...")
    
    def show_error(self, message: str) -> None:
        """
        에러 메시지 출력
        
        Args:
            message: 출력할 에러 메시지
        """
        print(f"\n❌ 오류: {message}")
        logger.error(message)
    
    def show_message(self, message: str) -> None:
        """
        일반 메시지 출력
    
        Args:
        message: 출력할 메시지
        """
        print(message)
        logger.info(message)
    
    def show_success(self, message: str) -> None:
        """
        성공 메시지 출력
        
        Args:
            message: 출력할 성공 메시지
        """
        print(f"\n✅ 성공: {message}")
        logger.info(message)
    
    
    def show_warning(self, message: str) -> None:
        """
        경고 메시지 출력
        
        Args:
            message: 출력할 경고 메시지
        """
        print(f"\n⚠️ 경고: {message}")
        logger.warning(message)
    
    def show_menu(self, options: List[str], title: str = "메뉴 선택") -> int:
        """
        메뉴 표시 및 사용자 선택 받기
        
        Args:
            options: 메뉴 옵션 목록
            title: 메뉴 제목
            
        Returns:
            int: 선택한 메뉴 인덱스 (0부터 시작)
        """
        print(f"\n{title}:")
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        
        while True:
            try:
                choice = input("\n선택 (1-{}): ".format(len(options)))
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(options):
                    return choice_idx
                else:
                    print(f"유효한 번호를 입력하세요 (1-{len(options)})")
            except ValueError:
                print("숫자를 입력하세요")
    
    @abstractmethod
    def main_menu(self) -> None:
        """
        메인 메뉴 표시 (하위 클래스에서 구현)
        """
        pass
    
    @abstractmethod
    def run(self) -> None:
        """
        CLI 실행 (하위 클래스에서 구현)
        """
        pass

if __name__ == "__main__":
    # 이 모듈은 직접 실행하지 않음
    print("이 모듈은 직접 실행할 수 없습니다.")
    print("run.py를 통해 실행하세요.")