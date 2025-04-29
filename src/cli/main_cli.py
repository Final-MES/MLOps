#!/usr/bin/env python
"""
메인 CLI 모듈

이 모듈은 여러 CLI 모듈에 대한 진입점 역할을 합니다.
사용자는 이 메뉴를 통해 센서 데이터 분석, 데이터베이스 추출 등
다양한 기능을 선택할 수 있습니다.
"""

import os
import sys
import logging
from typing import Dict, List, Any, Optional

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기본 CLI 클래스 임포트
from src.cli.base_cli import BaseCLI

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

class MainCLI(BaseCLI):
    """
    메인 CLI 클래스
    
    여러 CLI 모듈에 대한 메인 인터페이스 역할을 합니다.
    """
    
    def __init__(self):
        """메인 CLI 초기화"""
        super().__init__(title="MLOps 도구")
        logger.info("메인 CLI 초기화 완료")
    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header()
            print("MLOps 도구 모음에 오신 것을 환영합니다.")
            print("아래 메뉴에서 사용할 도구를 선택하세요.\n")
            
            menu_options = [
                "센서 데이터 분석",
                "텍스트 데이터 분석",
                "이미지 데이터 분석",
                "데이터베이스에서 CSV 추출",
                "종료"
            ]
            
            choice = self.show_menu(menu_options, "메인 메뉴")
            
            if choice == 0:
                self.run_sensor_cli()
            elif choice == 1:
                self.run_text_cli()
            elif choice == 2:
                self.run_image_cli()
            elif choice == 3:
                self.run_db_export_cli()
            elif choice == 4:
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
    def run_sensor_cli(self) -> None:
        """센서 데이터 CLI 실행"""
        try:
            # 센서 CLI 모듈 임포트
            from src.cli.sensor_cli import SensorCLI
            
            # CLI 인스턴스 생성 및 실행
            sensor_cli = SensorCLI()
            sensor_cli.run()
            
        except ImportError:
            self.show_error("센서 데이터 CLI 모듈을 로드할 수 없습니다.")
            logger.exception("센서 CLI 모듈 로드 실패")
            self.wait_for_user()
        except Exception as e:
            self.show_error(f"센서 데이터 CLI 실행 중 오류 발생: {str(e)}")
            logger.exception("센서 CLI 실행 오류")
            self.wait_for_user()
    
    def run_db_export_cli(self) -> None:
        """데이터베이스 추출 CLI 실행"""
        try:
            # 데이터베이스 추출 CLI 모듈 임포트
            from src.cli.db_export_cli import DBExportCLI
            
            # CLI 인스턴스 생성 및 실행
            db_export_cli = DBExportCLI()
            db_export_cli.run()
            
        except ImportError:
            self.show_error("데이터베이스 추출 CLI 모듈을 로드할 수 없습니다.")
            logger.exception("DB 추출 CLI 모듈 로드 실패")
            self.wait_for_user()
        except Exception as e:
            self.show_error(f"데이터베이스 추출 CLI 실행 중 오류 발생: {str(e)}")
            logger.exception("DB 추출 CLI 실행 오류")
            self.wait_for_user()
    def run_text_cli(self) -> None:
        """센서 데이터 CLI 실행"""
        try:
            # 센서 CLI 모듈 임포트
            from src.cli.text_cli import TextCLI
            
            # CLI 인스턴스 생성 및 실행
            text_cli = TextCLI()
            text_cli.run()
            
        except ImportError:
            self.show_error("텍스트 데이터 CLI 모듈을 로드할 수 없습니다.")
            logger.exception("텍스트 CLI 모듈 로드 실패")
            self.wait_for_user()
        except Exception as e:
            self.show_error(f"텍스트 데이터 CLI 실행 중 오류 발생: {str(e)}")
            logger.exception("텍스트 CLI 실행 오류")
            self.wait_for_user()
    def run(self) -> None:
        """CLI 실행"""
        try:
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        except Exception as e:
            self.show_error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            logger.exception("예상치 못한 오류 발생")


def main():
    """메인 함수: CLI 실행"""
    try:
        # CLI 인스턴스 생성 및 실행
        cli = MainCLI()
        cli.run()
        
    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        print("로그 파일을 확인하세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())