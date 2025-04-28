"""
설정 파일 로더 모듈

이 모듈은 데이터베이스 연결 정보 등 설정 파일을 로드하는 기능을 제공합니다.
보안에 민감한 정보를 코드에서 분리하여 관리할 수 있게 합니다.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import getpass

# 로깅 설정
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """프로젝트 루트 디렉토리 경로 반환"""
    # 현재 파일의 디렉토리에서 시작하여 상위로 이동
    return Path(__file__).parent.parent.parent.absolute()

def load_db_config(profile: str = "default", config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    데이터베이스 연결 설정 로드
    
    Args:
        profile: 설정 프로필명 (default, development, production 등)
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        Dict[str, Any]: 데이터베이스 연결 설정
    """
    if config_path is None:
        # 기본 설정 파일 경로
        config_path = os.path.join(get_project_root(), "config", "db_config.json")
    
    try:
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 프로필 설정 추출
        if profile in config["connections"]:
            db_config = config["connections"][profile]
            
            # 비밀번호가 비어있으면 사용자에게 입력 요청
            if not db_config.get("password"):
                # 비밀번호가 필요한 데이터베이스 유형인 경우에만 요청
                # (SQLite는 비밀번호가 필요 없음)
                if db_config.get("username"):
                    print(f"\n'{profile}' 프로필의 데이터베이스 비밀번호를 입력하세요:")
                    db_config["password"] = getpass.getpass()
            
            logger.info(f"'{profile}' 프로필의 데이터베이스 설정을 로드했습니다.")
            return db_config
        else:
            logger.warning(f"'{profile}' 프로필이 설정 파일에 없습니다. 기본 설정을 사용합니다.")
            return config["connections"]["default"]
            
    except FileNotFoundError:
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        # 기본 설정 반환
        return {
            "host": "localhost",
            "port": {
                "mysql": 3306,
                "postgresql": 5432,
                "sqlserver": 1433,
                "oracle": 1521
            },
            "database": "",
            "username": "",
            "password": ""
        }
    except json.JSONDecodeError:
        logger.error(f"설정 파일 형식이 잘못되었습니다: {config_path}")
        # 기본 설정 반환
        return {
            "host": "localhost",
            "port": {
                "mysql": 3306,
                "postgresql": 5432,
                "sqlserver": 1433,
                "oracle": 1521
            },
            "database": "",
            "username": "",
            "password": ""
        }
    except Exception as e:
        logger.error(f"설정 파일 로드 중 오류 발생: {str(e)}")
        # 기본 설정 반환
        return {
            "host": "localhost",
            "port": {
                "mysql": 3306,
                "postgresql": 5432,
                "sqlserver": 1433,
                "oracle": 1521
            },
            "database": "",
            "username": "",
            "password": ""
        }

def load_export_settings(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    CSV 추출 설정 로드
    
    Args:
        config_path: 설정 파일 경로 (None이면 기본 경로 사용)
        
    Returns:
        Dict[str, Any]: CSV 추출 설정
    """
    if config_path is None:
        # 기본 설정 파일 경로
        config_path = os.path.join(get_project_root(), "config", "db_config.json")
    
    try:
        # 설정 파일 로드
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 추출 설정 추출
        if "export_settings" in config:
            export_settings = config["export_settings"]
            
            # 출력 경로는 Path 객체로 변환
            if "output_path" in export_settings:
                export_settings["output_path"] = Path(export_settings["output_path"])
                
                # 상대 경로인 경우 프로젝트 루트 기준으로 변환
                if not export_settings["output_path"].is_absolute():
                    export_settings["output_path"] = get_project_root() / export_settings["output_path"]
            
            logger.info("CSV 추출 설정을 로드했습니다.")
            return export_settings
        else:
            logger.warning(f"설정 파일에 추출 설정이 없습니다. 기본 설정을 사용합니다.")
            return {
                "csv_separator": ",",
                "include_header": True,
                "chunk_size": 10000,
                "output_path": Path(get_project_root()) / "data" / "raw" / "extracted"
            }
            
    except Exception as e:
        logger.error(f"추출 설정 로드 중 오류 발생: {str(e)}")
        # 기본 설정 반환
        return {
            "csv_separator": ",",
            "include_header": True,
            "chunk_size": 10000,
            "output_path": Path(get_project_root()) / "data" / "raw" / "extracted"
        }

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # DB 설정 로드 테스트
    db_config = load_db_config()
    print("DB 연결 설정:", db_config)
    
    # 추출 설정 로드 테스트
    export_settings = load_export_settings()
    print("추출 설정:", export_settings)