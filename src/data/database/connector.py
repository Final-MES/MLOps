"""
데이터베이스 연결 모듈

이 모듈은 다양한 데이터베이스 시스템에 대한 연결 관리 기능을 제공합니다.
각종 데이터베이스에 대한 연결을 생성하고, 이를 관리하는 인터페이스를 제공합니다.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import json
from pathlib import Path

# 기존 데이터베이스 유틸리티 함수 활용
from src.utils.db_utils import get_db_connection

# 로깅 설정
logger = logging.getLogger(__name__)

class DatabaseConnector:
    """
    데이터베이스 연결 관리 클래스
    
    다양한 데이터베이스 시스템과의 연결을 관리합니다.
    """
    
    def __init__(self, 
                db_type: str, 
                host: Optional[str] = None, 
                port: Optional[int] = None,
                database: Optional[str] = None, 
                username: Optional[str] = None, 
                password: Optional[str] = None,
                **kwargs):
        """
        DatabaseConnector 초기화
        
        Args:
            db_type: 데이터베이스 유형 ('mysql', 'postgresql', 'sqlite', 'sqlserver', 'oracle')
            host: 호스트 이름 또는 IP 주소 (SQLite 제외)
            port: 포트 번호 (SQLite 제외)
            database: 데이터베이스 이름 또는 SQLite의 경우 파일 경로
            username: 사용자 이름 (SQLite 제외)
            password: 비밀번호 (SQLite 제외)
            **kwargs: 추가 연결 매개변수
        """
        self.db_type = db_type.lower()
        self.host = host
        self.port = port
        self.database = database
        self.username = username
        self.password = password
        self.kwargs = kwargs
        self.connection = None
        
        # 연결 상태
        self.is_connected = False
    
    def connect(self) -> bool:
        """
        데이터베이스에 연결
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.connection = get_db_connection(
                db_type=self.db_type,
                host=self.host,
                port=self.port,
                database=self.database,
                username=self.username,
                password=self.password,
                **self.kwargs
            )
            self.is_connected = True
            logger.info(f"{self.db_type} 데이터베이스에 연결 성공")
            return True
        except Exception as e:
            self.is_connected = False
            logger.error(f"데이터베이스 연결 실패: {str(e)}")
            return False
    
    def disconnect(self) -> bool:
        """
        데이터베이스 연결 종료
        
        Returns:
            bool: 연결 종료 성공 여부
        """
        if self.connection:
            try:
                self.connection.close()
                self.is_connected = False
                logger.info("데이터베이스 연결 종료")
                return True
            except Exception as e:
                logger.error(f"데이터베이스 연결 종료 실패: {str(e)}")
                return False
        return True
    
    def check_connection(self) -> bool:
        """
        연결 상태 확인
        
        Returns:
            bool: 연결 상태
        """
        if not self.is_connected or not self.connection:
            logger.warning("데이터베이스에 연결되어 있지 않습니다.")
            return False
        
        # DB 유형별 연결 상태 확인 방법
        try:
            if self.db_type == 'mysql':
                return self.connection.is_connected()
            elif self.db_type == 'postgresql':
                return not self.connection.closed
            elif self.db_type == 'sqlite':
                # SQLite는 간단한 쿼리로 확인
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                return True
            elif self.db_type == 'sqlserver':
                # 커서 생성으로 확인
                cursor = self.connection.cursor()
                cursor.close()
                return True
            elif self.db_type == 'oracle':
                # 간단한 쿼리로 확인
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1 FROM DUAL")
                cursor.close()
                return True
            else:
                return False
        except Exception:
            self.is_connected = False
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        쿼리 실행
        
        Args:
            query: 실행할 쿼리
            params: 쿼리 매개변수 (선택 사항)
            
        Returns:
            Any: 쿼리 결과 또는 영향받은 행 수
        """
        if not self.check_connection():
            if not self.connect():
                raise ConnectionError("데이터베이스에 연결할 수 없습니다.")
        
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
                
            # SELECT 쿼리인 경우 결과 반환
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                self.connection.commit()
                return cursor.rowcount  # 영향받은 행 수 반환
                
        except Exception as e:
            if hasattr(self.connection, 'rollback'):
                self.connection.rollback()  # 트랜잭션 롤백
            logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
            raise
        finally:
            if cursor:
                cursor.close()
    
    def get_table_list(self) -> List[str]:
        """
        테이블 목록 조회
        
        Returns:
            List[str]: 테이블 목록
        """
        if not self.check_connection():
            if not self.connect():
                raise ConnectionError("데이터베이스에 연결할 수 없습니다.")
        
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            if self.db_type == 'mysql':
                cursor.execute("SHOW TABLES")
                return [row[0] for row in cursor.fetchall()]
                
            elif self.db_type == 'postgresql':
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public'"
                )
                return [row[0] for row in cursor.fetchall()]
                
            elif self.db_type == 'sqlite':
                cursor.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                return [row[0] for row in cursor.fetchall()]
                
            elif self.db_type == 'sqlserver':
                cursor.execute(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_type = 'BASE TABLE'"
                )
                return [row[0] for row in cursor.fetchall()]
                
            elif self.db_type == 'oracle':
                cursor.execute(
                    "SELECT table_name FROM user_tables "
                    "ORDER BY table_name"
                )
                return [row[0] for row in cursor.fetchall()]
                
            else:
                logger.error(f"지원하지 않는 데이터베이스 유형: {self.db_type}")
                return []
                
        except Exception as e:
            logger.error(f"테이블 목록 조회 중 오류 발생: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def get_table_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        테이블 스키마 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            List[Dict[str, str]]: 컬럼 정보 목록 (이름, 유형, 속성)
        """
        if not self.check_connection():
            if not self.connect():
                raise ConnectionError("데이터베이스에 연결할 수 없습니다.")
        
        cursor = None
        try:
            cursor = self.connection.cursor()
            
            if self.db_type == 'mysql':
                cursor.execute(f"DESCRIBE {table_name}")
                schema = []
                for row in cursor.fetchall():
                    schema.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': 'YES' if row[2] == 'YES' else 'NO',
                        'key': row[3],
                        'default': row[4],
                        'extra': row[5]
                    })
                return schema
                
            elif self.db_type == 'postgresql':
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    f"WHERE table_name = '{table_name}' "
                    "ORDER BY ordinal_position"
                )
                schema = []
                for row in cursor.fetchall():
                    schema.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'default': row[3]
                    })
                return schema
                
            elif self.db_type == 'sqlite':
                cursor.execute(f"PRAGMA table_info({table_name})")
                schema = []
                for row in cursor.fetchall():
                    schema.append({
                        'name': row[1],
                        'type': row[2],
                        'nullable': 'NO' if row[3] else 'YES',
                        'default': row[4],
                        'pk': 'YES' if row[5] else 'NO'
                    })
                return schema
                
            elif self.db_type == 'sqlserver':
                cursor.execute(
                    "SELECT column_name, data_type, is_nullable, column_default "
                    "FROM information_schema.columns "
                    f"WHERE table_name = '{table_name}' "
                    "ORDER BY ordinal_position"
                )
                schema = []
                for row in cursor.fetchall():
                    schema.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'default': row[3]
                    })
                return schema
                
            elif self.db_type == 'oracle':
                cursor.execute(
                    "SELECT column_name, data_type, nullable, data_default "
                    "FROM user_tab_columns "
                    f"WHERE table_name = '{table_name.upper()}' "
                    "ORDER BY column_id"
                )
                schema = []
                for row in cursor.fetchall():
                    schema.append({
                        'name': row[0],
                        'type': row[1],
                        'nullable': row[2],
                        'default': row[3]
                    })
                return schema
                
            else:
                logger.error(f"지원하지 않는 데이터베이스 유형: {self.db_type}")
                return []
                
        except Exception as e:
            logger.error(f"테이블 스키마 조회 중 오류 발생: {str(e)}")
            return []
        finally:
            if cursor:
                cursor.close()
    
    def save_connection_info(self, file_path: str) -> bool:
        """
        연결 정보를 파일로 저장
        
        Args:
            file_path: 저장할 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 비밀번호를 제외한 연결 정보 저장
            conn_info = {
                'db_type': self.db_type,
                'host': self.host,
                'port': self.port,
                'database': self.database,
                'username': self.username,
                # 'password': self.password  # 보안을 위해 저장하지 않음
            }
            
            with open(file_path, 'w') as f:
                json.dump(conn_info, f, indent=4)
                
            logger.info(f"연결 정보를 '{file_path}'에 저장했습니다.")
            return True
        except Exception as e:
            logger.error(f"연결 정보 저장 중 오류 발생: {str(e)}")
            return False
    
    @classmethod
    def load_connection_info(cls, file_path: str, password: str) -> 'DatabaseConnector':
        """
        파일에서 연결 정보를 로드하여 연결자 인스턴스 생성
        
        Args:
            file_path: 연결 정보 파일 경로
            password: 연결 비밀번호 (파일에 저장되지 않음)
            
        Returns:
            DatabaseConnector: 로드된 연결 정보로 생성된 인스턴스
        """
        try:
            with open(file_path, 'r') as f:
                conn_info = json.load(f)
            
            # 비밀번호 추가
            conn_info['password'] = password
            
            logger.info(f"파일 '{file_path}'에서 연결 정보를 로드했습니다.")
            
            # 새 인스턴스 생성
            return cls(**conn_info)
        except Exception as e:
            logger.error(f"연결 정보 로드 중 오류 발생: {str(e)}")
            raise

