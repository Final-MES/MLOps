"""
데이터베이스 연결 및 기본 작업 모듈

이 모듈은 다양한 데이터베이스 시스템에 대한 연결 관리 및 기본 쿼리 실행 기능을 제공합니다.
재사용 가능한 DBConnector 클래스를 통해 모든 CLI 도구에서 일관된 DB 액세스를 제공합니다.
"""

import os
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# 설정 로더 임포트
from src.utils.config_loader import load_db_config

# 기존 데이터베이스 유틸리티 함수 활용
from src.utils.db_utils import (
    get_db_connection, execute_query, get_available_tables,
    get_table_schema, get_row_count
)

# 로깅 설정
logger = logging.getLogger(__name__)

class DBConnector:
    """
    데이터베이스 연결 및 상태 관리 클래스
    
    다양한 데이터베이스 시스템(MySQL, PostgreSQL, SQLite 등)에 대한
    연결 생성, 상태 관리, 기본 쿼리 실행 기능을 제공합니다.
    """
    
    def __init__(self, config_profile: str = "default"):
        """
        DBConnector 초기화
        
        Args:
            config_profile: 설정 프로필명 (default, development, production 등)
        """
        # 연결 정보 로드
        self.connection_params = load_db_config(config_profile)
        
        # 상태 초기화
        self.connection = None
        self.db_type = None
        self.config_profile = config_profile
        
        logger.info(f"DBConnector 초기화 완료 (프로필: {config_profile})")
    
    def connect(self, db_type: str, **kwargs) -> bool:
        """
        데이터베이스에 연결
        
        Args:
            db_type: 데이터베이스 유형 ('mysql', 'postgresql', 'sqlite', 'sqlserver', 'oracle')
            **kwargs: 연결 매개변수 (connection_params를 덮어쓰는 추가 파라미터)
            
        Returns:
            bool: 연결 성공 여부
        """
        # 기존 연결 닫기
        self.close()
        
        # DB 유형 처리
        db_type = db_type.lower()
        self.db_type = db_type
        
        # 연결 매개변수 준비
        params = self.connection_params.copy()
        
        # kwargs로 전달된 매개변수로 기본값 덮어쓰기
        for key, value in kwargs.items():
            params[key] = value
        
        # 포트 처리
        if 'port' in params and isinstance(params['port'], dict):
            if db_type in params['port']:
                params['port'] = params['port'][db_type]
            else:
                # 기본 포트 설정
                default_ports = {
                    'mysql': 3306, 'postgresql': 5432, 'sqlserver': 1433, 'oracle': 1521
                }
                params['port'] = default_ports.get(db_type, 3306)
        
        try:
            # SQLite는 별도 처리 (파일 경로만 필요)
            if db_type == 'sqlite':
                database_path = kwargs.get('database', params.get('database', ':memory:'))
                self.connection = get_db_connection(db_type=db_type, database=database_path)
            else:
                # 다른 DB 유형은 모든 연결 정보 필요
                self.connection = get_db_connection(
                    db_type=db_type,
                    host=params.get('host', 'localhost'),
                    port=params.get('port', None),
                    database=params.get('database', ''),
                    username=params.get('username', ''),
                    password=params.get('password', '')
                )
            
            logger.info(f"{db_type} 데이터베이스에 연결되었습니다.")
            return True
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {str(e)}")
            self.connection = None
            self.db_type = None
            return False
    
    def close(self) -> bool:
        """
        데이터베이스 연결 종료
        
        Returns:
            bool: 종료 성공 여부
        """
        if self.connection:
            try:
                self.connection.close()
                logger.info("데이터베이스 연결을 닫았습니다.")
                self.connection = None
                self.db_type = None
                return True
            except Exception as e:
                logger.error(f"데이터베이스 연결 종료 실패: {str(e)}")
                return False
        return True
    
    def is_connected(self) -> bool:
        """
        연결 상태 확인
        
        Returns:
            bool: 연결 상태
        """
        if not self.connection or not self.db_type:
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
            return False
    
    def execute_query(self, query: str, params: Optional[tuple] = None) -> Any:
        """
        쿼리 실행
        
        Args:
            query: 실행할 쿼리
            params: 쿼리 매개변수 (선택 사항)
            
        Returns:
            Any: 쿼리 결과 또는 영향받은 행 수
            
        Raises:
            ConnectionError: 연결이 없는 경우
            Exception: 쿼리 실행 실패
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
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
    
    def get_tables(self) -> List[str]:
        """
        테이블 목록 조회
        
        Returns:
            List[str]: 테이블 목록
            
        Raises:
            ConnectionError: 연결이 없는 경우
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
        return get_available_tables(self.connection, self.db_type)
    
    def get_schema(self, table_name: str) -> List[Dict[str, str]]:
        """
        테이블 스키마 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            List[Dict[str, str]]: 컬럼 정보 목록 (이름, 유형, 속성)
            
        Raises:
            ConnectionError: 연결이 없는 경우
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
        return get_table_schema(self.connection, table_name, self.db_type)
    
    def get_row_count(self, table_name: str) -> int:
        """
        테이블 행 수 조회
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            int: 테이블 행 수
            
        Raises:
            ConnectionError: 연결이 없는 경우
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
        return get_row_count(self.connection, table_name)
    
    def execute_script(self, script: str) -> bool:
        """
        SQL 스크립트 실행 (여러 쿼리 포함 가능)
        
        Args:
            script: 실행할 SQL 스크립트
            
        Returns:
            bool: 실행 성공 여부
            
        Raises:
            ConnectionError: 연결이 없는 경우
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
        # 세미콜론으로 스크립트 분리
        queries = [q.strip() for q in script.split(';') if q.strip()]
        
        try:
            for query in queries:
                self.execute_query(query)
            return True
        except Exception as e:
            logger.error(f"스크립트 실행 중 오류 발생: {str(e)}")
            return False
    
    def table_exists(self, table_name: str) -> bool:
        """
        테이블 존재 여부 확인
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            bool: 테이블 존재 여부
        """
        try:
            tables = self.get_tables()
            return table_name in tables
        except Exception:
            return False
    
    def create_table(self, table_name: str, columns: List[Dict[str, str]]) -> bool:
        """
        테이블 생성
        
        Args:
            table_name: 테이블 이름
            columns: 컬럼 정의 리스트 [{'name': 'col1', 'type': 'INT', 'constraint': 'PRIMARY KEY'}, ...]
            
        Returns:
            bool: 생성 성공 여부
        """
        if not self.is_connected():
            raise ConnectionError("데이터베이스에 연결되어 있지 않습니다.")
        
        try:
            # 컬럼 정의 생성
            column_defs = []
            for col in columns:
                col_def = f"{col['name']} {col['type']}"
                if 'constraint' in col and col['constraint']:
                    col_def += f" {col['constraint']}"
                column_defs.append(col_def)
            
            # SQL 작성
            create_sql = f"CREATE TABLE IF NOT EXISTS {table_name} (\n  "
            create_sql += ",\n  ".join(column_defs)
            create_sql += "\n)"
            
            # 테이블 생성
            self.execute_query(create_sql)
            logger.info(f"테이블 '{table_name}' 생성 완료")
            return True
            
        except Exception as e:
            logger.error(f"테이블 생성 중 오류 발생: {str(e)}")
            return False
    
    def insert_data(self, table_name: str, data: List[Dict[str, Any]], batch_size: int = 1000) -> int:
        """
        데이터 삽입
        
        Args:
            table_name: 테이블 이름
            data: 삽입할 데이터 [{col1: val1, col2: val2, ...}, ...]
            batch_size: 배치 크기
            
        Returns:
            int: 삽입된 행 수 (실패 시 -1)
        """
        if not self.is_connected() or not data:
            return -1
        
        try:
            total_inserted = 0
            
            # 데이터에서 컬럼 추출
            columns = list(data[0].keys())
            placeholders = ', '.join(['%s'] * len(columns))
            
            # 쿼리 준비
            query = f"INSERT INTO {table_name} ({', '.join(columns)}) VALUES ({placeholders})"
            
            # 배치 처리
            cursor = self.connection.cursor()
            
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                values = [[row[col] for col in columns] for row in batch]
                
                # 실행
                cursor.executemany(query, values)
                self.connection.commit()
                
                total_inserted += len(batch)
                logger.debug(f"{len(batch)}개 행 삽입 완료 (총 {total_inserted}개)")
            
            cursor.close()
            logger.info(f"총 {total_inserted}개 행이 '{table_name}' 테이블에 삽입되었습니다.")
            return total_inserted
            
        except Exception as e:
            logger.error(f"데이터 삽입 중 오류 발생: {str(e)}")
            if hasattr(self.connection, 'rollback'):
                self.connection.rollback()
            return -1
    
    def get_connection_params(self) -> Dict[str, Any]:
        """현재 연결 파라미터 반환"""
        return self.connection_params.copy()

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # DB 연결
    connector = DBConnector()