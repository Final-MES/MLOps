"""
데이터베이스 유틸리티 모듈

이 모듈은 다양한 데이터베이스 시스템(MySQL, PostgreSQL, SQLite 등)과의 
연결 및 SQL 쿼리 실행, 테이블 메타데이터 조회 등의 기능을 제공합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Tuple
# 로깅 설정
logger = logging.getLogger(__name__)

def get_db_connection(
    db_type: str,
    host: str = None,
    port: int = None,
    database: str = None,
    username: str = None,
    password: str = None,
    **kwargs
) -> Any:
    """
    지정된 유형의 데이터베이스에 대한 연결을 생성합니다.
    
    Args:
        db_type: 데이터베이스 유형 ('mysql', 'postgresql', 'sqlite', 'sqlserver', 'oracle')
        host: 호스트 이름 또는 IP 주소
        port: 포트 번호
        database: 데이터베이스 이름 또는 SQLite의 경우 파일 경로
        username: 사용자 이름
        password: 비밀번호
        **kwargs: 추가 연결 매개변수
        
    Returns:
        Any: 데이터베이스 연결 객체
        
    Raises:
        ImportError: 필요한 데이터베이스 드라이버가 설치되지 않은 경우
        Exception: 연결 중 오류 발생
    """
    db_type = db_type.lower()
    
    try:
        if db_type == 'mysql':
            import mysql.connector
            connection = mysql.connector.connect(
                host=host,
                port=port,
                database=database,
                user=username,
                password=password,
                **kwargs
            )
            logger.info(f"MySQL 데이터베이스 '{database}'에 연결되었습니다")
            return connection
            
        elif db_type == 'postgresql':
            import psycopg2
            connection = psycopg2.connect(
                host=host,
                port=port,
                dbname=database,
                user=username,
                password=password,
                **kwargs
            )
            logger.info(f"PostgreSQL 데이터베이스 '{database}'에 연결되었습니다")
            return connection
            
        elif db_type == 'sqlite':
            import sqlite3
            connection = sqlite3.connect(database, **kwargs)
            logger.info(f"SQLite 데이터베이스 '{database}'에 연결되었습니다")
            return connection
            
        elif db_type == 'sqlserver':
            import pyodbc
            connection_string = (
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={host},{port};"
                f"DATABASE={database};"
                f"UID={username};"
                f"PWD={password}"
            )
            connection = pyodbc.connect(connection_string, **kwargs)
            logger.info(f"SQL Server 데이터베이스 '{database}'에 연결되었습니다")
            return connection
            
        elif db_type == 'oracle':
            import cx_Oracle
            dsn = cx_Oracle.makedsn(host, port, service_name=database)
            connection = cx_Oracle.connect(
                user=username,
                password=password,
                dsn=dsn,
                **kwargs
            )
            logger.info(f"Oracle 데이터베이스 '{database}'에 연결되었습니다")
            return connection
            
        else:
            raise ValueError(f"지원되지 않는 데이터베이스 유형: {db_type}")
            
    except ImportError as e:
        logger.error(f"{db_type} 데이터베이스 드라이버를 로드할 수 없습니다: {str(e)}")
        raise ImportError(
            f"{db_type} 데이터베이스를 사용하려면 적절한 드라이버가 필요합니다. "
            f"pip install mysql-connector-python (MySQL), "
            f"pip install psycopg2 (PostgreSQL), "
            f"pip install pyodbc (SQL Server), "
            f"pip install cx_Oracle (Oracle)"
        )
    except Exception as e:
        logger.error(f"데이터베이스 연결 중 오류 발생: {str(e)}")
        raise

def execute_query(connection: Any, query: str, params: tuple = None) -> Any:
    """
    SQL 쿼리를 실행합니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        query: 실행할 SQL 쿼리
        params: 쿼리 매개변수 (선택 사항)
        
    Returns:
        Any: 쿼리 결과 (있는 경우)
        
    Raises:
        Exception: 쿼리 실행 중 오류 발생
    """
    cursor = None
    try:
        cursor = connection.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
            
        # SELECT 쿼리인 경우 결과 반환
        if query.strip().upper().startswith("SELECT"):
            return cursor.fetchall()
        else:
            connection.commit()
            return cursor.rowcount  # 영향받은 행 수 반환
            
    except Exception as e:
        if connection.is_connected() if hasattr(connection, 'is_connected') else True:
            connection.rollback()  # 트랜잭션 롤백
        logger.error(f"쿼리 실행 중 오류 발생: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()

def get_available_tables(connection: Any, db_type: str) -> List[str]:
    """
    데이터베이스에서 사용 가능한 테이블 목록을 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        db_type: 데이터베이스 유형
        
    Returns:
        List[str]: 테이블 이름 목록
    """
    db_type = db_type.lower()
    cursor = None
    
    try:
        cursor = connection.cursor()
        
        if db_type == 'mysql':
            cursor.execute("SHOW TABLES")
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'postgresql':
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public'"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'sqlite':
            cursor.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'sqlserver':
            cursor.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_type = 'BASE TABLE'"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'oracle':
            cursor.execute(
                "SELECT table_name FROM user_tables "
                "ORDER BY table_name"
            )
            return [row[0] for row in cursor.fetchall()]
            
        else:
            raise ValueError(f"지원되지 않는 데이터베이스 유형: {db_type}")
            
    except Exception as e:
        logger.error(f"테이블 목록 조회 중 오류 발생: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()

def get_table_schema(connection: Any, table_name: str, db_type: str) -> List[Dict[str, str]]:
    """
    지정된 테이블의 스키마 정보를 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        table_name: 테이블 이름
        db_type: 데이터베이스 유형
        
    Returns:
        List[Dict[str, str]]: 컬럼 정보 목록 (이름, 유형, 기타 속성)
    """
    db_type = db_type.lower()
    cursor = None
    
    try:
        cursor = connection.cursor()
        
        if db_type == 'mysql':
            cursor.execute(f"DESCRIBE {table_name}")
            schema = []
            for row in cursor.fetchall():
                schema.append({
                    'name': row[0],
                    'type': row[1],
                    'extra': f"{'PK' if row[3] == 'PRI' else ''} {'NN' if row[2] == 'NO' else ''} {row[5] if row[5] else ''}"
                })
            return schema
            
        elif db_type == 'postgresql':
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
                    'extra': f"{'NN' if row[2] == 'NO' else ''} {'DEFAULT: ' + row[3] if row[3] else ''}"
                })
            return schema
            
        elif db_type == 'sqlite':
            cursor.execute(f"PRAGMA table_info({table_name})")
            schema = []
            for row in cursor.fetchall():
                schema.append({
                    'name': row[1],
                    'type': row[2],
                    'extra': f"{'PK' if row[5] == 1 else ''} {'NN' if row[3] == 1 else ''} {'DEFAULT: ' + row[4] if row[4] else ''}"
                })
            return schema
            
        elif db_type == 'sqlserver':
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
                    'extra': f"{'NN' if row[2] == 'NO' else ''} {'DEFAULT: ' + row[3] if row[3] else ''}"
                })
            return schema
            
        elif db_type == 'oracle':
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
                    'extra': f"{'NN' if row[2] == 'N' else ''} {'DEFAULT: ' + row[3] if row[3] else ''}"
                })
            return schema
            
        else:
            raise ValueError(f"지원되지 않는 데이터베이스 유형: {db_type}")
            
    except Exception as e:
        logger.error(f"테이블 스키마 조회 중 오류 발생: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()

def get_row_count(connection: Any, table_name: str) -> int:
    """
    지정된 테이블의 행 수를 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        table_name: 테이블 이름
        
    Returns:
        int: 테이블의 행 수
    """
    cursor = None
    
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        result = cursor.fetchone()
        return result[0]
    except Exception as e:
        logger.error(f"행 수 조회 중 오류 발생: {str(e)}")
        # 행 수를 조회할 수 없으면 -1 반환
        return -1
    finally:
        if cursor:
            cursor.close()

def get_sample_data(connection: Any, table_name: str, limit: int = 10) -> List[tuple]:
    """
    지정된 테이블에서 샘플 데이터를 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        table_name: 테이블 이름
        limit: 가져올 행 수
        
    Returns:
        List[tuple]: 샘플 데이터 행
    """
    cursor = None
    
    try:
        cursor = connection.cursor()
        cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
        return cursor.fetchall()
    except Exception as e:
        logger.error(f"샘플 데이터 조회 중 오류 발생: {str(e)}")
        raise
    finally:
        if cursor:
            cursor.close()

def get_primary_key(connection: Any, table_name: str, db_type: str) -> List[str]:
    """
    지정된 테이블의 기본 키 컬럼 이름을 가져옵니다.
    
    Args:
        connection: 데이터베이스 연결 객체
        table_name: 테이블 이름
        db_type: 데이터베이스 유형
        
    Returns:
        List[str]: 기본 키 컬럼 이름 목록
    """
    db_type = db_type.lower()
    cursor = None
    
    try:
        cursor = connection.cursor()
        
        if db_type == 'mysql':
            cursor.execute(
                "SELECT column_name FROM information_schema.key_column_usage "
                f"WHERE table_name = '{table_name}' AND constraint_name = 'PRIMARY'"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'postgresql':
            cursor.execute(
                "SELECT a.attname "
                "FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                f"WHERE i.indrelid = '{table_name}'::regclass AND i.indisprimary"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'sqlite':
            cursor.execute(f"PRAGMA table_info({table_name})")
            return [row[1] for row in cursor.fetchall() if row[5] == 1]  # 5번 인덱스가 pk 여부
            
        elif db_type == 'sqlserver':
            cursor.execute(
                "SELECT column_name "
                "FROM information_schema.key_column_usage "
                f"WHERE objectproperty(object_id(constraint_name), 'IsPrimaryKey') = 1 "
                f"AND table_name = '{table_name}'"
            )
            return [row[0] for row in cursor.fetchall()]
            
        elif db_type == 'oracle':
            cursor.execute(
                "SELECT cols.column_name "
                "FROM all_constraints cons, all_cons_columns cols "
                "WHERE cols.table_name = :1 "
                "AND cons.constraint_type = 'P' "
                "AND cons.constraint_name = cols.constraint_name "
                "AND cons.owner = cols.owner "
                "ORDER BY cols.position",
                (table_name.upper(),)
            )
            return [row[0] for row in cursor.fetchall()]
            
        else:
            raise ValueError(f"지원되지 않는 데이터베이스 유형: {db_type}")
            
    except Exception as e:
        logger.error(f"기본 키 조회 중 오류 발생: {str(e)}")
        return []  # 오류 발생 시 빈 목록 반환
    finally:
        if cursor:
            cursor.close()

# 사용 예시
if __name__ == "__main__":
    # 기본적인 사용 예시
    # SQLite 예시
    try:
        sqlite_conn = get_db_connection('sqlite', database=':memory:')
        execute_query(sqlite_conn, 'CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)')
        execute_query(sqlite_conn, "INSERT INTO test VALUES (1, 'Test1'), (2, 'Test2')")
        
        tables = get_available_tables(sqlite_conn, 'sqlite')
        print(f"테이블: {tables}")
        
        schema = get_table_schema(sqlite_conn, 'test', 'sqlite')
        print(f"스키마: {schema}")
        
        count = get_row_count(sqlite_conn, 'test')
        print(f"행 수: {count}")
        
        sample = get_sample_data(sqlite_conn, 'test')
        print(f"샘플 데이터: {sample}")
        
    except Exception as e:
        print(f"오류 발생: {str(e)}")
        
    finally:
        if 'sqlite_conn' in locals() and sqlite_conn:
            sqlite_conn.close()