"""
데이터베이스 데이터 추출 모듈

이 모듈은 데이터베이스에서 데이터를 추출하고 가공하는 기능을 제공합니다.
SQL 쿼리를 실행하고 그 결과를 다양한 형태로 변환하는 기능을 포함합니다.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import csv
import time
from datetime import datetime

# 데이터베이스 연결자 임포트
from src.data.database.connector import DatabaseConnector

# 로깅 설정
logger = logging.getLogger(__name__)

class DatabaseExtractor:
    """
    데이터베이스 데이터 추출 클래스
    
    이 클래스는 다음과 같은 기능을 제공합니다:
    - 테이블 또는 쿼리 결과를 DataFrame으로 변환
    - 데이터 청크 단위 처리 (메모리 효율적인 대용량 데이터 처리)
    - 결과를 CSV, Excel 등 다양한 포맷으로 저장
    - 센서 데이터, 시계열 데이터 등에 대한 특화된 처리
    """
    
    def __init__(self, connector: DatabaseConnector):
        """
        DatabaseExtractor 초기화
        
        Args:
            connector: 데이터베이스 연결 객체
        """
        self.connector = connector
    
    def query_to_dataframe(self, query: str, params: Optional[tuple] = None) -> pd.DataFrame:
        """
        SQL 쿼리 결과를 DataFrame으로 변환
        
        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 매개변수 (선택 사항)
            
        Returns:
            pd.DataFrame: 쿼리 결과 DataFrame
        """
        # 연결 상태 확인
        if not self.connector.check_connection():
            if not self.connector.connect():
                raise ConnectionError("데이터베이스에 연결할 수 없습니다.")
        
        try:
            # 쿼리 실행
            cursor = self.connector.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 결과 가져오기
            result = cursor.fetchall()
            
            # 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]
            
            # DataFrame 생성
            df = pd.DataFrame(result, columns=column_names)
            
            logger.info(f"쿼리 결과를 DataFrame으로 변환 완료: {len(df)} 행")
            return df
            
        except Exception as e:
            logger.error(f"DataFrame 변환 중 오류 발생: {str(e)}")
            raise
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
    
    def query_to_csv(self, query: str, output_path: Union[str, Path], 
                   params: Optional[tuple] = None,
                   chunksize: int = 10000,
                   encoding: str = 'utf-8',
                   include_header: bool = True,
                   delimiter: str = ',') -> str:
        """
        SQL 쿼리 결과를 CSV 파일로 저장
        
        Args:
            query: 실행할 SQL 쿼리
            output_path: 출력 파일 경로
            params: 쿼리 매개변수 (선택 사항)
            chunksize: 청크 크기 (대용량 데이터 처리용)
            encoding: 파일 인코딩
            include_header: 헤더 포함 여부
            delimiter: CSV 구분자
            
        Returns:
            str: 저장된 파일 경로
        """
        # 연결 상태 확인
        if not self.connector.check_connection():
            if not self.connector.connect():
                raise ConnectionError("데이터베이스에 연결할 수 없습니다.")
        
        try:
            # 출력 디렉토리 생성
            output_path = Path(output_path)
            os.makedirs(output_path.parent, exist_ok=True)
            
            # 쿼리 실행
            cursor = self.connector.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]
            
            # CSV 파일로 저장 (청크 단위)
            with open(output_path, 'w', newline='', encoding=encoding) as f:
                writer = csv.writer(f, delimiter=delimiter)
                
                # 헤더 쓰기
                if include_header:
                    writer.writerow(column_names)
                
                # 데이터 쓰기 (청크 단위)
                total_rows = 0
                while True:
                    rows = cursor.fetchmany(chunksize)
                    if not rows:
                        break
                    
                    writer.writerows(rows)
                    total_rows += len(rows)
                    logger.info(f"CSV 내보내기 진행 중: {total_rows:,}개 행 처리됨")
            
            logger.info(f"쿼리 결과를 CSV 파일 '{output_path}'에 저장 완료: {total_rows:,}개 행")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"CSV 내보내기 중 오류 발생: {str(e)}")
            raise
        finally:
            if 'cursor' in locals() and cursor:
                cursor.close()
    
    def extract_table(self, table_name: str, columns: Optional[List[str]] = None,
                    condition: Optional[str] = None, 
                    order_by: Optional[str] = None,
                    limit: Optional[int] = None) -> pd.DataFrame:
        """
        테이블 데이터 추출
        
        Args:
            table_name: 테이블 이름
            columns: 추출할 컬럼 목록 (None이면 모든 컬럼)
            condition: WHERE 조건절 (선택 사항)
            order_by: ORDER BY 절 (선택 사항)
            limit: 최대 행 수 (선택 사항)
            
        Returns:
            pd.DataFrame: 추출된 데이터
        """
        # SQL 쿼리 구성
        column_str = "*" if not columns else ", ".join(columns)
        query = f"SELECT {column_str} FROM {table_name}"
        
        if condition:
            query += f" WHERE {condition}"
            
        if order_by:
            query += f" ORDER BY {order_by}"
            
        if limit:
            # 데이터베이스 유형별 LIMIT 문법
            if self.connector.db_type in ['mysql', 'postgresql', 'sqlite']:
                query += f" LIMIT {limit}"
            elif self.connector.db_type == 'sqlserver':
                # SQL Server 2012 이상에서는 TOP 사용
                query = query.replace(f"SELECT {column_str}", f"SELECT TOP {limit} {column_str}")
            elif self.connector.db_type == 'oracle':
                # Oracle 12c 이상에서는 FETCH FIRST 사용
                query += f" FETCH FIRST {limit} ROWS ONLY"
        
        # 쿼리 실행 및 DataFrame 반환
        return self.query_to_dataframe(query)
    
    def extract_to_csv(self, table_name: str, output_path: Union[str, Path],
                     columns: Optional[List[str]] = None,
                     condition: Optional[str] = None,
                     order_by: Optional[str] = None,
                     chunksize: int = 10000,
                     delimiter: str = ',') -> str:
        """
        테이블 데이터를 CSV 파일로 추출
        
        Args:
            table_name: 테이블 이름
            output_path: 출력 파일 경로
            columns: 추출할 컬럼 목록 (None이면 모든 컬럼)
            condition: WHERE 조건절 (선택 사항)
            order_by: ORDER BY 절 (선택 사항)
            chunksize: 청크 크기 (대용량 데이터 처리용)
            delimiter: CSV 구분자
            
        Returns:
            str: 저장된 파일 경로
        """
        # SQL 쿼리 구성
        column_str = "*" if not columns else ", ".join(columns)
        query = f"SELECT {column_str} FROM {table_name}"
        
        if condition:
            query += f" WHERE {condition}"
            
        if order_by:
            query += f" ORDER BY {order_by}"
        
        # 쿼리 실행 및 CSV 저장
        return self.query_to_csv(
            query=query,
            output_path=output_path,
            chunksize=chunksize,
            delimiter=delimiter
        )
    
    def extract_sensor_data(self, table_name: str, 
                         time_column: str,
                         sensor_columns: List[str],
                         start_time: Optional[Union[str, datetime]] = None,
                         end_time: Optional[Union[str, datetime]] = None,
                         status_column: Optional[str] = None,
                         equipment_id_column: Optional[str] = None,
                         equipment_id: Optional[str] = None) -> pd.DataFrame:
        """
        센서 데이터 특화 추출
        
        Args:
            table_name: 센서 데이터 테이블 이름
            time_column: 시간 컬럼명
            sensor_columns: 센서 값 컬럼 목록
            start_time: 시작 시간 (선택 사항)
            end_time: 종료 시간 (선택 사항)
            status_column: 상태 컬럼명 (선택 사항)
            equipment_id_column: 장비 ID 컬럼명 (선택 사항)
            equipment_id: 추출할 장비 ID (선택 사항)
            
        Returns:
            pd.DataFrame: 추출된 센서 데이터
        """
        # 추출할 컬럼 목록 준비
        columns = [time_column] + sensor_columns
        if status_column:
            columns.append(status_column)
        if equipment_id_column:
            columns.append(equipment_id_column)
            
        # 조건절 구성
        conditions = []
        
        if start_time:
            # 시작 시간이 문자열인 경우 데이터베이스 유형에 맞게 변환
            if isinstance(start_time, str):
                if self.connector.db_type in ['mysql', 'postgresql']:
                    conditions.append(f"{time_column} >= '{start_time}'")
                elif self.connector.db_type == 'sqlite':
                    conditions.append(f"{time_column} >= datetime('{start_time}')")
                elif self.connector.db_type == 'sqlserver':
                    conditions.append(f"{time_column} >= CONVERT(DATETIME, '{start_time}')")
                elif self.connector.db_type == 'oracle':
                    conditions.append(f"{time_column} >= TO_DATE('{start_time}', 'YYYY-MM-DD HH24:MI:SS')")
            else:
                # datetime 객체인 경우 문자열로 변환
                start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
                if self.connector.db_type in ['mysql', 'postgresql']:
                    conditions.append(f"{time_column} >= '{start_time_str}'")
                elif self.connector.db_type == 'sqlite':
                    conditions.append(f"{time_column} >= datetime('{start_time_str}')")
                elif self.connector.db_type == 'sqlserver':
                    conditions.append(f"{time_column} >= CONVERT(DATETIME, '{start_time_str}')")
                elif self.connector.db_type == 'oracle':
                    conditions.append(f"{time_column} >= TO_DATE('{start_time_str}', 'YYYY-MM-DD HH24:MI:SS')")
        
        if end_time:
            # 종료 시간 조건 추가
            if isinstance(end_time, str):
                if self.connector.db_type in ['mysql', 'postgresql']:
                    conditions.append(f"{time_column} <= '{end_time}'")
                elif self.connector.db_type == 'sqlite':
                    conditions.append(f"{time_column} <= datetime('{end_time}')")
                elif self.connector.db_type == 'sqlserver':
                    conditions.append(f"{time_column} <= CONVERT(DATETIME, '{end_time}')")
                elif self.connector.db_type == 'oracle':
                    conditions.append(f"{time_column} <= TO_DATE('{end_time}', 'YYYY-MM-DD HH24:MI:SS')")
            else:
                end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
                if self.connector.db_type in ['mysql', 'postgresql']:
                    conditions.append(f"{time_column} <= '{end_time_str}'")
                elif self.connector.db_type == 'sqlite':
                    conditions.append(f"{time_column} <= datetime('{end_time_str}')")
                elif self.connector.db_type == 'sqlserver':
                    conditions.append(f"{time_column} <= CONVERT(DATETIME, '{end_time_str}')")
                elif self.connector.db_type == 'oracle':
                    conditions.append(f"{time_column} <= TO_DATE('{end_time_str}', 'YYYY-MM-DD HH24:MI:SS')")
        
        # 장비 ID 조건 추가
        if equipment_id and equipment_id_column:
            conditions.append(f"{equipment_id_column} = '{equipment_id}'")
        
        # 조건절 조합
        condition = " AND ".join(conditions) if conditions else None
        
        # 시간 기준 정렬
        order_by = f"{time_column} ASC"
        
        # 데이터 추출
        return self.extract_table(
            table_name=table_name,
            columns=columns,
            condition=condition,
            order_by=order_by
        )
    
    def extract_time_series(self, query: str, time_column: str,
                          resample_freq: Optional[str] = None,
                          aggregation: Optional[Dict[str, str]] = None,
                          fill_method: Optional[str] = None) -> pd.DataFrame:
        """
        시계열 데이터 추출 및 가공
        
        Args:
            query: 실행할 SQL 쿼리
            time_column: 시간 컬럼명
            resample_freq: 리샘플링 주기 (예: '1H', '5T', '1D')
            aggregation: 컬럼별 집계 방법 사전 (예: {'temp': 'mean', 'humidity': 'max'})
            fill_method: 결측치 처리 방법 ('ffill', 'bfill', 'interpolate')
            
        Returns:
            pd.DataFrame: 가공된 시계열 데이터
        """
        # 데이터 추출
        df = self.query_to_dataframe(query)
        
        # 시간 컬럼이 없으면 오류
        if time_column not in df.columns:
            raise ValueError(f"시간 컬럼 '{time_column}'이 결과에 없습니다.")
        
        # 시간 컬럼 변환 (문자열 -> datetime)
        if not pd.api.types.is_datetime64_any_dtype(df[time_column]):
            df[time_column] = pd.to_datetime(df[time_column])
        
        # 시간 컬럼을 인덱스로 설정
        df.set_index(time_column, inplace=True)
        
        # 시계열 데이터 리샘플링 (지정된 경우)
        if resample_freq:
            if aggregation:
                # 컬럼별 다른 집계 방법 적용
                resampled = df.resample(resample_freq)
                df_resampled = pd.DataFrame()
                
                for col, agg_method in aggregation.items():
                    if col in df.columns:
                        df_resampled[col] = getattr(resampled[col], agg_method)()
                
                df = df_resampled
            else:
                # 기본 집계 방법 (평균) 적용
                df = df.resample(resample_freq).mean()
        
        # 결측치 처리 (지정된 경우)
        if fill_method:
            if fill_method == 'ffill':
                df = df.fillna(method='ffill')
            elif fill_method == 'bfill':
                df = df.fillna(method='bfill')
            elif fill_method == 'interpolate':
                df = df.interpolate(method='time')
        
        # 인덱스 재설정 (시간 컬럼 복원)
        df.reset_index(inplace=True)
        
        return df
    
    def extract_join_tables(self, tables: List[str], join_conditions: List[str],
                          columns: Optional[List[str]] = None,
                          condition: Optional[str] = None) -> pd.DataFrame:
        """
        여러 테이블을 조인하여 데이터 추출
        
        Args:
            tables: 테이블 이름 목록
            join_conditions: 조인 조건 목록 (예: "table1.id = table2.table1_id")
            columns: 추출할 컬럼 목록 (None이면 모든 컬럼)
            condition: WHERE 조건절 (선택 사항)
            
        Returns:
            pd.DataFrame: 조인된 데이터
        """
        # SQL 쿼리 구성
        from_clause = tables[0]
        for i, join_condition in enumerate(join_conditions):
            if i + 1 < len(tables):
                from_clause += f" JOIN {tables[i+1]} ON {join_condition}"
        
        # 컬럼 목록 준비
        column_str = "*" if not columns else ", ".join(columns)
        
        # SQL 쿼리 생성
        query = f"SELECT {column_str} FROM {from_clause}"
        
        if condition:
            query += f" WHERE {condition}"
        
        # 쿼리 실행
        return self.query_to_dataframe(query)
    
    def extract_timeseries_to_csv_chunked(self, table_name: str, time_column: str,
                                       output_path: Union[str, Path],
                                       start_time: Optional[Union[str, datetime]] = None,
                                       end_time: Optional[Union[str, datetime]] = None,
                                       columns: Optional[List[str]] = None,
                                       condition: Optional[str] = None,
                                       chunksize: int = 10000,
                                       chunk_time_interval: str = '1D') -> str:
        """
        대용량 시계열 데이터를 시간 청크 단위로 CSV 파일로 추출
        
        Args:
            table_name: 테이블 이름
            time_column: 시간 컬럼명
            output_path: 출력 파일 경로
            start_time: 시작 시간 (선택 사항)
            end_time: 종료 시간 (선택 사항)
            columns: 추출할 컬럼 목록 (None이면 모든 컬럼)
            condition: 추가 WHERE 조건절 (선택 사항)
            chunksize: 한 번에 처리할 행 수
            chunk_time_interval: 시간 청크 간격 (예: '1D', '4H', '60T')
            
        Returns:
            str: 저장된 파일 경로
        """
        # 출력 디렉토리 생성
        output_path = Path(output_path)
        os.makedirs(output_path.parent, exist_ok=True)
        
        # 시작/종료 시간 변환
        if start_time is None:
            # 최소 시간 조회
            min_time_query = f"SELECT MIN({time_column}) FROM {table_name}"
            min_time_result = self.connector.execute_query(min_time_query)
            start_time = min_time_result[0][0]
        
        if end_time is None:
            # 최대 시간 조회
            max_time_query = f"SELECT MAX({time_column}) FROM {table_name}"
            max_time_result = self.connector.execute_query(max_time_query)
            end_time = max_time_result[0][0]
        
        # datetime 객체로 변환
        if isinstance(start_time, str):
            start_time = pd.to_datetime(start_time)
        if isinstance(end_time, str):
            end_time = pd.to_datetime(end_time)
        
        # 시간 범위 생성
        time_ranges = pd.date_range(start=start_time, end=end_time, freq=chunk_time_interval)
        
        # 파일 열기
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = None  # CSV 작성자 (첫 번째 청크에서 초기화)
            total_rows = 0
            
            # 첫 번째 청크로 헤더 설정 여부 플래그
            header_written = False
            
            # 각 시간 청크에 대해 데이터 추출
            for i in range(len(time_ranges) - 1):
                chunk_start = time_ranges[i]
                chunk_end = time_ranges[i + 1]
                
                # 컬럼 목록 준비
                column_str = "*" if not columns else ", ".join(columns)
                
                # 시간 조건
                time_condition = f"{time_column} >= '{chunk_start}' AND {time_column} < '{chunk_end}'"
                
                # 조건 결합
                if condition:
                    combined_condition = f"({time_condition}) AND ({condition})"
                else:
                    combined_condition = time_condition
                
                # SQL 쿼리 구성
                query = f"SELECT {column_str} FROM {table_name} WHERE {combined_condition}"
                
                # 데이터 추출
                cursor = self.connector.connection.cursor()
                cursor.execute(query)
                
                # 첫 번째 청크일 경우 컬럼 이름 가져와서 헤더 설정
                if not header_written:
                    column_names = [desc[0] for desc in cursor.description]
                    writer = csv.writer(f)
                    writer.writerow(column_names)
                    header_written = True
                
                # 행 단위로 처리 (메모리 효율적)
                chunk_rows = 0
                while True:
                    rows = cursor.fetchmany(chunksize)
                    if not rows:
                        break
                    
                    writer.writerows(rows)
                    chunk_rows += len(rows)
                
                # 청크 완료 로그
                total_rows += chunk_rows
                logger.info(f"시간 청크 {chunk_start} ~ {chunk_end} 처리 완료: {chunk_rows:,}개 행 (누적: {total_rows:,}개)")
                
                # 커서 닫기
                cursor.close()
        
        logger.info(f"대용량 시계열 데이터 추출 완료: {total_rows:,}개 행이 '{output_path}'에 저장됨")
        return str(output_path)
    
    def execute_analysis_query(self, query_template: str, params: Dict[str, Any]) -> pd.DataFrame:
        """
        템플릿 기반 분석 쿼리 실행
        
        Args:
            query_template: SQL 쿼리 템플릿 (파라미터는 {param_name} 형식)
            params: 템플릿 파라미터
            
        Returns:
            pd.DataFrame: 분석 결과
        """
        # 템플릿에 파라미터 적용
        try:
            query = query_template.format(**params)
        except KeyError as e:
            raise ValueError(f"쿼리 템플릿에 필요한 파라미터가 없습니다: {e}")
        
        # 쿼리 실행
        return self.query_to_dataframe(query)

# 예제 사용법
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # SQLite 메모리 DB 연결 (테스트용)
    connector = DatabaseConnector('sqlite', database=':memory:')
    connector.connect()
    
    # 테스트 테이블 생성
    connector.execute_query('''
        CREATE TABLE test_sensors (
            id INTEGER PRIMARY KEY,
            timestamp TEXT,
            sensor_id TEXT,
            temperature REAL,
            humidity REAL
        )
    ''')
    
    # 테스트 데이터 삽입
    for i in range(100):
        timestamp = (datetime.now() - pd.Timedelta(hours=i)).strftime('%Y-%m-%d %H:%M:%S')
        connector.execute_query(
            "INSERT INTO test_sensors (timestamp, sensor_id, temperature, humidity) VALUES (?, ?, ?, ?)",
            (timestamp, f"sensor_{i % 3 + 1}", 20 + i % 10, 50 + i % 20)
        )
    
    # 추출기 생성
    extractor = DatabaseExtractor(connector)
    
    # 테이블 데이터 추출
    df = extractor.extract_table("test_sensors", limit=10)
    print(f"추출된 데이터:\n{df}")
    
    # 센서 데이터 추출
    sensor_df = extractor.extract_sensor_data(
        table_name="test_sensors",
        time_column="timestamp",
        sensor_columns=["temperature", "humidity"],
        equipment_id_column="sensor_id",
        equipment_id="sensor_1"
    )
    print(f"센서 데이터:\n{sensor_df.head()}")
    
    # 시계열 데이터 변환
    ts_df = extractor.extract_time_series(
        query="SELECT * FROM test_sensors",
        time_column="timestamp",
        resample_freq="4H",
        aggregation={"temperature": "mean", "humidity": "max"}
    )
    print(f"시계열 데이터:\n{ts_df.head()}")
    
    # 연결 종료
    connector.disconnect()