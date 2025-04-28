"""
데이터베이스 데이터 내보내기 모듈

이 모듈은 데이터베이스에서 데이터를 CSV 파일로 내보내는 기능을 제공합니다.
이는 db_export_cli.py의 핵심 기능을 재사용 가능한 클래스로 제공합니다.
"""

import os
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# DB 커넥터 임포트
from src.utils.db.connector import DBConnector

# 로깅 설정
logger = logging.getLogger(__name__)

class DBExporter:
    """
    데이터베이스 데이터 내보내기 클래스
    
    데이터베이스에서 데이터를 CSV 파일로 내보내는 기능을 제공합니다.
    """
    
    def __init__(self, connector: DBConnector):
        """
        DBExporter 초기화
        
        Args:
            connector: 데이터베이스 연결 객체
        """
        self.connector = connector
        
        # 내보내기 설정
        self.export_params = {
            'csv_separator': ',',
            'include_header': True,
            'chunk_size': 10000,
            'output_path': Path('data/raw/extracted')
        }
        
        logger.info("DBExporter 초기화 완료")
    
    def set_export_params(self, params: Dict[str, Any]) -> None:
        """
        내보내기 설정 업데이트
        
        Args:
            params: 내보내기 설정 {'csv_separator': ',', 'include_header': True, ...}
        """
        for key, value in params.items():
            if key in self.export_params:
                # 출력 경로는 Path 객체로 변환
                if key == 'output_path' and not isinstance(value, Path):
                    value = Path(value)
                
                self.export_params[key] = value
    
    def export_query_to_csv(self, 
                           query: str, 
                           output_file: Union[str, Path],
                           params: Optional[tuple] = None) -> Optional[Path]:
        """
        SQL 쿼리 결과를 CSV 파일로 내보내기
        
        Args:
            query: 실행할 SQL 쿼리
            output_file: 출력 파일 경로
            params: 쿼리 매개변수 (선택 사항)
            
        Returns:
            Optional[Path]: 내보내기 성공 시 파일 경로, 실패 시 None
        """
        if not self.connector.is_connected():
            logger.error("데이터베이스에 연결되어 있지 않습니다.")
            return None
        
        # 출력 파일 경로 처리
        output_file = Path(output_file)
        os.makedirs(output_file.parent, exist_ok=True)
        
        try:
            # 쿼리 실행
            cursor = self.connector.connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]
            
            # CSV 파일로 저장
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=self.export_params['csv_separator'])
                
                # 헤더 쓰기
                if self.export_params['include_header']:
                    writer.writerow(column_names)
                
                # 청크 단위로 데이터 쓰기
                total_rows = 0
                chunk_size = self.export_params['chunk_size']
                
                while True:
                    rows = cursor.fetchmany(chunk_size)
                    if not rows:
                        break
                    
                    writer.writerows(rows)
                    total_rows += len(rows)
                    logger.debug(f"{total_rows:,}개 행 내보내기 완료...")
            
            cursor.close()
            logger.info(f"쿼리 결과 {total_rows:,}개 행이 '{output_file}'에 저장되었습니다.")
            return output_file
            
        except Exception as e:
            logger.error(f"쿼리 결과 내보내기 중 오류 발생: {str(e)}")
            return None
    
    def export_table_to_csv(self, 
                           table_name: str, 
                           output_file: Union[str, Path],
                           columns: Optional[List[str]] = None,
                           condition: Optional[str] = None,
                           order_by: Optional[str] = None,
                           limit: Optional[int] = None) -> Optional[Path]:
        """
        테이블을 CSV 파일로 내보내기
        
        Args:
            table_name: 테이블 이름
            output_file: 출력 파일 경로
            columns: 내보낼 컬럼 목록 (None이면 모든 컬럼)
            condition: WHERE 조건절 (선택 사항)
            order_by: ORDER BY 절 (선택 사항)
            limit: 최대 행 수 (선택 사항)
            
        Returns:
            Optional[Path]: 내보내기 성공 시 파일 경로, 실패 시 None
        """
        # SQL 쿼리 구성
        column_str = "*" if not columns else ", ".join(columns)
        query = f"SELECT {column_str} FROM {table_name}"
        
        if condition:
            query += f" WHERE {condition}"
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            # DB 유형에 따라 적절한 LIMIT 구문 추가
            if self.connector.db_type in ['mysql', 'postgresql', 'sqlite']:
                query += f" LIMIT {limit}"
            elif self.connector.db_type == 'sqlserver':
                # SQL Server 2012 이상에서는 TOP 사용
                query = query.replace(f"SELECT {column_str}", f"SELECT TOP {limit} {column_str}")
            elif self.connector.db_type == 'oracle':
                # Oracle 12c 이상에서는 FETCH FIRST 사용
                query += f" FETCH FIRST {limit} ROWS ONLY"
                
        # 쿼리 실행 및 CSV 내보내기
        return self.export_query_to_csv(query, output_file)
    
    def export_join_to_csv(self, 
                          tables: List[str], 
                          join_conditions: List[str],
                          output_file: Union[str, Path],
                          columns: Optional[List[str]] = None,
                          condition: Optional[str] = None,
                          order_by: Optional[str] = None,
                          limit: Optional[int] = None) -> Optional[Path]:
        """
        여러 테이블 조인 결과를 CSV 파일로 내보내기
        
        Args:
            tables: 테이블 이름 목록
            join_conditions: 조인 조건 목록 (예: "t1.id = t2.t1_id")
            output_file: 출력 파일 경로
            columns: 내보낼 컬럼 목록 (None이면 모든 컬럼)
            condition: WHERE 조건절 (선택 사항)
            order_by: ORDER BY 절 (선택 사항)
            limit: 최대 행 수 (선택 사항)
            
        Returns:
            Optional[Path]: 내보내기 성공 시 파일 경로, 실패 시 None
        """
        # FROM 절 구성
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
        
        if order_by:
            query += f" ORDER BY {order_by}"
        
        if limit:
            # DB 유형에 따라 적절한 LIMIT 구문 추가
            if self.connector.db_type in ['mysql', 'postgresql', 'sqlite']:
                query += f" LIMIT {limit}"
            elif self.connector.db_type == 'sqlserver':
                # SQL Server 2012 이상에서는 TOP 사용
                query = query.replace(f"SELECT {column_str}", f"SELECT TOP {limit} {column_str}")
            elif self.connector.db_type == 'oracle':
                # Oracle 12c 이상에서는 FETCH FIRST 사용
                query += f" FETCH FIRST {limit} ROWS ONLY"
                
        # 쿼리 실행 및 CSV 내보내기
        return self.export_query_to_csv(query, output_file)
    
    def export_model_results_to_csv(self, 
                                   output_file: Union[str, Path],
                                   model_name: Optional[str] = None,
                                   limit: Optional[int] = None,
                                   table_name: str = "model_results") -> Optional[Path]:
        """
        모델 결과를 CSV 파일로 내보내기
        
        Args:
            output_file: 출력 파일 경로
            model_name: 특정 모델 이름으로 필터링 (선택 사항)
            limit: 최대 행 수 (선택 사항)
            table_name: 모델 결과 테이블 이름
            
        Returns:
            Optional[Path]: 내보내기 성공 시 파일 경로, 실패 시 None
        """
        # 테이블 존재 확인
        if not self.connector.table_exists(table_name):
            logger.error(f"테이블 '{table_name}'이(가) 존재하지 않습니다.")
            return None
        
        # 조건 구성
        condition = None
        if model_name:
            condition = f"model_name = '{model_name}'"
        
        # 데이터 추출
        return self.export_table_to_csv(
            table_name=table_name,
            output_file=output_file,
            condition=condition,
            order_by="created_at DESC",
            limit=limit
        )
    
    def export_prediction_history_to_csv(self, 
                                        output_file: Union[str, Path],
                                        model_name: Optional[str] = None,
                                        is_correct: Optional[bool] = None,
                                        limit: Optional[int] = None,
                                        table_name: str = "prediction_history") -> Optional[Path]:
        """
        예측 이력을 CSV 파일로 내보내기
        
        Args:
            output_file: 출력 파일 경로
            model_name: 특정 모델 이름으로 필터링 (선택 사항)
            is_correct: 예측 정확성 여부로 필터링 (선택 사항)
            limit: 최대 행 수 (선택 사항)
            table_name: 예측 이력 테이블 이름
            
        Returns:
            Optional[Path]: 내보내기 성공 시 파일 경로, 실패 시 None
        """
        # 테이블 존재 확인
        if not self.connector.table_exists(table_name):
            logger.error(f"테이블 '{table_name}'이(가) 존재하지 않습니다.")
            return None
        
        # 조건 구성
        conditions = []
        if model_name:
            conditions.append(f"model_name = '{model_name}'")
        if is_correct is not None:
            conditions.append(f"is_correct = {1 if is_correct else 0}")
        
        condition = None
        if conditions:
            condition = " AND ".join(conditions)
        
        # 데이터 추출
        return self.export_table_to_csv(
            table_name=table_name,
            output_file=output_file,
            condition=condition,
            order_by="timestamp DESC",
            limit=limit
        )

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # DB 연결
    from src.utils.db.connector import DBConnector
    
    connector = DBConnector()