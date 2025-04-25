#!/usr/bin/env python
"""
데이터베이스 추출 CLI 모듈

이 모듈은 SQL 데이터베이스에서 데이터를 CSV로 추출하기 위한 대화형 인터페이스를 제공합니다.
다양한 데이터베이스 유형(MySQL, PostgreSQL, SQLite 등)을 지원합니다.
"""

import os
import sys
import logging
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기본 CLI 클래스 임포트
from src.cli.base_cli import BaseCLI

# 데이터베이스 유틸리티 임포트
from src.utils.db_utils import (
    get_db_connection, execute_query, get_available_tables, 
    get_table_schema, get_row_count
)

# 로깅 설정
logger = logging.getLogger(__name__)

class DBExportCLI(BaseCLI):
    """
    데이터베이스 추출 CLI 클래스
    
    SQL 데이터베이스에서 CSV로 데이터를 추출하는 기능을 제공합니다.
    """
    
    def __init__(self):
        """데이터베이스 추출 CLI 초기화"""
        super().__init__(title="SQL 데이터베이스 CSV 추출 도구")
        
        # 상태 초기화
        self.state = {
            'DB 연결': None,
            'DB 유형': None,
            '선택된 테이블': None,
            '마지막 추출 파일': None
        }
        
        # 연결 정보 초기화
        self.connection_params = {
            'host': 'localhost',
            'port': {
                'mysql': 3306,
                'postgresql': 5432,
                'sqlserver': 1433
            },
            'database': '',
            'username': '',
            'password': ''
        }
        
        # 추출 설정 초기화
        self.export_params = {
            'csv_separator': ',',
            'include_header': True,
            'chunk_size': 10000,
            'output_path': self.paths["data"] / "raw" / "extracted"
        }
        
        logger.info("데이터베이스 추출 CLI 초기화 완료")
    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header()
            print("SQL 데이터베이스에서 CSV로 데이터를 추출하는 도구입니다.")
            print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
            
            menu_options = [
                "데이터베이스 연결 설정",
                "테이블 목록 보기",
                "테이블 스키마 보기",
                "SQL 쿼리 실행 및 CSV 추출",
                "테이블 직접 CSV 추출",
                "추출 설정",
                "종료"
            ]
            
            self.print_status()
            choice = self.show_menu(menu_options, "메인 메뉴")
            
            if choice == 0:
                self.db_connection_menu()
            elif choice == 1:
                self.list_tables_menu()
            elif choice == 2:
                self.view_schema_menu()
            elif choice == 3:
                self.sql_query_menu()
            elif choice == 4:
                self.table_export_menu()
            elif choice == 5:
                self.export_settings_menu()
            elif choice == 6:
                print("\n프로그램을 종료합니다. 감사합니다!")
                # 연결 닫기
                if self.state['DB 연결'] is not None:
                    self.state['DB 연결'].close()
                    print("데이터베이스 연결을 닫았습니다.")
                break
    
    def db_connection_menu(self) -> None:
        """데이터베이스 연결 설정 메뉴"""
        self.print_header("데이터베이스 연결 설정")
        
        print("데이터베이스 연결 정보를 설정합니다.\n")
        
        # 기존 연결 닫기
        if self.state['DB 연결'] is not None:
            self.state['DB 연결'].close()
            self.update_state('DB 연결', None)
            print("기존 데이터베이스 연결을 닫았습니다.")
        
        # 데이터베이스 유형 선택
        print("\n데이터베이스 유형 선택:")
        db_types = ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Oracle"]
        db_type_idx = self.show_menu(db_types, "데이터베이스 유형")
        db_type_lower = db_types[db_type_idx].lower()
        
        # SQLite는 파일 경로만 필요
        if db_type_lower == 'sqlite':
            db_file = self.get_input("SQLite 데이터베이스 파일 경로", "database.db")
            
            try:
                import sqlite3
                connection = sqlite3.connect(db_file)
                self.update_state('DB 연결', connection)
                self.update_state('DB 유형', 'sqlite')
                self.show_success(f"SQLite 데이터베이스 '{db_file}'에 연결되었습니다.")
            except Exception as e:
                self.show_error(f"SQLite 데이터베이스 연결 실패: {str(e)}")
                logger.exception("SQLite 연결 실패")
            
            self.wait_for_user()
            return
        
        # 다른 데이터베이스는 연결 정보 필요
        print(f"\n{db_types[db_type_idx]} 연결 정보 입력:")
        
        host = self.get_input("호스트", self.connection_params['host'])
        port = self.get_numeric_input(
            "포트", 
            self.connection_params['port'].get(db_type_lower, 3306), 
            min_val=1, 
            max_val=65535
        )
        database = self.get_input("데이터베이스 이름", self.connection_params['database'])
        username = self.get_input("사용자 이름", self.connection_params['username'])
        password = self.get_input("비밀번호")
        
        # 연결 정보 저장
        self.connection_params['host'] = host
        self.connection_params['port'][db_type_lower] = port
        self.connection_params['database'] = database
        self.connection_params['username'] = username
        self.connection_params['password'] = password
        
        # 데이터베이스 연결 시도
        try:
            connection = get_db_connection(
                db_type=db_type_lower,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password
            )
            
            self.update_state('DB 연결', connection)
            self.update_state('DB 유형', db_type_lower)
            self.show_success(f"{db_types[db_type_idx]} 데이터베이스에 연결되었습니다.")
            
        except Exception as e:
            self.show_error(f"데이터베이스 연결 실패: {str(e)}")
            logger.exception("데이터베이스 연결 실패")
        
        self.wait_for_user()
    
    def list_tables_menu(self) -> None:
        """테이블 목록 보기 메뉴"""
        self.print_header("테이블 목록")
        
        # 연결 확인
        if self.state['DB 연결'] is None:
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        try:
            tables = get_available_tables(self.state['DB 연결'], self.state['DB 유형'])
            
            if not tables:
                print("데이터베이스에 테이블이 없습니다.")
            else:
                print(f"데이터베이스에 {len(tables)}개의 테이블이 있습니다:\n")
                for i, table in enumerate(tables, 1):
                    print(f"{i}. {table}")
                
                # 테이블 선택 옵션
                select_table = self.get_yes_no_input("\n테이블을 선택하시겠습니까?")
                if select_table:
                    choice = self.get_input("선택할 테이블 번호", "1")
                    try:
                        choice_idx = int(choice) - 1
                        if 0 <= choice_idx < len(tables):
                            selected_table = tables[choice_idx]
                            self.update_state('선택된 테이블', selected_table)
                            self.show_success(f"테이블 '{selected_table}'을(를) 선택했습니다.")
                            
                            # 테이블 행 수 표시
                            count = get_row_count(self.state['DB 연결'], selected_table)
                            print(f"테이블 '{selected_table}'에는 약 {count:,}개의 행이 있습니다.")
                        else:
                            self.show_error(f"유효한 번호를 입력하세요 (1-{len(tables)})")
                    except ValueError:
                        self.show_error("숫자를 입력하세요")
        
        except Exception as e:
            self.show_error(f"테이블 목록 조회 실패: {str(e)}")
            logger.exception("테이블 목록 조회 실패")
        
        self.wait_for_user()
    
    def view_schema_menu(self) -> None:
        """테이블 스키마 보기 메뉴"""
        self.print_header("테이블 스키마 보기")
        
        # 연결 확인
        if self.state['DB 연결'] is None:
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        # 테이블 선택
        table_name = None
        if self.state['선택된 테이블'] is not None:
            use_selected = self.get_yes_no_input(
                f"현재 선택된 테이블 '{self.state['선택된 테이블']}'의 스키마를 보시겠습니까?", default=True
            )
            if use_selected:
                table_name = self.state['선택된 테이블']
        
        if table_name is None:
            try:
                tables = get_available_tables(self.state['DB 연결'], self.state['DB 유형'])
                
                if not tables:
                    self.show_error("데이터베이스에 테이블이 없습니다.")
                    self.wait_for_user()
                    return
                
                print("사용 가능한 테이블:")
                for i, table in enumerate(tables, 1):
                    print(f"{i}. {table}")
                
                choice = self.get_input("\n스키마를 볼 테이블 번호를 입력하세요", "1")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(tables):
                        table_name = tables[choice_idx]
                    else:
                        self.show_error(f"유효한 번호를 입력하세요 (1-{len(tables)})")
                        self.wait_for_user()
                        return
                except ValueError:
                    self.show_error("숫자를 입력하세요")
                    self.wait_for_user()
                    return
            
            except Exception as e:
                self.show_error(f"테이블 목록 조회 실패: {str(e)}")
                logger.exception("테이블 목록 조회 실패")
                self.wait_for_user()
                return
        
        # 선택한 테이블의 스키마 보기
        try:
            schema = get_table_schema(self.state['DB 연결'], table_name, self.state['DB 유형'])
            
            print(f"\n테이블 '{table_name}'의 스키마:\n")
            print("-" * 50)
            print(f"{'컬럼명':<20} {'데이터 유형':<15} {'기타 속성':<15}")
            print("-" * 50)
            
            for column in schema:
                print(f"{column['name']:<20} {column['type']:<15} {column.get('extra', ''):<15}")
            
            print("-" * 50)
            
            # 행 수 표시
            count = get_row_count(self.state['DB 연결'], table_name)
            print(f"\n테이블 '{table_name}'에는 약 {count:,}개의 행이 있습니다.")
            
            # 현재 테이블로 설정
            self.update_state('선택된 테이블', table_name)
            
        except Exception as e:
            self.show_error(f"테이블 스키마 조회 실패: {str(e)}")
            logger.exception("테이블 스키마 조회 실패")
        
        self.wait_for_user()
    
    def sql_query_menu(self) -> None:
        """SQL 쿼리 실행 및 CSV 추출 메뉴"""
        self.print_header("SQL 쿼리 실행 및 CSV 추출")
        
        # 연결 확인
        if self.state['DB 연결'] is None:
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        # SQL 쿼리 입력 방법 선택
        print("SQL 쿼리 입력 방법을 선택하세요:\n")
        input_methods = ["쿼리 직접 입력", "파일에서 쿼리 로드"]
        input_method_idx = self.show_menu(input_methods, "입력 방법")
        
        query = ""
        
        if input_method_idx == 0:  # 직접 입력
            print("\nSQL 쿼리를 입력하세요 (입력 완료 후 빈 줄에서 Enter):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            query = "\n".join(lines)
        
        else:  # 파일에서 로드
            query_file = self.get_input("SQL 쿼리 파일 경로", "query.sql")
            try:
                with open(query_file, 'r') as f:
                    query = f.read()
                print(f"\n파일 '{query_file}'에서 쿼리를 로드했습니다.")
            except Exception as e:
                self.show_error(f"쿼리 파일 로드 실패: {str(e)}")
                self.wait_for_user()
                return
        
        # 쿼리가 비어있는지 확인
        if not query.strip():
            self.show_error("빈 쿼리는 실행할 수 없습니다.")
            self.wait_for_user()
            return
        
        print(f"\n실행할 쿼리:")
        print("-" * 50)
        print(query)
        print("-" * 50)
        
        # 출력 경로 설정
        output_dir = self.export_params['output_path']
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = self.get_input(
            "출력 CSV 파일 경로", 
            output_dir / "query_result.csv"
        )
        
        # 쿼리 실행 및 CSV 출력 확인
        execute = self.get_yes_no_input("\n쿼리를 실행하고 결과를 CSV로 저장하시겠습니까?")
        if not execute:
            print("쿼리 실행을 취소합니다.")
            self.wait_for_user()
            return
        
        try:
            # 쿼리 실행 및 CSV 저장
            cursor = self.state['DB 연결'].cursor()
            cursor.execute(query)
            
            # 결과 가져오기
            result = cursor.fetchall()
            
            # 결과가 비어있는지 확인
            if not result:
                self.show_warning("쿼리 결과가 비어있습니다.")
                self.wait_for_user()
                return
            
            # 컬럼 이름 가져오기
            column_names = [desc[0] for desc in cursor.description]
            
            # CSV 파일로 저장
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter=self.export_params['csv_separator'])
                
                # 헤더 쓰기
                if self.export_params['include_header']:
                    writer.writerow(column_names)
                
                # 데이터 쓰기
                writer.writerows(result)
            
            self.update_state('마지막 추출 파일', output_file)
            self.show_success(f"쿼리 결과가 '{output_file}'에 저장되었습니다. ({len(result):,}개 행)")
            
        except Exception as e:
            self.show_error(f"쿼리 실행 실패: {str(e)}")
            logger.exception("쿼리 실행 실패")
        
        self.wait_for_user()
    
    def table_export_menu(self) -> None:
        """테이블 직접 CSV 추출 메뉴"""
        self.print_header("테이블 직접 CSV 추출")
        
        # 연결 확인
        if self.state['DB 연결'] is None:
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        # 테이블 선택
        table_name = None
        if self.state['선택된 테이블'] is not None:
            use_selected = self.get_yes_no_input(
                f"현재 선택된 테이블 '{self.state['선택된 테이블']}'을(를) 추출하시겠습니까?", default=True
            )
            if use_selected:
                table_name = self.state['선택된 테이블']
        
        if table_name is None:
            try:
                tables = get_available_tables(self.state['DB 연결'], self.state['DB 유형'])
                
                if not tables:
                    self.show_error("데이터베이스에 테이블이 없습니다.")
                    self.wait_for_user()
                    return
                
                print("사용 가능한 테이블:")
                for i, table in enumerate(tables, 1):
                    print(f"{i}. {table}")
                
                choice = self.get_input("\n추출할 테이블 번호를 입력하세요", "1")
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(tables):
                        table_name = tables[choice_idx]
                    else:
                        self.show_error(f"유효한 번호를 입력하세요 (1-{len(tables)})")
                        self.wait_for_user()
                        return
                except ValueError:
                    self.show_error("숫자를 입력하세요")
                    self.wait_for_user()
                    return
            
            except Exception as e:
                self.show_error(f"테이블 목록 조회 실패: {str(e)}")
                logger.exception("테이블 목록 조회 실패")
                self.wait_for_user()
                return
        
        # 출력 경로 설정
        output_dir = self.export_params['output_path']
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = self.get_input(
            "출력 CSV 파일 경로", 
            output_dir / f"{table_name}.csv"
        )
        
        # 조건 설정 (선택적)
        use_condition = self.get_yes_no_input("\n조건(WHERE 절)을 추가하시겠습니까?", default=False)
        condition = ""
        if use_condition:
            print("WHERE 절에 사용할 조건을 입력하세요:")
            condition = input().strip()
        
        # 정렬 설정 (선택적)
        use_order = self.get_yes_no_input("정렬(ORDER BY 절)을 추가하시겠습니까?", default=False)
        order_by = ""
        if use_order:
            print("ORDER BY 절에 사용할 정렬 기준을 입력하세요:")
            order_by = input().strip()
        
        # 제한 설정 (선택적)
        use_limit = self.get_yes_no_input("행 수 제한(LIMIT 절)을 추가하시겠습니까?", default=False)
        limit = 0
        if use_limit:
            limit = self.get_numeric_input("추출할 최대 행 수", 1000, min_val=1)
        
        # 쿼리 구성
        query = f"SELECT * FROM {table_name}"
        if condition:
            query += f" WHERE {condition}"
        if order_by:
            query += f" ORDER BY {order_by}"
        if use_limit:
            query += f" LIMIT {limit}"
        
        print(f"\n실행할 쿼리:")
        print("-" * 50)
        print(query)
        print("-" * 50)
        
        # 테이블 추출 확인
        execute = self.get_yes_no_input("\n테이블을 추출하시겠습니까?")
        if not execute:
            print("테이블 추출을 취소합니다.")
            self.wait_for_user()
            return
        
        try:
            # 추출 시작
            print("\n테이블 추출 중...")
            
            # 테이블 스키마 가져오기
            cursor = self.state['DB 연결'].cursor()
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
                    print(f"\r{total_rows:,}개 행 추출 완료...", end="")
            
            self.update_state('마지막 추출 파일', output_file)
            self.show_success(f"\n테이블 '{table_name}'의 데이터가 '{output_file}'에 저장되었습니다. ({total_rows:,}개 행)")
            
        except Exception as e:
            self.show_error(f"테이블 추출 실패: {str(e)}")
            logger.exception("테이블 추출 실패")
        
        self.wait_for_user()
    
    def export_settings_menu(self) -> None:
        """추출 설정 메뉴"""
        self.print_header("추출 설정")
        
        print("CSV 추출 관련 설정을 변경합니다.\n")
        
        # 현재 설정 표시
        print("현재 설정:")
        print(f"- CSV 구분자: '{self.export_params['csv_separator']}'")
        print(f"- 헤더 포함: {self.export_params['include_header']}")
        print(f"- 청크 크기: {self.export_params['chunk_size']:,}")
        print(f"- 출력 경로: {self.export_params['output_path']}")
        
        # 설정 변경
        print("\n설정 변경:")
        
        # CSV 구분자
        separators = [",", ";", "\\t", "|", " "]
        print("\nCSV 구분자 선택:")
        for i, sep in enumerate(separators, 1):
            sep_display = sep if sep != "\\t" else "탭"
            print(f"{i}. {sep_display}")
        
        sep_choice = self.get_input("구분자 번호", "1")
        try:
            sep_idx = int(sep_choice) - 1
            if 0 <= sep_idx < len(separators):
                self.export_params['csv_separator'] = separators[sep_idx] if separators[sep_idx] != "\\t" else "\t"
            else:
                self.show_error(f"유효한 번호를 입력하세요 (1-{len(separators)})")
        except ValueError:
            self.show_error("숫자를 입력하세요")
        
        # 헤더 포함 여부
        self.export_params['include_header'] = self.get_yes_no_input(
            "CSV 파일에 헤더(컬럼명)를 포함하시겠습니까?",
            default=self.export_params['include_header']
        )
        
        # 청크 크기
        self.export_params['chunk_size'] = self.get_numeric_input(
            "청크 크기 (한 번에 가져올 행 수)",
            self.export_params['chunk_size'],
            min_val=100,
            max_val=100000
        )
        
        # 출력 경로
        new_output_path = Path(self.get_input())