#!/usr/bin/env python
"""
데이터베이스 추출 CLI 모듈

이 모듈은 SQL 데이터베이스에서 데이터를 CSV로 추출하기 위한 대화형 인터페이스를 제공합니다.
다양한 데이터베이스 유형(MySQL, PostgreSQL, SQLite 등)을 지원하며,
src/utils/db 패키지의 기능들을 활용합니다.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 기본 CLI 클래스 임포트
from src.cli.base_cli import BaseCLI

# 데이터베이스 유틸리티 임포트
from src.utils.db.connector import DBConnector
from src.utils.db.exporter import DBExporter
from src.utils.config_loader import load_db_config, load_export_settings

# 로깅 설정
logger = logging.getLogger(__name__)

class DBExportCLI(BaseCLI):
    """
    데이터베이스 추출 CLI 클래스
    
    SQL 데이터베이스에서 CSV로 데이터를 추출하는 기능을 제공합니다.
    src/utils/db 패키지의 DBConnector와 DBExporter를 활용합니다.
    """
    
    def __init__(self, config_profile: str = "default"):
        """
        데이터베이스 추출 CLI 초기화
        
        Args:
            config_profile: 설정 프로필명 (default, development, production 등)
        """
        super().__init__(title="SQL 데이터베이스 CSV 추출 도구")
        
        # 상태 초기화
        self.state = {
            '설정 프로필': config_profile,
            '마지막 추출 파일': None,
            '선택된 테이블': None
        }
        
        # DB 커넥터 초기화
        self.db_connector = DBConnector(config_profile)
        
        # DB 익스포터 초기화
        self.db_exporter = None  # 연결 후 초기화
        
        # 외부 설정 파일에서 추출 설정 로드
        self.export_params = load_export_settings()
        
        # 출력 경로가 Path 객체가 아니면 변환
        if not isinstance(self.export_params['output_path'], Path):
            self.export_params['output_path'] = Path(self.export_params['output_path'])
        
        # 출력 디렉토리 생성
        os.makedirs(self.export_params['output_path'], exist_ok=True)
        
        logger.info(f"데이터베이스 추출 CLI 초기화 완료 (프로필: {config_profile})")
    
    def print_status(self) -> None:
        """현재 상태 출력"""
        print("\n현재 상태:")
        print("-" * 40)
        
        # DB 연결 상태
        if self.db_connector.is_connected():
            print(f"✅ DB 연결: {self.db_connector.db_type} 데이터베이스에 연결됨")
        else:
            print("❌ DB 연결: 연결되지 않음")
        
        # 선택된 테이블
        if self.state['선택된 테이블']:
            print(f"✅ 선택된 테이블: {self.state['선택된 테이블']}")
        else:
            print("❌ 선택된 테이블: 없음")
        
        # 마지막 추출 파일
        if self.state['마지막 추출 파일']:
            print(f"✅ 마지막 추출 파일: {self.state['마지막 추출 파일']}")
        else:
            print("❌ 마지막 추출 파일: 없음")
            
        # 설정 프로필
        print(f"⚙️ 설정 프로필: {self.state['설정 프로필']}")
        
        print("-" * 40)
    
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
                "설정 프로필 변경",
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
                self.change_profile_menu()
            elif choice == 7:
                print("\n프로그램을 종료합니다. 감사합니다!")
                # 연결 닫기
                if self.db_connector.is_connected():
                    self.db_connector.close()
                    print("데이터베이스 연결을 닫았습니다.")
                break
    
    def change_profile_menu(self) -> None:
        """설정 프로필 변경 메뉴"""
        self.print_header("설정 프로필 변경")
        
        print("데이터베이스 연결에 사용할 설정 프로필을 변경합니다.")
        print("설정 프로필은 config/db_config.json 파일에 정의되어 있습니다.\n")
        
        # 현재 프로필 표시
        current_profile = self.state['설정 프로필']
        print(f"현재 프로필: {current_profile}\n")
        
        # 기본 프로필 목록
        profiles = ["default", "development", "production"]
        
        # 설정 파일에서 추가 프로필 확인
        try:
            config_path = os.path.join(project_root, "config", "db_config.json")
            
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 프로필 목록 업데이트
                if "connections" in config:
                    profiles = list(config["connections"].keys())
        except:
            # 오류 발생 시 기본 프로필 목록 사용
            pass
        
        # 프로필 선택 메뉴
        print("사용 가능한 프로필:")
        for i, profile in enumerate(profiles, 1):
            if profile == current_profile:
                print(f"{i}. {profile} (현재)")
            else:
                print(f"{i}. {profile}")
        
        # 사용자 입력
        choice = self.get_input("\n사용할 프로필 번호 또는 이름", "1")
        
        # 번호로 입력한 경우
        try:
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(profiles):
                new_profile = profiles[choice_idx]
            else:
                self.show_error(f"유효한 번호를 입력하세요 (1-{len(profiles)})")
                self.wait_for_user()
                return
        except ValueError:
            # 이름으로 입력한 경우
            new_profile = choice.strip()
        
        # 프로필 변경
        if new_profile != current_profile:
            # 기존 연결 닫기
            if self.db_connector.is_connected():
                self.db_connector.close()
            
            # 새 프로필로 커넥터 초기화
            self.db_connector = DBConnector(new_profile)
            self.db_exporter = None  # 연결 후 다시 초기화됨
            
            # 상태 업데이트
            self.update_state('설정 프로필', new_profile)
            self.update_state('선택된 테이블', None)
            
            self.show_success(f"설정 프로필을 '{new_profile}'(으)로 변경했습니다.")
        else:
            self.show_message(f"현재 사용 중인 프로필 '{current_profile}'을(를) 유지합니다.")
        
        self.wait_for_user()
    
    def db_connection_menu(self) -> None:
        """데이터베이스 연결 설정 메뉴"""
        self.print_header("데이터베이스 연결 설정")
        
        print("데이터베이스 연결 정보를 설정합니다.\n")
        
        # 현재 프로필 표시
        print(f"현재 설정 프로필: {self.state['설정 프로필']}")
        
        # 기존 연결 닫기
        if self.db_connector.is_connected():
            self.db_connector.close()
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
                if self.db_connector.connect(db_type_lower, database=db_file):
                    # 연결 성공 시 익스포터 초기화
                    self.db_exporter = DBExporter(self.db_connector)
                    self.db_exporter.set_export_params(self.export_params)
                    
                    self.show_success(f"SQLite 데이터베이스 '{db_file}'에 연결되었습니다.")
                else:
                    self.show_error("SQLite 데이터베이스 연결 실패")
            except Exception as e:
                self.show_error(f"SQLite 데이터베이스 연결 실패: {str(e)}")
                logger.exception("SQLite 연결 실패")
            
            self.wait_for_user()
            return
        
        # 다른 데이터베이스는 연결 정보 필요
        print(f"\n{db_types[db_type_idx]} 연결 정보 입력:")
        
        # 연결 정보 입력 (설정 파일의 값을 기본값으로 사용)
        connection_params = self.db_connector.get_connection_params()
        
        host = self.get_input("호스트", connection_params.get('host', 'localhost'))
        
        # 포트 기본값 설정
        default_ports = {
            'mysql': 3306, 'postgresql': 5432, 'sqlserver': 1433, 'oracle': 1521
        }
        default_port = default_ports.get(db_type_lower, 3306)
        
        # 포트 처리
        if 'port' in connection_params and isinstance(connection_params['port'], dict):
            if db_type_lower in connection_params['port']:
                default_port = connection_params['port'][db_type_lower]
        
        port = self.get_numeric_input("포트", default_port, min_val=1, max_val=65535)
        database = self.get_input("데이터베이스 이름", connection_params.get('database', ''))
        username = self.get_input("사용자 이름", connection_params.get('username', ''))
        
        # 비밀번호는 설정에 값이 있어도 표시하지 않음
        password = self.get_input("비밀번호 (입력하지 않으면 설정 파일의 값 사용)")
        if not password and 'password' in connection_params:
            password = connection_params['password']
        
        # 데이터베이스 연결 시도
        try:
            if self.db_connector.connect(
                db_type=db_type_lower,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password
            ):
                # 연결 성공 시 익스포터 초기화
                self.db_exporter = DBExporter(self.db_connector)
                self.db_exporter.set_export_params(self.export_params)
                
                self.show_success(f"{db_types[db_type_idx]} 데이터베이스에 연결되었습니다.")
            else:
                self.show_error("데이터베이스 연결 실패")
            
        except Exception as e:
            self.show_error(f"데이터베이스 연결 실패: {str(e)}")
            logger.exception("데이터베이스 연결 실패")
        
        self.wait_for_user()
    
    def list_tables_menu(self) -> None:
        """테이블 목록 보기 메뉴"""
        self.print_header("테이블 목록")
        
        # 연결 확인
        if not self.db_connector.is_connected():
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        try:
            tables = self.db_connector.get_tables()
            
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
                            count = self.db_connector.get_row_count(selected_table)
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
        if not self.db_connector.is_connected():
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
                tables = self.db_connector.get_tables()
                
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
            schema = self.db_connector.get_schema(table_name)
            
            print(f"\n테이블 '{table_name}'의 스키마:\n")
            print("-" * 50)
            print(f"{'컬럼명':<20} {'데이터 유형':<15} {'기타 속성':<15}")
            print("-" * 50)
            
            for column in schema:
                print(f"{column['name']:<20} {column['type']:<15} {column.get('extra', ''):<15}")
            
            print("-" * 50)
            
            # 행 수 표시
            count = self.db_connector.get_row_count(table_name)
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
        
        # 연결 및 익스포터 확인
        if not self.db_connector.is_connected():
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        if self.db_exporter is None:
            self.db_exporter = DBExporter(self.db_connector)
            self.db_exporter.set_export_params(self.export_params)
        
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
            # DBExporter를 사용하여 쿼리 실행 및 결과 추출
            result_path = self.db_exporter.export_query_to_csv(query, output_file)
            
            if result_path:
                self.update_state('마지막 추출 파일', str(result_path))
                self.show_success(f"쿼리 결과가 '{result_path}'에 저장되었습니다.")
            else:
                self.show_error("쿼리 실행 또는 결과 추출에 실패했습니다.")
            
        except Exception as e:
            self.show_error(f"쿼리 실행 실패: {str(e)}")
            logger.exception("쿼리 실행 실패")
        
        self.wait_for_user()
    
    def table_export_menu(self) -> None:
        """테이블 직접 CSV 추출 메뉴"""
        self.print_header("테이블 직접 CSV 추출")
        
        # 연결 및 익스포터 확인
        if not self.db_connector.is_connected():
            self.show_error("데이터베이스 연결이 설정되지 않았습니다. 먼저 연결 설정을 수행하세요.")
            self.wait_for_user()
            return
        
        if self.db_exporter is None:
            self.db_exporter = DBExporter(self.db_connector)
            self.db_exporter.set_export_params(self.export_params)
        
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
                tables = self.db_connector.get_tables()
                
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
        condition = None
        if use_condition:
            print("WHERE 절에 사용할 조건을 입력하세요:")
            condition = input().strip()
        
        # 정렬 설정 (선택적)
        use_order = self.get_yes_no_input("정렬(ORDER BY 절)을 추가하시겠습니까?", default=False)
        order_by = None
        if use_order:
            print("ORDER BY 절에 사용할 정렬 기준을 입력하세요:")
            order_by = input().strip()
        
        # 제한 설정 (선택적)
        use_limit = self.get_yes_no_input("행 수 제한(LIMIT 절)을 추가하시겠습니까?", default=False)
        limit = None
        if use_limit:
            limit = self.get_numeric_input("추출할 최대 행 수", 1000, min_val=1)
        
        # 테이블 추출 확인
        execute = self.get_yes_no_input("\n테이블을 추출하시겠습니까?")
        if not execute:
            print("테이블 추출을 취소합니다.")
            self.wait_for_user()
            return
        
        try:
            # 추출 시작
            print("\n테이블 추출 중...")
            
            # DBExporter를 사용하여 테이블 추출
            result_path = self.db_exporter.export_table_to_csv(
                table_name=table_name,
                output_file=output_file,
                condition=condition,
                order_by=order_by,
                limit=limit
            )
            
            if result_path:
                self.update_state('마지막 추출 파일', str(result_path))
                self.update_state('선택된 테이블', table_name)
                self.show_success(f"\n테이블 '{table_name}'의 데이터가 '{result_path}'에 저장되었습니다.")
            else:
                self.show_error("\n테이블 추출에 실패했습니다.")
            
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
        output_path_str = self.get_input("출력 디렉토리 경로", str(self.export_params['output_path']))
        self.export_params['output_path'] = Path(output_path_str)
        os.makedirs(self.export_params['output_path'], exist_ok=True)
        
        # 익스포터가 있으면 설정 업데이트
        if self.db_exporter:
            self.db_exporter.set_export_params(self.export_params)
        
        # 현재 설정을 설정 파일에 저장할지 여부
        save_to_config = self.get_yes_no_input("\n변경된 설정을 설정 파일에 저장하시겠습니까?", default=False)
        if save_to_config:
            try:
                # 설정 파일 경로
                config_path = os.path.join(project_root, "config", "db_config.json")
                
                # 설정 파일 로드
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # 추출 설정 업데이트
                config["export_settings"] = {
                    "csv_separator": self.export_params['csv_separator'],
                    "include_header": self.export_params['include_header'],
                    "chunk_size": self.export_params['chunk_size'],
                    "output_path": str(self.export_params['output_path'])
                }
                
                # 설정 파일 저장
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=4)
                
                self.show_success("설정이 파일에 저장되었습니다.")
            except Exception as e:
                self.show_error(f"설정 저장 실패: {str(e)}")
                logger.exception("설정 저장 실패")
        else:
            self.show_success("추출 설정이 변경되었습니다. (이 세션에만 적용)")
        
        self.wait_for_user()
    
    def run(self) -> None:
        """CLI 실행"""
        try:
            # 메인 메뉴 실행
            self.main_menu()
        except KeyboardInterrupt:
            print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
            # 연결 닫기
            if self.db_connector.is_connected():
                self.db_connector.close()
        except Exception as e:
            self.show_error(f"예상치 못한 오류가 발생했습니다: {str(e)}")
            logger.exception("예상치 못한 오류 발생")
            # 연결 닫기
            if self.db_connector.is_connected():
                self.db_connector.close()

def main():
    """메인 함수: CLI 실행"""
    # 로깅 설정
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(log_dir, 'db_export.log'))
        ]
    )
    
    # 명령줄 인자 파싱
    import argparse
    parser = argparse.ArgumentParser(description='SQL 데이터베이스에서 CSV로 데이터를 추출하는 도구')
    parser.add_argument('--profile', type=str, default='default',
                      help='사용할 설정 프로필 (default, development, production)')
    parser.add_argument('--config', type=str, default=None,
                      help='설정 파일 경로 (기본값: config/db_config.json)')
    
    args = parser.parse_args()
    
    try:
        # CLI 인스턴스 생성 및 실행
        cli = DBExportCLI(config_profile=args.profile)
        cli.run()
        
        return 0  # 성공적인 종료
        
    except Exception as e:
        logger.critical(f"치명적 오류 발생: {str(e)}", exc_info=True)
        print(f"\n❌ 치명적 오류 발생: {str(e)}")
        print("로그 파일을 확인하세요.")
        return 1  # 오류 종료

if __name__ == "__main__":
    sys.exit(main())