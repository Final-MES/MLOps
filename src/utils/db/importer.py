"""
데이터베이스 데이터 가져오기 모듈

이 모듈은 다양한 유형의 데이터(모델 결과, 센서 데이터 등)를 데이터베이스에 가져오는
기능을 제공합니다. 특히 sensor_cli.py에서 모델 평가 결과를 DB에 저장하는 기능을 지원합니다.
"""

import os
import json
import logging
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# DB 커넥터 임포트
from src.utils.db.connector import DBConnector

# 로깅 설정
logger = logging.getLogger(__name__)

class DBImporter:
    """
    데이터베이스 데이터 가져오기 클래스
    
    모델 결과, 센서 데이터 등 다양한 유형의 데이터를 데이터베이스에 가져오는 기능을 제공합니다.
    """
    
    def __init__(self, connector: DBConnector):
        """
        DBImporter 초기화
        
        Args:
            connector: 데이터베이스 연결 객체
        """
        self.connector = connector
        logger.info("DBImporter 초기화 완료")
    
    def create_model_results_table(self, table_name: str = "model_results") -> bool:
        """
        모델 결과 저장을 위한 테이블 생성
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            bool: 생성 성공 여부
        """
        # 테이블이 이미 존재하는지 확인
        if self.connector.table_exists(table_name):
            logger.info(f"테이블 '{table_name}'이(가) 이미 존재합니다.")
            return True
        
        # 테이블 스키마 정의
        columns = [
            {'name': 'result_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY AUTOINCREMENT'},
            {'name': 'model_name', 'type': 'VARCHAR(255)', 'constraint': 'NOT NULL'},
            {'name': 'model_type', 'type': 'VARCHAR(50)', 'constraint': ''},
            {'name': 'accuracy', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'test_loss', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'f1_score', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'confusion_matrix', 'type': 'TEXT', 'constraint': ''},
            {'name': 'class_report', 'type': 'TEXT', 'constraint': ''},
            {'name': 'created_at', 'type': 'DATETIME', 'constraint': 'DEFAULT CURRENT_TIMESTAMP'},
            {'name': 'metadata', 'type': 'TEXT', 'constraint': ''}
        ]
        
        # SQLite에서는 AUTOINCREMENT가 다르게 처리됨
        if self.connector.db_type == 'sqlite':
            columns[0] = {'name': 'result_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY'}
        
        # 테이블 생성
        return self.connector.create_table(table_name, columns)
    
    def save_model_results(self, 
                          results: Dict[str, Any], 
                          table_name: str = "model_results", 
                          model_name: str = "unknown_model",
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        모델 평가 결과를 데이터베이스에 저장
        
        Args:
            results: 모델 평가 결과 딕셔너리
            table_name: 테이블 이름
            model_name: 모델 이름
            metadata: 추가 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        # 테이블 생성 확인
        if not self.create_model_results_table(table_name):
            logger.error(f"모델 결과 테이블 생성 실패")
            return False
        
        try:
            # 결과 데이터 가공
            processed_results = {}
            
            # 기본 정보 추출
            processed_results['model_name'] = model_name
            processed_results['model_type'] = results.get('model_type', 'unknown')
            
            # 성능 지표 추출
            processed_results['accuracy'] = results.get('accuracy', 0.0)
            processed_results['test_loss'] = results.get('test_loss', 0.0)
            
            # 여러 가지 f1_score 중 가장 중요한 것 선택
            if 'f1_score' in results:
                if isinstance(results['f1_score'], list):
                    # 리스트인 경우 평균
                    processed_results['f1_score'] = sum(results['f1_score']) / len(results['f1_score'])
                else:
                    processed_results['f1_score'] = results['f1_score']
            elif 'classification_report' in results and isinstance(results['classification_report'], dict):
                # 분류 보고서에서 가중 평균 f1_score 추출
                if 'weighted avg' in results['classification_report']:
                    processed_results['f1_score'] = results['classification_report']['weighted avg'].get('f1-score', 0.0)
                else:
                    processed_results['f1_score'] = 0.0
            else:
                processed_results['f1_score'] = 0.0
            
            # 복잡한 구조를 JSON으로 직렬화
            if 'confusion_matrix' in results:
                # 혼동 행렬이 numpy 배열인 경우 처리
                if hasattr(results['confusion_matrix'], 'tolist'):
                    confusion_matrix = results['confusion_matrix'].tolist()
                else:
                    confusion_matrix = results['confusion_matrix']
                processed_results['confusion_matrix'] = json.dumps(confusion_matrix)
            else:
                processed_results['confusion_matrix'] = None
            
            # 분류 보고서 직렬화
            if 'classification_report' in results:
                processed_results['class_report'] = json.dumps(results['classification_report'])
            else:
                processed_results['class_report'] = None
            
            # 생성 시간
            processed_results['created_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 메타데이터 처리
            if metadata:
                processed_results['metadata'] = json.dumps(metadata)
            else:
                processed_results['metadata'] = None
            
            # 데이터 삽입
            result = self.connector.insert_data(table_name, [processed_results])
            
            if result > 0:
                logger.info(f"모델 결과가 '{table_name}' 테이블에 저장되었습니다.")
                return True
            else:
                logger.error(f"모델 결과 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"모델 결과 저장 중 오류 발생: {str(e)}")
            return False
    
    def create_sensor_data_table(self, table_name: str = "sensor_data") -> bool:
        """
        센서 데이터 저장을 위한 테이블 생성
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            bool: 생성 성공 여부
        """
        # 테이블이 이미 존재하는지 확인
        if self.connector.table_exists(table_name):
            logger.info(f"테이블 '{table_name}'이(가) 이미 존재합니다.")
            return True
        
        # 테이블 스키마 정의
        columns = [
            {'name': 'data_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY AUTOINCREMENT'},
            {'name': 'timestamp', 'type': 'DATETIME', 'constraint': 'DEFAULT CURRENT_TIMESTAMP'},
            {'name': 'sensor_id', 'type': 'VARCHAR(50)', 'constraint': ''},
            {'name': 'sensor_type', 'type': 'VARCHAR(50)', 'constraint': ''},
            {'name': 'value', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'predicted_value', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'predicted_class', 'type': 'VARCHAR(50)', 'constraint': ''},
            {'name': 'confidence', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'metadata', 'type': 'TEXT', 'constraint': ''}
        ]
        
        # SQLite에서는 AUTOINCREMENT가 다르게 처리됨
        if self.connector.db_type == 'sqlite':
            columns[0] = {'name': 'data_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY'}
        
        # 테이블 생성
        return self.connector.create_table(table_name, columns)
    
    def save_sensor_data(self, 
                        data: Dict[str, Any], 
                        prediction: Optional[Dict[str, Any]] = None,
                        table_name: str = "sensor_data",
                        metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        센서 데이터 및 예측 결과를 데이터베이스에 저장
        
        Args:
            data: 센서 데이터 {'sensor_id': 'sensor1', 'sensor_type': 'temperature', 'value': 25.5}
            prediction: 예측 결과 {'predicted_value': 26.0, 'predicted_class': 'normal', 'confidence': 0.95}
            table_name: 테이블 이름
            metadata: 추가 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        # 테이블 생성 확인
        if not self.create_sensor_data_table(table_name):
            logger.error(f"센서 데이터 테이블 생성 실패")
            return False
        
        try:
            # 데이터 가공
            processed_data = data.copy()
            
            # 현재 시간 추가
            if 'timestamp' not in processed_data:
                processed_data['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 예측 데이터 추가
            if prediction:
                processed_data.update(prediction)
            
            # 메타데이터 처리
            if metadata:
                processed_data['metadata'] = json.dumps(metadata)
            
            # 데이터 삽입
            result = self.connector.insert_data(table_name, [processed_data])
            
            if result > 0:
                logger.info(f"센서 데이터가 '{table_name}' 테이블에 저장되었습니다.")
                return True
            else:
                logger.error(f"센서 데이터 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"센서 데이터 저장 중 오류 발생: {str(e)}")
            return False
    
    def save_batch_sensor_data(self, 
                              data_list: List[Dict[str, Any]], 
                              predictions: Optional[List[Dict[str, Any]]] = None,
                              table_name: str = "sensor_data",
                              metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        센서 데이터 및 예측 결과를 배치로 데이터베이스에 저장
        
        Args:
            data_list: 센서 데이터 목록
            predictions: 예측 결과 목록 (data_list와 길이가 같아야 함)
            table_name: 테이블 이름
            metadata: 추가 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        # 테이블 생성 확인
        if not self.create_sensor_data_table(table_name):
            logger.error(f"센서 데이터 테이블 생성 실패")
            return False
        
        # 예측 데이터 확인
        if predictions and len(predictions) != len(data_list):
            logger.error(f"데이터와 예측 결과의 길이가 다릅니다: {len(data_list)} vs {len(predictions)}")
            return False
        
        try:
            # 데이터 가공
            processed_data_list = []
            
            for i, data in enumerate(data_list):
                processed_data = data.copy()
                
                # 현재 시간 추가
                if 'timestamp' not in processed_data:
                    processed_data['timestamp'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                # 예측 데이터 추가
                if predictions:
                    processed_data.update(predictions[i])
                
                # 메타데이터 처리
                if metadata:
                    processed_data['metadata'] = json.dumps(metadata)
                
                processed_data_list.append(processed_data)
            
            # 데이터 삽입
            result = self.connector.insert_data(table_name, processed_data_list)
            
            if result > 0:
                logger.info(f"{result}개의 센서 데이터가 '{table_name}' 테이블에 저장되었습니다.")
                return True
            else:
                logger.error(f"센서 데이터 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"센서 데이터 저장 중 오류 발생: {str(e)}")
            return False
    
    def create_prediction_history_table(self, table_name: str = "prediction_history") -> bool:
        """
        모델 예측 이력 저장을 위한 테이블 생성
        
        Args:
            table_name: 테이블 이름
            
        Returns:
            bool: 생성 성공 여부
        """
        # 테이블이 이미 존재하는지 확인
        if self.connector.table_exists(table_name):
            logger.info(f"테이블 '{table_name}'이(가) 이미 존재합니다.")
            return True
        
        # 테이블 스키마 정의
        columns = [
            {'name': 'prediction_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY AUTOINCREMENT'},
            {'name': 'timestamp', 'type': 'DATETIME', 'constraint': 'DEFAULT CURRENT_TIMESTAMP'},
            {'name': 'model_name', 'type': 'VARCHAR(255)', 'constraint': ''},
            {'name': 'input_data', 'type': 'TEXT', 'constraint': ''},
            {'name': 'prediction', 'type': 'TEXT', 'constraint': ''},
            {'name': 'actual_value', 'type': 'TEXT', 'constraint': ''},
            {'name': 'accuracy', 'type': 'FLOAT', 'constraint': ''},
            {'name': 'is_correct', 'type': 'BOOLEAN', 'constraint': ''},
            {'name': 'metadata', 'type': 'TEXT', 'constraint': ''}
        ]
        
        # SQLite에서는 AUTOINCREMENT가 다르게 처리됨
        if self.connector.db_type == 'sqlite':
            columns[0] = {'name': 'prediction_id', 'type': 'INTEGER', 'constraint': 'PRIMARY KEY'}
        
        # 테이블 생성
        return self.connector.create_table(table_name, columns)
    
    def save_prediction_history(self, 
                               model_name: str,
                               input_data: Any,
                               prediction: Any,
                               actual_value: Optional[Any] = None,
                               accuracy: Optional[float] = None,
                               is_correct: Optional[bool] = None,
                               table_name: str = "prediction_history",
                               metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        모델 예측 이력을 데이터베이스에 저장
        
        Args:
            model_name: 모델 이름
            input_data: 입력 데이터
            prediction: 예측 결과
            actual_value: 실제 값 (선택 사항)
            accuracy: 정확도 (선택 사항)
            is_correct: 예측 정확성 여부 (선택 사항)
            table_name: 테이블 이름
            metadata: 추가 메타데이터
            
        Returns:
            bool: 저장 성공 여부
        """
        # 테이블 생성 확인
        if not self.create_prediction_history_table(table_name):
            logger.error(f"예측 이력 테이블 생성 실패")
            return False
        
        try:
            # 데이터 가공
            processed_data = {
                'model_name': model_name,
                'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # 복잡한 객체는 JSON으로 직렬화
            try:
                processed_data['input_data'] = self._serialize_object(input_data)
                processed_data['prediction'] = self._serialize_object(prediction)
                
                if actual_value is not None:
                    processed_data['actual_value'] = self._serialize_object(actual_value)
            except Exception as e:
                logger.error(f"데이터 직렬화 중 오류 발생: {str(e)}")
                processed_data['input_data'] = str(input_data)
                processed_data['prediction'] = str(prediction)
                if actual_value is not None:
                    processed_data['actual_value'] = str(actual_value)
            
            # 정확도 및 정확성 정보
            if accuracy is not None:
                processed_data['accuracy'] = accuracy
            if is_correct is not None:
                processed_data['is_correct'] = is_correct
            
            # 메타데이터 처리
            if metadata:
                processed_data['metadata'] = json.dumps(metadata)
            
            # 데이터 삽입
            result = self.connector.insert_data(table_name, [processed_data])
            
            if result > 0:
                logger.info(f"예측 이력이 '{table_name}' 테이블에 저장되었습니다.")
                return True
            else:
                logger.error(f"예측 이력 저장 실패")
                return False
                
        except Exception as e:
            logger.error(f"예측 이력 저장 중 오류 발생: {str(e)}")
            return False
    
    def _serialize_object(self, obj: Any) -> str:
        """
        객체를 JSON 문자열로 직렬화
        
        Args:
            obj: 직렬화할 객체
            
        Returns:
            str: 직렬화된 JSON 문자열
        """
        # NumPy 배열 처리
        if isinstance(obj, np.ndarray):
            return json.dumps(obj.tolist())
        
        # NumPy 데이터 타입 처리
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return json.dumps(int(obj))
        
        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return json.dumps(float(obj))
        
        if isinstance(obj, (np.bool_)):
            return json.dumps(bool(obj))
        
        # 리스트, 딕셔너리 등의 일반 객체 처리
        try:
            return json.dumps(obj)
        except TypeError:
            # JSON으로 직렬화할 수 없는 경우 문자열로 변환
            return str(obj)

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # DB 연결
    from src.utils.db.connector import DBConnector
    
    connector = DBConnector()