#!/usr/bin/env python
"""
실시간 센서 모니터링 및 SQL 서버 데이터 전송 스크립트

이 스크립트는 g2 센서 데이터를 실시간으로 처리하고, 
처리된 결과를 SQL 서버 데이터베이스에 저장합니다.
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 필요한 모듈 임포트
from src.data.sensor.sensor_processor import SensorDataProcessor
from src.utils.db.connector import DBConnector
from src.utils.db.importer import DBImporter
from src.models.sensor.lstm_classifier import MultiSensorLSTMClassifier

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'realtime_monitoring.log'))
    ]
)
logger = logging.getLogger(__name__)


class RealtimeMonitor:
    """실시간 센서 모니터링 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 설정 정보 딕셔너리
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/raw'))
        self.model_path = Path(config.get('model_path', 'models/sensor_classifier.pth'))
        self.db_profile = config.get('db_profile', 'default')
        self.input_size = config.get('input_size', 4)  # 센서 수
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.num_classes = config.get('num_classes', 4)
        self.sequence_length = config.get('sequence_length', 100)
        self.window_size = config.get('window_size', 15)
        self.sample_interval = config.get('sample_interval', 1.0)  # 초 단위
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 프로세서 초기화
        self.processor = SensorDataProcessor(
            interpolation_step=config.get('interp_step', 0.001),
            window_size=self.window_size
        )
        
        # DB 연결 초기화
        self.db_connector = None
        self.db_importer = None
        
        # 모델 초기화
        self.model = None
        
        # 누적 데이터 (시퀀스 구성용)
        self.accumulated_data = []
        
        logger.info("실시간 모니터링 초기화 완료")

    def load_model(self) -> bool:
        """
        모델 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            logger.info(f"모델 로드 중: {self.model_path}")
            
            # 모델 인스턴스 생성
            self.model = MultiSensorLSTMClassifier(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                num_classes=self.num_classes
            ).to(self.device)
            
            # 가중치 로드
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.eval()
            
            logger.info("모델 로드 완료")
            return True
            
        except Exception as e:
            logger.error(f"모델 로드 실패: {str(e)}")
            return False

    def connect_to_db(self) -> bool:
        """
        데이터베이스 연결 (기존 테이블 스키마 사용)
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            logger.info(f"DB 연결 시도 (프로필: {self.db_profile})")
            
            # DB 커넥터 초기화
            self.db_connector = DBConnector(config_profile=self.db_profile)
            
            # 데이터베이스 연결 (설정에 따라 달라짐)
            db_type = self.config.get('db_type', 'sqlserver')  # SQL 서버가 기본값
            host = self.config.get('db_host', 'localhost')
            port = self.config.get('db_port', 1433)  # SQL 서버 기본 포트
            database = self.config.get('db_name', 'sensor_monitoring')
            username = self.config.get('db_user', 'sa')
            password = self.config.get('db_password', '')
            
            # 연결 시도
            if self.db_connector.connect(
                db_type=db_type,
                host=host,
                port=port,
                database=database,
                username=username,
                password=password
            ):
                logger.info("DB 연결 완료")
                
                # 테이블 존재 확인
                try:
                    # classification_results 테이블이 존재하는지 확인
                    tables = self.db_connector.get_tables()
                    if 'classification_results' not in tables:
                        logger.warning("'classification_results' 테이블이 존재하지 않습니다. 올바른 데이터베이스에 연결되었는지 확인하세요.")
                except Exception as e:
                    logger.warning(f"테이블 확인 중 오류 발생: {str(e)}")
                
                return True
            else:
                logger.error("DB 연결 실패")
                return False
                
        except Exception as e:
            logger.error(f"DB 연결 중 오류 발생: {str(e)}")
            return False

    def process_g2_data(self) -> Optional[np.ndarray]:
        """
        g2 센서 데이터 처리
        
        Returns:
            Optional[np.ndarray]: 처리된 데이터 또는 실패 시 None
        """
        try:
            # g2 파일 로드 및 보간
            logger.info("g2 센서 데이터 로드 및 보간 중...")
            interpolated_data = self.processor.process_g2_realtime_data(
                self.data_dir, prefix='g2'
            )
            
            if not interpolated_data:
                logger.error("g2 센서 데이터를 로드할 수 없습니다.")
                return None
            
            # 데이터 결합 및 전처리
            logger.info("센서 데이터 결합 및 전처리 중...")
            processed_data = self.processor.combine_and_preprocess_sensor_data(interpolated_data)
            
            # 데이터 형태 (result['normal'], result['type1'] 등)
            # 여기서는 최신 데이터 (마지막 window_size 샘플)을 사용
            latest_data = {}
            
            for state, data in processed_data.items():
                latest_data[state] = data[-self.window_size:]
            
            # 현재 상태 추론을 위한 데이터 생성 
            # (여기서는 normal 상태 데이터를 사용)
            current_data = latest_data['normal']
            
            logger.info(f"데이터 처리 완료: 형태={current_data.shape}")
            return current_data
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return None

    def predict_state(self, sensor_data: np.ndarray) -> Tuple[int, float]:
        """
        센서 데이터로부터 상태 예측
        
        Args:
            sensor_data: 센서 데이터 배열
            
        Returns:
            Tuple[int, float]: (예측 클래스, 신뢰도)
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다.")
            return -1, 0.0
        
        # 누적 데이터에 추가
        self.accumulated_data.append(sensor_data)
        
        # 시퀀스 길이 제한
        if len(self.accumulated_data) > self.sequence_length:
            self.accumulated_data = self.accumulated_data[-self.sequence_length:]
        
        # 시퀀스 길이가 충분하지 않으면 예측 불가
        if len(self.accumulated_data) < self.sequence_length:
            logger.warning(f"누적 데이터가 부족합니다: {len(self.accumulated_data)}/{self.sequence_length}")
            return -1, 0.0
        
        try:
            # 예측용 텐서 생성
            sequence_data = np.array(self.accumulated_data, dtype=np.float32)
            sequence_tensor = torch.tensor(sequence_data, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # 예측 수행
            with torch.no_grad():
                outputs = self.model(sequence_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # CPU로 변환
            predicted_class = predicted.item()
            confidence_value = confidence.item()
            
            logger.info(f"상태 예측: 클래스={predicted_class}, 신뢰도={confidence_value:.4f}")
            return predicted_class, confidence_value
            
        except Exception as e:
            logger.error(f"예측 중 오류 발생: {str(e)}")
            return -1, 0.0

    def save_to_db(self, sensor_data: np.ndarray, prediction: Tuple[int, float]) -> bool:
        """
        분류 결과만 DB에 저장 (기존 테이블 스키마에 맞춤)
        
        Args:
            sensor_data: 센서 데이터 배열 (저장용이 아닌 참조용)
            prediction: (예측 클래스, 신뢰도) 튜플
            
        Returns:
            bool: 저장 성공 여부
        """
        if self.db_connector is None:
            logger.error("DB 연결이 설정되지 않았습니다.")
            return False
        
        try:
            # 현재 시간 생성
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # 예측 결과 추출
            predicted_class, confidence = prediction
            
            # 유효한 예측인 경우만 저장
            if predicted_class >= 0:
                # 클래스 매핑 (0: 'normal', 1: 'type1' 등)
                class_names = ['normal', 'type1', 'type2', 'type3']
                if 0 <= predicted_class < len(class_names):
                    predicted_label = class_names[predicted_class]
                else:
                    predicted_label = f'unknown_{predicted_class}'
                
                # 직접 SQL 쿼리 실행 (정해진 테이블 스키마에 맞춤)
                # SQL Server에서는 파라미터 바인딩이 ?가 아닌 @p1, @p2 등을 사용합니다
                if self.db_connector.db_type == 'sqlserver':
                    query = "INSERT INTO classification_results (timestamp, result) VALUES (@p1, @p2)"
                else:
                    query = "INSERT INTO classification_results (timestamp, result) VALUES (?, ?)"
                
                # 쿼리 파라미터
                params = (timestamp, predicted_label)
                
                # 쿼리 실행
                result = self.db_connector.execute_query(query, params)
                
                if result > 0:
                    logger.info(f"DB에 분류 결과 저장 완료: 타임스탬프={timestamp}, 결과={predicted_label}")
                    return True
                else:
                    logger.error("DB 저장 실패")
                    return False
            else:
                logger.warning("유효한 예측 결과가 없어 DB에 저장하지 않음")
                return False
                
        except Exception as e:
            logger.error(f"DB 저장 중 오류 발생: {str(e)}")
            return False

    def run(self, duration: int = -1) -> None:
        """
        실시간 모니터링 실행
        
        Args:
            duration: 실행 시간 (초), -1이면 무한 실행
        """
        # 모델 로드
        if not self.load_model():
            logger.error("모델 로드 실패로 종료합니다.")
            return
        
        # DB 연결
        if not self.connect_to_db():
            logger.error("DB 연결 실패로 종료합니다.")
            return
        
        # 실행 시간 계산
        start_time = time.time()
        end_time = start_time + duration if duration > 0 else float('inf')
        
        logger.info(f"실시간 모니터링 시작 (간격: {self.sample_interval}초, 기간: {duration if duration > 0 else '무한'}초)")
        
        try:
            # 메인 루프
            sample_count = 0
            while time.time() < end_time:
                loop_start = time.time()
                
                # 데이터 처리
                sensor_data = self.process_g2_data()
                
                if sensor_data is not None:
                    # 상태 예측
                    prediction = self.predict_state(sensor_data)
                    
                    # DB 저장
                    self.save_to_db(sensor_data, prediction)
                    
                    sample_count += 1
                    logger.info(f"샘플 {sample_count}개 처리 완료")
                
                # 간격 조정
                elapsed = time.time() - loop_start
                if elapsed < self.sample_interval:
                    time.sleep(self.sample_interval - elapsed)
                
            logger.info(f"실행 시간 {duration}초 완료, 총 {sample_count}개 샘플 처리")
            
        except KeyboardInterrupt:
            logger.info("사용자에 의해 중단되었습니다.")
        except Exception as e:
            logger.error(f"실시간 모니터링 중 오류 발생: {str(e)}")
        finally:
            # DB 연결 종료
            if self.db_connector:
                self.db_connector.close()
            
            logger.info("실시간 모니터링 종료")


def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='g2 센서 데이터 실시간 모니터링 및 SQL 서버 데이터 전송')
    
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='센서 데이터 디렉토리')
    parser.add_argument('--model_path', type=str, default='models/sensor_classifier.pth',
                      help='모델 파일 경로')
    parser.add_argument('--db_profile', type=str, default='default',
                      help='데이터베이스 설정 프로필')
    parser.add_argument('--db_type', type=str, default='mysql',
                      help='데이터베이스 유형 (mysql, postgresql, sqlite, sqlserver, oracle)')
    parser.add_argument('--db_host', type=str, default='localhost',
                      help='데이터베이스 호스트')
    parser.add_argument('--db_port', type=int, default=3306,
                      help='데이터베이스 포트')
    parser.add_argument('--db_name', type=str, default='sensor_monitoring',
                      help='데이터베이스 이름')
    parser.add_argument('--db_user', type=str, default='root',
                      help='데이터베이스 사용자')
    parser.add_argument('--db_password', type=str, default='',
                      help='데이터베이스 비밀번호')
    parser.add_argument('--sequence_length', type=int, default=100,
                      help='시퀀스 길이')
    parser.add_argument('--sample_interval', type=float, default=1.0,
                      help='샘플링 간격 (초)')
    parser.add_argument('--duration', type=int, default=-1,
                      help='실행 시간 (초), -1이면 무한 실행')
    
    args = parser.parse_args()
    
    # 설정 딕셔너리 생성
    config = vars(args)
    
    # 모니터 생성 및 실행
    monitor = RealtimeMonitor(config)
    monitor.run(duration=args.duration)


if __name__ == "__main__":
    main()