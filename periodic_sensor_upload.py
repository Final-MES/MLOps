#!/usr/bin/env python
"""
주기적 센서 데이터 분류 및 업로드 스크립트

이 스크립트는 다음 기능을 수행합니다:
1. 일정 간격으로 센서 데이터 로드 및 전처리
2. 저장된 모델을 사용하여 센서 데이터 분류
3. 분류 결과 및 센서 데이터를 API 서버에 업로드
4. 설정된 간격으로 주기적으로 반복 실행
"""

import os
import sys
import time
import json
import logging
import argparse
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import torch
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 필요한 모듈 임포트
from src.data.sensor.sensor_processor import SensorDataProcessor, prepare_sequence_data
from src.models.sensor.lstm_classifier import MultiSensorLSTMClassifier

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'sensor_upload.log'))
    ]
)
logger = logging.getLogger(__name__)

# API 엔드포인트 설정
API_URL_INSERT = "http://3.34.90.243:8000/vibration-diagnosis/bulk"  # 대량 업로드 엔드포인트
API_URL_COUNT = "http://3.34.90.243:8000/vibration-diagnosis"  # 전체 진단 데이터 조회 엔드포인트

class SensorDataUploader:
    """센서 데이터 분류 및 업로드 클래스"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        초기화
        
        Args:
            config: 설정 정보 딕셔너리
        """
        self.config = config
        self.data_dir = Path(config.get('data_dir', 'data/raw'))
        self.model_path = Path(config.get('model_path', 'models/sensor_classifier.pth'))
        self.model_info_path = Path(config.get('model_info_path', 'models/model_info.json'))
        self.input_size = config.get('input_size', 4)  # 센서 수
        self.hidden_size = config.get('hidden_size', 64)
        self.num_layers = config.get('num_layers', 2)
        self.num_classes = config.get('num_classes', 4)
        self.sequence_length = config.get('sequence_length', 50)
        self.window_size = config.get('window_size', 15)
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.batch_size = config.get('batch_size', 1000)  # API 업로드 배치 크기
        
        # 주기 설정 (분 단위)
        self.interval_minutes = config.get('interval_minutes', 30) 
        
        # 여러 센서 파일 접두사 지정 (g1, g2, g3, g4, g5)
        self.file_prefixes = config.get('file_prefixes', ['g1', 'g2', 'g3', 'g4', 'g5'])
        
        # 프로세서 초기화
        self.processor = SensorDataProcessor(
            interpolation_step=config.get('interp_step', 0.001),
            window_size=self.window_size
        )
        
        # 모델 초기화
        self.model = None
        
        # 분류 결과 매핑
        self.class_names = config.get('class_names', ['normal', 'type1', 'type2', 'type3'])
        
        # 마지막 처리 시간 기록
        self.last_processed_times = {prefix: None for prefix in self.file_prefixes}
        
        logger.info("센서 데이터 분류 및 업로드 클래스 초기화 완료")

    def load_model(self) -> bool:
        """
        모델 로드
        
        Returns:
            bool: 로드 성공 여부
        """
        try:
            logger.info(f"모델 로드 중: {self.model_path}")
            
            # 모델 정보 로드 (있는 경우)
            if self.model_info_path.exists():
                with open(self.model_info_path, 'r') as f:
                    model_info = json.load(f)
                    self.input_size = model_info.get('input_size', self.input_size)
                    self.hidden_size = model_info.get('hidden_size', self.hidden_size)
                    self.num_layers = model_info.get('num_layers', self.num_layers)
                    self.num_classes = model_info.get('num_classes', self.num_classes)
                    self.sequence_length = model_info.get('sequence_length', self.sequence_length)
                    logger.info(f"모델 정보 로드: input_size={self.input_size}, hidden_size={self.hidden_size}, sequence_length={self.sequence_length}")
            
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
    
    def process_sensor_data(self, file_prefix: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        센서 데이터 처리
        
        Args:
            file_prefix: 처리할 센서 파일 접두사 (g1, g2 등)
            
        Returns:
            Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]: (학습, 검증, 테스트) 데이터 또는 실패 시 None
        """
        try:
            # 센서 파일 로드 및 보간
            logger.info(f"{file_prefix} 센서 데이터 로드 및 보간 중...")
            interpolated_data = self.processor.load_and_interpolate_sensor_data(
                self.data_dir, prefix=file_prefix
            )
            
            if not interpolated_data:
                logger.error(f"{file_prefix} 센서 데이터를 로드할 수 없습니다.")
                return None
            
            # 데이터 결합 및 전처리
            logger.info("센서 데이터 결합 및 전처리 중...")
            processed_data = self.processor.combine_and_preprocess_sensor_data(interpolated_data)
            
            # 데이터 분할
            train_data, valid_data, test_data = self.processor.split_and_combine_data(
                processed_data,
                train_ratio=0.6,
                valid_ratio=0.2,
                test_ratio=0.2
            )
            
            logger.info(f"데이터 처리 완료: 학습={train_data.shape}, 검증={valid_data.shape}, 테스트={test_data.shape}")
            return train_data, valid_data, test_data
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return None
    
    def classify_sensor_data(self, sensor_data: np.ndarray) -> List[Dict[str, Any]]:
        """
        센서 데이터 분류 및 결과 생성
        
        Args:
            sensor_data: 센서 데이터 배열
            
        Returns:
            List[Dict[str, Any]]: 분류 결과 목록
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다.")
            return []
        
        try:
            # 시퀀스 데이터 준비
            logger.info("시퀀스 데이터 준비 중...")
            X_data, y_data = prepare_sequence_data(sensor_data, sequence_length=self.sequence_length)
            
            logger.info(f"시퀀스 데이터 형태: {X_data.shape}")
            
            # 분류 결과 목록
            classification_results = []
            
            # 배치 단위로 처리
            batch_size = 32  # 모델 추론 배치 크기
            num_samples = X_data.shape[0]
            
            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch_data = X_data[i:end_idx]
                
                # 텐서 변환
                batch_tensor = torch.tensor(batch_data, dtype=torch.float32).to(self.device)
                
                # 예측 수행
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # 결과 변환
                for j in range(batch_data.shape[0]):
                    data_idx = i + j
                    pred_class = predicted[j].item()
                    conf_value = confidence[j].item()
                    
                    # 실제 레이블이 있으면 함께 저장
                    actual_class = y_data[data_idx] if y_data is not None else None
                    
                    # 클래스명 매핑
                    predicted_label = self.class_names[pred_class] if pred_class < len(self.class_names) else f"unknown_{pred_class}"
                    
                    # 결과 저장
                    classification_results.append({
                        "sequence_data": batch_data[j].tolist(),  # 시퀀스 데이터 저장
                        "predicted_class": int(pred_class),  # 예측 클래스 (정수)
                        "predicted_label": predicted_label,  # 예측 클래스명
                        "confidence": float(conf_value),  # 신뢰도
                        "actual_class": int(actual_class) if actual_class is not None else None,  # 실제 클래스 (정수, 있는 경우)
                        "timestamp": datetime.now().isoformat()  # 타임스탬프
                    })
            
            logger.info(f"분류 완료: {len(classification_results)}개 결과 생성")
            return classification_results
            
        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return []
    
    def format_for_api(self, machine_name: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        API 업로드용으로 결과 형식 변환
        
        Args:
            machine_name: 기계 이름 (g1, g2 등)
            results: 분류 결과 목록
            
        Returns:
            List[Dict[str, Any]]: API 형식 데이터
        """
        api_data = []
        
        for result in results:
            # fault_type 변환: 0=normal, 1,2,3=고장 유형
            fault_type = 0 if result["predicted_class"] == 0 else result["predicted_class"]
            
            # 원시 시퀀스 데이터를 JSON 문자열로 변환
            sequence_data_json = json.dumps({
                "sequence": result["sequence_data"],
                "confidence": result["confidence"],
                "actual_class": result["actual_class"]
            })
            
            # API 형식으로 변환
            api_data.append({
                "machine_name": machine_name,
                "detected_at": result["timestamp"],
                "fault_type": fault_type,
                "sensor_data": sequence_data_json  # 센서 데이터를 추가 필드로 전송
            })
        
        return api_data
    
    def upload_to_api(self, api_data: List[Dict[str, Any]]) -> int:
        """
        분류 결과를 API 서버로 업로드
        
        Args:
            api_data: 업로드할 API 형식 데이터
            
        Returns:
            int: 업로드된 레코드 수
        """
        if not api_data:
            logger.warning("업로드할 데이터가 없습니다.")
            return 0
        
        total_uploaded = 0
        
        try:
            # 배치 단위로 처리
            for i in range(0, len(api_data), self.batch_size):
                batch = api_data[i:i+self.batch_size]
                
                # API 요청
                response = requests.post(API_URL_INSERT, json=batch)
                
                if response.status_code == 200:
                    total_uploaded += len(batch)
                    logger.info(f"✅ 누적 {total_uploaded}개 업로드 완료: {response.json()}")
                else:
                    logger.error(f"❌ 업로드 실패 ({i} ~ {i+len(batch)}): {response.status_code} {response.text}")
                    break
            
            return total_uploaded
            
        except Exception as e:
            logger.error(f"API 업로드 중 오류 발생: {str(e)}")
            return total_uploaded
    
    def check_total_count(self) -> int:
        """
        API 서버 내 전체 데이터 수 확인
        
        Returns:
            int: 전체 데이터 수
        """
        try:
            response = requests.get(API_URL_COUNT)
            response.raise_for_status()
            total = len(response.json())
            logger.info(f"📊 현재 진단 데이터 총 {total}건 존재합니다.")
            return total
        except Exception as e:
            logger.error(f"❌ 전체 데이터 수 조회 실패: {str(e)}")
            return -1

    def process_single_cycle(self) -> int:
        """
        단일 주기 처리 (모든 센서에 대해 한 번 처리)
        
        Returns:
            int: 업로드된 총 레코드 수
        """
        # 현재 API 서버 데이터 수 확인
        initial_count = self.check_total_count()
        
        total_results = 0
        
        # 각 센서 파일 처리
        for file_prefix in self.file_prefixes:
            logger.info(f"\n=== {file_prefix} 센서 데이터 처리 시작 ===")
            
            # 센서 데이터 처리
            result = self.process_sensor_data(file_prefix)
            if result is None:
                logger.warning(f"{file_prefix} 센서 데이터 처리를 건너뜁니다.")
                continue
            
            train_data, valid_data, test_data = result
            
            # 학습 데이터만 샘플링하여 사용 (전체 데이터를 다 올리면 너무 많을 수 있음)
            # 추가: 매번 다른 샘플을 사용하기 위해 랜덤 시드 변경
            np.random.seed(int(time.time()))
            
            if train_data.shape[0] > 1000:
                sample_indices = np.random.choice(train_data.shape[0], 1000, replace=False)
                sample_data = train_data[sample_indices]
                logger.info(f"{file_prefix} 학습 데이터에서 1000개 샘플 추출: {sample_data.shape}")
            else:
                sample_data = train_data
            
            # 센서 데이터 분류
            classification_results = self.classify_sensor_data(sample_data)
            if not classification_results:
                logger.warning(f"{file_prefix} 분류 결과가 없습니다.")
                continue
            
            # API 전송 형식으로 변환
            api_data = self.format_for_api(file_prefix, classification_results)
            
            # API 서버로 업로드
            uploaded = self.upload_to_api(api_data)
            total_results += uploaded
            
            # 마지막 처리 시간 업데이트
            self.last_processed_times[file_prefix] = datetime.now()
            logger.info(f"{file_prefix} 처리 완료: {uploaded}개 결과 업로드")
        
        # 최종 결과 출력
        final_count = self.check_total_count()
        
        logger.info("\n=== 처리 주기 완료 ===")
        logger.info(f"- 초기 데이터 수: {initial_count}")
        logger.info(f"- 업로드된 결과 수: {total_results}")
        logger.info(f"- 최종 데이터 수: {final_count}")
        
        if final_count >= 0:
            difference = final_count - initial_count
            if difference != total_results:
                logger.warning(f"업로드된 결과 수 ({total_results})와 실제 증가한 데이터 수 ({difference})가 일치하지 않습니다.")
                
        return total_results
    
    def run_periodic(self, max_cycles: int = -1) -> None:
        """
        주기적으로 실행
        
        Args:
            max_cycles: 최대 실행 주기 수 (-1은 무한 반복)
        """
        # 모델 로드
        if not self.load_model():
            logger.error("모델 로드 실패로 종료합니다.")
            return
        
        logger.info(f"주기적 실행 시작: 간격 {self.interval_minutes}분, 최대 주기 {max_cycles if max_cycles > 0 else '무한'}")
        
        try:
            cycle_count = 0
            
            while max_cycles < 0 or cycle_count < max_cycles:
                cycle_start_time = time.time()
                cycle_count += 1
                
                logger.info(f"\n===== 주기 {cycle_count} 시작 =====")
                logger.info(f"현재 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # 단일 주기 처리
                total_uploaded = self.process_single_cycle()
                
                # 진행 상황 출력
                logger.info(f"주기 {cycle_count} 완료: {total_uploaded}개 레코드 업로드됨")
                
                # 다음 주기까지 대기 (처리 시간 고려)
                cycle_end_time = time.time()
                cycle_duration = cycle_end_time - cycle_start_time
                
                wait_time = (self.interval_minutes * 60) - cycle_duration
                if wait_time > 0:
                    logger.info(f"다음 주기까지 {wait_time:.1f}초 대기 중...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"주기 처리에 {cycle_duration:.1f}초 소요, 간격({self.interval_minutes * 60}초)보다 길어 즉시 다음 주기 시작")
                
        except KeyboardInterrupt:
            logger.info("\n사용자에 의해 중단되었습니다.")
        except Exception as e:
            logger.error(f"주기적 실행 중 오류 발생: {str(e)}")
        finally:
            logger.info("주기적 실행 종료")


def main():
    """메인 함수"""
    # 명령줄 인자 파싱
    parser = argparse.ArgumentParser(description='주기적 센서 데이터 분류 및 API 서버 업로드')
    
    parser.add_argument('--data_dir', type=str, default='data/raw',
                      help='센서 데이터 디렉토리')
    parser.add_argument('--model_path', type=str, default='models/sensor_classifier.pth',
                      help='모델 파일 경로')
    parser.add_argument('--model_info_path', type=str, default='models/model_info.json',
                      help='모델 정보 파일 경로')
    parser.add_argument('--sequence_length', type=int, default=50,
                      help='시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='API 업로드 배치 크기')
    parser.add_argument('--file_prefixes', type=str, nargs='+', default=['g1', 'g2', 'g3', 'g4', 'g5'],
                      help='처리할 센서 파일 접두사 목록')
    parser.add_argument('--interval_minutes', type=int, default=30,
                      help='처리 주기 (분 단위)')
    parser.add_argument('--max_cycles', type=int, default=-1,
                      help='최대 실행 주기 수 (-1은 무한 반복)')
    
    args = parser.parse_args()
    
    # 설정 딕셔너리 생성
    config = vars(args)
    
    # 업로더 생성 및 주기적 실행
    uploader = SensorDataUploader(config)
    uploader.run_periodic(max_cycles=args.max_cycles)


if __name__ == "__main__":
    main()