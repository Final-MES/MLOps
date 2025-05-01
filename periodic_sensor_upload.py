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
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple

# 프로젝트 루트 경로 추가
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 필요한 모듈 임포트
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
        
        # 주기 설정 (초 단위)
        self.interval_minutes = config.get('interval_minutes', 60) 
        
        # 여러 센서 파일 접두사 지정 (g1, g2, g3, g4, g5)
        self.file_prefixes = config.get('file_prefixes', ['g1', 'g2'])
        
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
    
    def process_sensor_data(self, file_prefix: str) -> Optional[np.ndarray]:
        """
        센서 데이터 처리
        
        Args:
            file_prefix: 처리할 센서 파일 접두사 (g1, g2 등)
            
        Returns:
            Optional[np.ndarray]: 처리된 데이터 또는 실패 시 None
        """
        try:
            # 센서 파일 경로 구성 (4개 센서)
            sensor_files = [
                f"{file_prefix}_sensor1_blocks.csv",
                f"{file_prefix}_sensor2_blocks.csv",
                f"{file_prefix}_sensor3_blocks.csv",
                f"{file_prefix}_sensor4_blocks.csv"
            ]
            
            # 각 센서 데이터 로드 및 처리
            sensor_data = {}
            
            for i, sensor_file in enumerate(sensor_files, start=1):
                file_path = os.path.join(self.data_dir, sensor_file)
                
                if not os.path.exists(file_path):
                    logger.warning(f"센서 {i}의 파일이 없습니다: {file_path}")
                    continue
                
                # 데이터 로드
                try:
                    # 'time'과 'value' 컬럼을 가진 CSV 파일 로드
                    df = pd.read_csv(file_path)
                    
                    # 필요한 컬럼 확인
                    if 'time' not in df.columns or 'value' not in df.columns:
                        # 컬럼 이름이 다르면 기본 이름으로 가정
                        if len(df.columns) >= 2:
                            df.columns = ['time', 'value']
                        else:
                            logger.error(f"{file_path} 파일에 필요한 컬럼이 없습니다.")
                            continue
                    
                    # 시간순으로 정렬
                    df = df.sort_values(by='time')
                    
                    # 데이터 정규화 (필요한 경우)
                    # MinMax 스케일링 적용 (-1 ~ 1 범위)
                    value_col = df['value'].values
                    min_val = value_col.min()
                    max_val = value_col.max()
                    
                    if max_val > min_val:
                        normalized_data = -1 + 2 * (value_col - min_val) / (max_val - min_val)
                    else:
                        normalized_data = np.zeros_like(value_col)
                    
                    # 처리된 데이터 저장
                    sensor_data[f'sensor{i}'] = normalized_data
                    logger.info(f"센서 {i} 데이터 로드 및 정규화 완료: {len(normalized_data)} 샘플")
                    
                except Exception as e:
                    logger.error(f"센서 {i} 데이터 처리 중 오류 발생: {str(e)}")
                    continue
            
            # 모든 센서가 없는 경우
            if not sensor_data:
                logger.error(f"처리할 센서 데이터가 없습니다: {file_prefix}")
                return None
            
            # 모든 센서 데이터의 길이를 맞춤
            min_length = min(len(data) for data in sensor_data.values())
            aligned_data = {}
            
            for sensor_name, data in sensor_data.items():
                aligned_data[sensor_name] = data[:min_length]
            
            # 센서 데이터 결합 (열방향)
            combined_data = np.column_stack([aligned_data[f'sensor{i}'] for i in range(1, 5) if f'sensor{i}' in aligned_data])
            
            # 없는 센서가 있는 경우 빈 열 추가
            missing_sensors = 4 - combined_data.shape[1]
            if missing_sensors > 0:
                padding = np.zeros((combined_data.shape[0], missing_sensors))
                combined_data = np.hstack((combined_data, padding))
                logger.warning(f"{missing_sensors}개 센서 데이터가 없어 0으로 채웠습니다.")
            
            logger.info(f"데이터 처리 완료: 형태={combined_data.shape}")
            return combined_data
            
        except Exception as e:
            logger.error(f"데이터 처리 중 오류 발생: {str(e)}")
            return None
        
    def classify_sensor_data(self, sensor_data: np.ndarray, interval: float = 1.0) -> List[Dict[str, Any]]:
        """
        센서 데이터 분류 및 결과 생성 - 1초마다 슬라이딩 윈도우 사용
        
        Args:
            sensor_data: 센서 데이터 배열
            interval: 분류 간격 (초 단위), 기본값 1.0초
            
        Returns:
            List[Dict[str, Any]]: API 형식 분류 결과 목록
        """
        if self.model is None:
            logger.error("모델이 로드되지 않았습니다.")
            return []
        
        window_size = self.sequence_length
        
        if len(sensor_data) < window_size:
            logger.error(f"데이터 길이({len(sensor_data)})가 윈도우 크기({window_size})보다 작습니다.")
            return []
        
        # API 형식 결과 목록 (바로 API 형식으로 생성)
        api_results = []
        
        try:
            # 윈도우 수 계산 (스텝 크기 1로 고정)
            step = 200
            num_windows = (len(sensor_data) - window_size) // step + 1
            logger.info(f"처리할 윈도우 수: {num_windows} (간격: {interval}초)")
            
            for i in range(0, num_windows):
                start_time = time.time()
                
                # 현재 윈도우 추출
                start_idx = i * step
                end_idx = start_idx + window_size
                window = sensor_data[start_idx:end_idx]
                
                # 모델 입력 형태로 변환
                model_input = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # 예측 수행
                with torch.no_grad():
                    outputs = self.model(model_input)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                # 결과 추출
                pred_class = predicted.item()
                conf_value = confidence.item()
                
                # 클래스명 매핑
                predicted_label = self.class_names[pred_class] if pred_class < len(self.class_names) else f"unknown_{pred_class}"
                
                # API 형식으로 바로 결과 생성
                api_result = {
                    "predicted_class": int(pred_class),
                    "predicted_label": predicted_label,
                    "confidence": float(conf_value),
                    "timestamp": datetime.now().isoformat()
                }
                api_results.append(api_result)
                
                # 다음 윈도우 처리 전에 일정 시간 대기
                elapsed = time.time() - start_time
                if elapsed < interval:
                    time.sleep(interval - elapsed)
                else:
                    logger.warning(f"처리 시간이 간격보다 깁니다: {elapsed:.4f}초 > {interval}초")
            
            logger.info(f"분류 완료: 총 {len(api_results)}개 윈도우 처리됨")
            return api_results
        
        except Exception as e:
            logger.error(f"분류 중 오류 발생: {str(e)}")
            return []

    def format_for_api(self, machine_name: str, classification_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        API 업로드용으로 결과 형식 변환
        
        Args:
            machine_name: 기계 이름 (g1, g2 등)
            classification_results: 분류 결과 목록
            
        Returns:
            List[Dict[str, Any]]: API 형식 데이터
        """
        api_data = []
        
        for result in classification_results:
            # fault_type 변환: 0=normal, 1,2,3=고장 유형
            fault_type = 0 if result["predicted_class"] == 0 else result["predicted_class"]
            
            # API 형식으로 변환
            api_data.append({
                "machine_name": machine_name,
                "detected_at": result["timestamp"],
                "fault_type": fault_type
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
            processed_data = self.process_sensor_data(file_prefix)
            if processed_data is None:
                logger.warning(f"{file_prefix} 센서 데이터 처리를 건너뜁니다.")
                continue
            
            # 데이터 샘플링 (전체 데이터가 너무 많은 경우)
            if len(processed_data) > 10000:  # 최대 10,000 샘플 처리
                # 랜덤 시드 설정 (매번 다른 샘플을 위해)
                np.random.seed(int(time.time()))
                sample_indices = np.random.choice(len(processed_data), 10000, replace=False)
                sample_data = processed_data[sample_indices]
                logger.info(f"{file_prefix} 데이터에서 10000개 샘플 추출: {sample_data.shape}")
            else:
                sample_data = processed_data
            
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
        
        logger.info(f"주기적 실행 시작: 간격 {self.interval_minutes}초, 최대 주기 {max_cycles if max_cycles > 0 else '무한'}")
        
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
                
                wait_time = (self.interval_minutes) - cycle_duration
                if wait_time > 0:
                    logger.info(f"다음 주기까지 {wait_time:.1f}초 대기 중...")
                    time.sleep(wait_time)
                else:
                    logger.warning(f"주기 처리에 {cycle_duration:.1f}초 소요, 간격({wait_time}초)보다 길어 즉시 다음 주기 시작")
                
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
    
    parser.add_argument('--data_dir', type=str, default='data/blocks',
                      help='센서 데이터 디렉토리')
    parser.add_argument('--model_path', type=str, default='models/sensor_classifier.pth',
                      help='모델 파일 경로')
    parser.add_argument('--model_info_path', type=str, default='models/model_info.json',
                      help='모델 정보 파일 경로')
    parser.add_argument('--sequence_length', type=int, default=50,
                      help='시퀀스 길이')
    parser.add_argument('--batch_size', type=int, default=1000,
                      help='API 업로드 배치 크기')
    parser.add_argument('--file_prefixes', type=str, nargs='+', default=['g1', 'g2'],
                      help='처리할 센서 파일 접두사 목록')
    parser.add_argument('--interval_minutes', type=int, default=60,
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