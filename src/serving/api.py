from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import os
import json
import logging
from datetime import datetime
import time
from prometheus_client import Counter, Histogram, Gauge
import prometheus_client

from src.models.sensor.lstm_model import LSTMModel

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'model_api.log'))
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수에서 설정 가져오기
MODEL_DIR = os.getenv('MODEL_DIR', 'models')
MODEL_NAME = os.getenv('MODEL_NAME', 'lstm_model')
MODEL_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}.pth")
MODEL_INFO_PATH = os.path.join(MODEL_DIR, f"{MODEL_NAME}_info.json")
X_SCALER_PATH = os.path.join(MODEL_DIR, "X_scaler.npy")
Y_SCALER_PATH = os.path.join(MODEL_DIR, "y_scaler.npy")
METRICS_PORT = int(os.getenv('METRICS_PORT', '8001'))

# Prometheus 메트릭 정의
REQUEST_COUNT = Counter('model_prediction_requests_total', 'Total number of prediction requests')
LATENCY = Histogram('model_prediction_latency_ms', 'Prediction latency in milliseconds')
PREDICTION_ERROR = Gauge('model_prediction_error', 'Prediction error (RMSE)')
PREDICTION_ACCURACY = Gauge('model_prediction_accuracy', 'Prediction accuracy')
ANOMALY_SCORE = Gauge('anomaly_score', 'Anomaly score by equipment', ['equipment_id'])
ANOMALY_COUNT = Counter('anomaly_count', 'Anomaly detection count', ['equipment_id'])
MODEL_DRIFT_DETECTED = Gauge('model_drift_detected', 'Data drift detected (1 = yes, 0 = no)')

# 메트릭 서버 시작
prometheus_client.start_http_server(METRICS_PORT)
logger.info(f"Prometheus 메트릭 서버 시작됨: 포트 {METRICS_PORT}")

# 입력 요청 모델
class PredictionRequest(BaseModel):
    sensor_data: List[Dict[str, float]] = Field(..., description="센서 데이터 시퀀스")
    equipment_id: Optional[str] = Field(None, description="장비 ID")

# 응답 모델
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="예측 값")
    confidence: float = Field(..., description="예측 신뢰도")
    timestamp: str = Field(..., description="예측 시간")
    model_version: str = Field(..., description="모델 버전")

# 이상 감지 요청 모델
class AnomalyDetectionRequest(BaseModel):
    sensor_data: List[Dict[str, float]] = Field(..., description="센서 데이터 시퀀스")
    equipment_id: str = Field(..., description="장비 ID")
    threshold: Optional[float] = Field(0.1, description="이상 감지 임계값")

# 이상 감지 응답 모델
class AnomalyDetectionResponse(BaseModel):
    is_anomaly: bool = Field(..., description="이상 감지 여부")
    anomaly_score: float = Field(..., description="이상 점수")
    details: Dict[str, Any] = Field(..., description="상세 정보")
    timestamp: str = Field(..., description="감지 시간")

# 모델 정보 응답 모델
class ModelInfoResponse(BaseModel):
    model_name: str
    input_size: int
    sequence_length: int
    feature_columns: List[str]
    target_column: str
    metrics: Dict[str, float]
    last_updated: str

# 모델 로드 함수
def load_model():
    """모델 및 관련 파일 로드"""
    try:
        # 모델 정보 로드
        with open(MODEL_INFO_PATH, 'r') as f:
            model_info = json.load(f)
        
        # 모델 초기화
        model = LSTMModel(
            input_size=model_info['input_size'],
            hidden_size=model_info['hidden_size'],
            num_layers=model_info['num_layers'],
            output_size=model_info['output_size']
        )
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        
        # 스케일러 로드
        X_scaler = np.load(X_SCALER_PATH, allow_pickle=True).item()
        y_scaler = np.load(Y_SCALER_PATH, allow_pickle=True).item()
        
        logger.info(f"모델 로드 성공: {MODEL_PATH}")
        
        return {
            'model': model,
            'model_info': model_info,
            'X_scaler': X_scaler,
            'y_scaler': y_scaler
        }
    except Exception as e:
        logger.error(f"모델 로드 실패: {str(e)}")
        raise HTTPException(status_code=500, detail=f"모델 로드 실패: {str(e)}")

# FastAPI 앱 초기화
app = FastAPI(
    title="Smart Factory LSTM Model API",
    description="스마트 팩토리 센서 데이터 분석을 위한 LSTM 모델 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델 데이터 의존성
def get_model_data():
    """모델 데이터 의존성 주입"""
    if not hasattr(app, 'model_data'):
        app.model_data = load_model()
    return app.model_data

# 요청 타이밍 미들웨어
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """요청 처리 시간 측정 미들웨어"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 헬스체크 엔드포인트
@app.get("/health")
def health_check():
    """API 헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 모델 정보 엔드포인트
@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(model_data: dict = Depends(get_model_data)):
    """모델 정보 반환"""
    model_info = model_data['model_info']
    
    return {
        "model_name": model_info.get('model_type', 'LSTM') + " Sensor Prediction Model",
        "input_size": model_info['input_size'],
        "sequence_length": model_info['sequence_length'],
        "feature_columns": model_info['feature_cols'],
        "target_column": model_info['target_col'],
        "metrics": {
            "best_val_loss": model_info.get('best_val_loss', 0),
            "rmse": model_info.get('rmse', 0)
        },
        "last_updated": datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).isoformat()
    }

# 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, model_data: dict = Depends(get_model_data)):
    """시계열 데이터 예측"""
    REQUEST_COUNT.inc()  # 요청 카운터 증가
    start_time = time.time()
    
    try:
        # 모델 및 스케일러 가져오기
        model = model_data['model']
        model_info = model_data['model_info']
        X_scaler = model_data['X_scaler']
        y_scaler = model_data['y_scaler']
        
        # 입력 데이터 변환
        feature_cols = model_info['feature_cols']
        sequence_length = model_info['sequence_length']
        
        # 입력 데이터 확인
        if len(request.sensor_data) < sequence_length:
            raise HTTPException(
                status_code=400, 
                detail=f"입력 시퀀스 길이가 부족합니다. 최소 {sequence_length}개의 데이터 포인트가 필요합니다."
            )
        
        # 시퀀스 데이터 추출 (최신 sequence_length 개수만큼)
        sensor_sequence = request.sensor_data[-sequence_length:]
        
        # 입력 특성 추출 및 정규화
        input_data = []
        for data_point in sensor_sequence:
            # 모든 특성을 올바른 순서로 추출
            features = []
            for feature in feature_cols:
                if feature in data_point:
                    features.append(data_point[feature])
                else:
                    # 특성이 없는 경우 0으로 대체 (또는 다른 대체 전략 사용)
                    features.append(0.0)
                    logger.warning(f"특성 '{feature}'이(가) 입력 데이터에 없습니다. 0으로 대체합니다.")
            
            input_data.append(features)
        
        # NumPy 배열로 변환
        input_array = np.array(input_data, dtype=np.float32)
        
        # 입력 데이터 정규화
        normalized_input = X_scaler.transform(input_array)
        
        # 텐서로 변환 및 차원 추가 (배치 차원)
        input_tensor = torch.tensor(normalized_input, dtype=torch.float32).unsqueeze(0)
        
        # 예측 수행
        with torch.no_grad():
            output = model(input_tensor)
            
        # 예측 결과 역정규화
        prediction = y_scaler.inverse_transform(output.numpy())[0][0]
        
        # 신뢰도 계산 (여기서는 간단히 예측 절대값 기준으로 계산)
        # 실제 애플리케이션에서는 예측 불확실성을 더 정교하게 모델링할 수 있음
        confidence = min(max(1.0 - abs(prediction) * 0.01, 0.5), 0.99)
        
        # 응답 생성
        response = {
            "prediction": float(prediction),
            "confidence": float(confidence),
            "timestamp": datetime.now().isoformat(),
            "model_version": os.path.basename(MODEL_PATH)
        }
        
        # 지연 시간 측정
        latency = (time.time() - start_time) * 1000  # 밀리초 단위
        LATENCY.observe(latency)
        
        # 예측 로깅
        logger.info(f"예측 성공: {response}")
        
        return response
        
    except Exception as e:
        logger.error(f"예측 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

# 이상 감지 엔드포인트
@app.post("/detect-anomaly", response_model=AnomalyDetectionResponse)
def detect_anomaly(request: AnomalyDetectionRequest, model_data: dict = Depends(get_model_data)):
    """실시간 이상 감지"""
    try:
        # 모델 기반 예측 수행 (predict 함수 호출과 유사)
        prediction_request = PredictionRequest(
            sensor_data=request.sensor_data,
            equipment_id=request.equipment_id
        )
        
        prediction_response = predict(prediction_request, model_data)
        actual_value = request.sensor_data[-1].get(model_data['model_info']['target_col'], None)
        
        if actual_value is None:
            # 실제 값이 없는 경우 이전 값들의 평균으로 대체
            previous_values = []
            for data_point in request.sensor_data[:-1]:
                if model_data['model_info']['target_col'] in data_point:
                    previous_values.append(data_point[model_data['model_info']['target_col']])
            
            if previous_values:
                actual_value = sum(previous_values) / len(previous_values)
            else:
                actual_value = prediction_response.prediction  # 대체할 값이 없으면 예측값 사용
        
        # 예측 오차 계산
        prediction_error = abs(prediction_response.prediction - actual_value)
        anomaly_score = prediction_error
        
        # 임계값과 비교하여 이상 감지
        is_anomaly = anomaly_score > request.threshold
        
        if is_anomaly:
            # 이상 감지 카운터 증가
            ANOMALY_COUNT.labels(equipment_id=request.equipment_id).inc()
        
        # 이상 점수 게이지 업데이트
        ANOMALY_SCORE.labels(equipment_id=request.equipment_id).set(anomaly_score)
        
        # 응답 생성
        response = {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "details": {
                "prediction": prediction_response.prediction,
                "actual_value": actual_value,
                "threshold": request.threshold,
                "prediction_error": float(prediction_error),
                "equipment_id": request.equipment_id
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"이상 감지 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"이상 감지 중 오류 발생: {str(e)}")

# 데이터 드리프트 감지 엔드포인트
@app.get("/drift-detection")
def detect_drift(model_data: dict = Depends(get_model_data)):
    """데이터 드리프트 감지 상태 반환"""
    # 여기서는 간단한 예시로 랜덤 값 반환
    # 실제 구현에서는 기준 데이터 분포와 현재 데이터 분포 비교
    from random import random
    
    drift_score = random() * 0.05  # 0~0.05 사이 랜덤 값
    drift_detected = drift_score > 0.04  # 임계값 0.04
    
    if drift_detected:
        MODEL_DRIFT_DETECTED.set(1)
    else:
        MODEL_DRIFT_DETECTED.set(0)
    
    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "features_drifted": [] if not drift_detected else ["feature_1", "feature_3"],
        "timestamp": datetime.now().isoformat()
    }

# 모델 리로드 엔드포인트 (관리자용)
@app.post("/reload-model")
def reload_model(background_tasks: BackgroundTasks):
    """모델 리로드 (백그라운드 작업으로 실행)"""
    def _reload():
        if hasattr(app, 'model_data'):
            delattr(app, 'model_data')
        try:
            app.model_data = load_model()
            logger.info("모델 리로드 성공")
        except Exception as e:
            logger.error(f"모델 리로드 실패: {str(e)}")
    
    background_tasks.add_task(_reload)
    return {"message": "모델 리로드 요청이 수락되었습니다."}

# 메인 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("src.serving.api:app", host="0.0.0.0", port=port, reload=False)