from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import torch
import numpy as np
import os
import json
import logging
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/model_api.log')
    ]
)
logger = logging.getLogger(__name__)

# 환경 변수에서 모델 경로 가져오기
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/best_lstm_model.pth')
MODEL_INFO_PATH = os.getenv('MODEL_INFO_PATH', '/app/models/model_info.json')
X_SCALER_PATH = os.getenv('X_SCALER_PATH', '/app/models/X_scaler.npy')
Y_SCALER_PATH = os.getenv('Y_SCALER_PATH', '/app/models/y_scaler.npy')

# LSTM 모델 클래스
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = torch.nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 완전 연결 레이어
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # LSTM 출력 (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시퀀스의 출력만 사용
        out = self.fc(lstm_out[:, -1, :])
        return out

# 모델 로드 함수
def load_model():
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

# 모델 의존성
def get_model():
    if not hasattr(app, 'model_data'):
        app.model_data = load_model()
    return app.model_data

# 예측 요청 모델
class PredictionRequest(BaseModel):
    sensor_data: List[Dict[str, float]] = Field(..., description="센서 데이터 시퀀스")
    equipment_id: Optional[str] = Field(None, description="장비 ID")

# 예측 응답 모델
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="예측 값")
    confidence: float = Field(..., description="예측 신뢰도")
    timestamp: str = Field(..., description="예측 시간")
    model_version: str = Field(..., description="모델 버전")

# 모델 정보 응답 모델
class ModelInfoResponse(BaseModel):
    model_name: str
    input_size: int
    sequence_length: int
    feature_columns: List[str]
    target_column: str
    metrics: Dict[str, float]
    last_updated: str

# 헬스체크 엔드포인트
@app.get("/health")
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# 모델 정보 엔드포인트
@app.get("/model-info", response_model=ModelInfoResponse)
def get_model_info(model_data: dict = Depends(get_model)):
    model_info = model_data['model_info']
    
    return {
        "model_name": "LSTM Sensor Prediction Model",
        "input_size": model_info['input_size'],
        "sequence_length": model_info['sequence_length'],
        "feature_columns": model_info['feature_cols'],
        "target_column": model_info['target_col'],
        "metrics": {
            "test_loss": model_info['test_loss'],
            "rmse": model_info['rmse']
        },
        "last_updated": datetime.fromtimestamp(
            os.path.getmtime(MODEL_PATH)
        ).isoformat()
    }

# 예측 엔드포인트
@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest, model_data: dict = Depends(get_model)):
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
        
        # 예측 로깅
        logger.info(f"예측 성공: {response}")
        
        return response
        
    except Exception as e:
        logger.error(f"예측 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"예측 중 오류 발생: {str(e)}")

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

# 이상 감지 엔드포인트
@app.post("/detect-anomaly", response_model=AnomalyDetectionResponse)
def detect_anomaly(request: AnomalyDetectionRequest, model_data: dict = Depends(get_model)):
    try:
        # 모델 및 스케일러 가져오기
        model = model_data['model']
        model_info = model_data['model_info']
        X_scaler = model_data['X_scaler']
        y_scaler = model_data['y_scaler']
        
        # 입력 데이터 가공 (predict 함수와 유사)
        # ...예측 로직과 유사...
        
        # 마지막 실제값과 예측값의 차이 계산
        # 여기서는 간소화를 위해 가상의 값을 생성
        prediction_error = np.random.normal(0, 0.05)
        anomaly_score = abs(prediction_error)
        
        # 임계값과 비교하여 이상 감지
        is_anomaly = anomaly_score > request.threshold
        
        # 응답 생성
        response = {
            "is_anomaly": is_anomaly,
            "anomaly_score": float(anomaly_score),
            "details": {
                "prediction_error": float(prediction_error),
                "threshold": request.threshold,
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
def detect_drift(model_data: dict = Depends(get_model)):
    # 데이터 드리프트 감지 로직
    # 실제 구현에서는 기준 데이터 분포와 현재 데이터 분포 비교
    return {
        "drift_detected": False,
        "drift_score": 0.02,
        "features_drifted": [],
        "timestamp": datetime.now().isoformat()
    }

# 메인 실행
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)