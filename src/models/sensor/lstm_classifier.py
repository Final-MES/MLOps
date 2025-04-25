"""
다중 센서 데이터 분류를 위한 LSTM 모델 모듈

이 모듈은 다중 센서 시계열 데이터 분류를 위한 LSTM 기반 신경망 모델을 정의합니다.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

# 로깅 설정
logger = logging.getLogger(__name__)

class MultiSensorLSTMClassifier(nn.Module):
    """
    다중 센서 시계열 데이터 분류를 위한 LSTM 모델
    
    특징:
    - 다층 LSTM을 사용한 시퀀스 모델링
    - 어텐션 메커니즘을 사용하여 중요한 시간 스텝에 가중치 부여
    - 다중 클래스 분류를 위한 완전 연결 레이어
    """
    
    def __init__(self, 
                input_size: int, 
                hidden_size: int, 
                num_layers: int, 
                num_classes: int, 
                dropout_rate: float = 0.3):
        """
        다중 센서 LSTM 분류기 초기화
        
        Args:
            input_size: 입력 특성 수 (센서 수)
            hidden_size: LSTM 은닉층 크기
            num_layers: LSTM 레이어 수
            num_classes: 출력 클래스 수 (상태 수)
            dropout_rate: 드롭아웃 비율
        """
        super(MultiSensorLSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 어텐션 메커니즘
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # 모델 초기화
        self._initialize_weights()
        
        logger.info(f"MultiSensorLSTMClassifier 초기화: input_size={input_size}, "
                    f"hidden_size={hidden_size}, num_layers={num_layers}, "
                    f"num_classes={num_classes}")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x: 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 클래스별 확률, 형태 (batch_size, num_classes)
        """
        # 입력 형태 검증
        batch_size, seq_len, features = x.size()
        
        # LSTM 출력 (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 어텐션 가중치 계산
        attn_weights = self.attention(lstm_out)  # (batch_size, seq_length, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)  # 시퀀스 차원을 따라 소프트맥스
        
        # 어텐션 가중치를 사용하여 컨텍스트 벡터 계산
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch_size, hidden_size)
        
        # 최종 분류 결과
        out = self.fc(context)  # (batch_size, num_classes)
        
        return out
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        주어진 입력 데이터에 대한 예측 수행
        
        Args:
            x: 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            tuple: (예측 클래스, 클래스별 확률)
        """
        self.eval()  # 평가 모드로 설정
        with torch.no_grad():
            outputs = self(x)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        return predicted, probabilities
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        입력 시퀀스에 대한 어텐션 가중치 계산
        
        Args:
            x: 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 어텐션 가중치, 형태 (batch_size, sequence_length)
        """
        self.eval()  # 평가 모드로 설정
        with torch.no_grad():
            # LSTM 출력
            lstm_out, _ = self.lstm(x)
            
            # 어텐션 가중치 계산
            attn_weights = self.attention(lstm_out)  # (batch_size, seq_length, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            
        return attn_weights.squeeze(-1)  # (batch_size, seq_length)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 구성 정보 반환
        
        Returns:
            dict: 모델 구성 정보
        """
        return {
            "model_type": "MultiSensorLSTMClassifier",
            "input_size": self.lstm.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }

# 테스트 코드
if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(level=logging.INFO)
    
    # 모델 테스트
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, seq_length, input_size = 16, 50, 4
    hidden_size, num_layers, num_classes = 64, 2, 4
    
    # 테스트 데이터 생성
    x = torch.randn(batch_size, seq_length, input_size).to(device)
    
    # 모델 초기화
    model = MultiSensorLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes
    ).to(device)
    
    # 모델 정보 출력
    model_info = model.get_model_info()
    print("모델 정보:", model_info)
    print(f"파라미터 수: {model_info['parameter_count']:,}")
    
    # 순방향 전파 테스트
    outputs = model(x)
    print(f"출력 형태: {outputs.shape}")
    
    # 예측 테스트
    predicted, probabilities = model.predict(x)
    print(f"예측 클래스 형태: {predicted.shape}")
    print(f"예측 확률 형태: {probabilities.shape}")
    
    # 어텐션 가중치 테스트
    attention_weights = model.get_attention_weights(x)
    print(f"어텐션 가중치 형태: {attention_weights.shape}")