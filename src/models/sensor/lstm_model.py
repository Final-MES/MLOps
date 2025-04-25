import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional, Dict, Any


class LSTMModel(nn.Module):
    """
    LSTM 모델 클래스 - 시계열 센서 데이터 예측을 위한 딥러닝 모델
    
    Args:
        input_size (int): 입력 특성의 수 (센서 변수 개수)
        hidden_size (int): LSTM 은닉층의 크기
        num_layers (int): LSTM 레이어의 수
        output_size (int): 출력 크기 (예측 변수 개수, 보통 1)
        dropout_rate (float, optional): 드롭아웃 비율. 기본값은 0.2
    """
    def __init__(self, 
                 input_size: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 output_size: int, 
                 dropout_rate: float = 0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 완전 연결 레이어
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x (torch.Tensor): 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            torch.Tensor: 출력 텐서, 형태 (batch_size, output_size)
        """
        # LSTM 출력 (batch_size, seq_length, hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # 마지막 시퀀스의 출력만 사용
        out = self.fc(lstm_out[:, -1, :])
        return out
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        예측 수행
        
        Args:
            x (torch.Tensor): 입력 텐서, 형태 (batch_size, sequence_length, input_size)
            
        Returns:
            np.ndarray: 예측 결과, numpy 배열로 변환됨
        """
        self.eval()  # 평가 모드 설정
        with torch.no_grad():
            predictions = self.forward(x)
        return predictions.cpu().numpy()
    
    def predict_sequence(self, 
                         initial_sequence: torch.Tensor, 
                         steps: int, 
                         feature_indices: Optional[List[int]] = None) -> np.ndarray:
        """
        미래 시퀀스 예측 (다중 스텝)
        
        Args:
            initial_sequence (torch.Tensor): 초기 시퀀스, 형태 (1, sequence_length, input_size)
            steps (int): 예측할 미래 스텝 수
            feature_indices (List[int], optional): 예측값을 다음 입력의 어떤 특성으로 사용할지 지정
                                             None이면 예측값은 마지막 특성으로 사용됨
                                             
        Returns:
            np.ndarray: 예측된 시퀀스, 형태 (steps, output_size)
        """
        self.eval()  # 평가 모드 설정
        
        # 초기 시퀀스 복사
        current_sequence = initial_sequence.clone()
        predictions = []
        
        with torch.no_grad():
            for _ in range(steps):
                # 현재 시퀀스로 다음 값 예측
                pred = self.forward(current_sequence)
                predictions.append(pred.item())
                
                # 새 시퀀스 생성: 가장 오래된 타임스텝 제거 후 새 예측값 추가
                # 여러 특성을 가진 경우 적절히 처리
                if feature_indices is not None:
                    # 예측값을 특정 특성에 할당
                    new_features = current_sequence[0, -1].clone()
                    for i, idx in enumerate(feature_indices):
                        if i < pred.shape[1]:  # 예측 차원 수 확인
                            new_features[idx] = pred[0, i]
                else:
                    # 기본적으로 예측값을 마지막 특성으로 사용
                    new_features = current_sequence[0, -1].clone()
                    new_features[-1] = pred.item()
                
                # 새 시퀀스 생성
                current_sequence = torch.cat([
                    current_sequence[:, 1:], 
                    new_features.unsqueeze(0).unsqueeze(0)
                ], dim=1)
        
        return np.array(predictions)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 아키텍처 및 구성 정보
        """
        return {
            "model_type": "LSTM",
            "input_size": self.lstm.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.fc.out_features,
            "has_dropout": self.lstm.dropout > 0,
            "dropout_rate": self.lstm.dropout if self.lstm.dropout > 0 else None,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }