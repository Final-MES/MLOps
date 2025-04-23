# src/models/base_model.py - 모델 기본 클래스
class BaseModel(nn.Module):
    """모든 모델의 기본 클래스"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(__name__)
    
    def save(self, path: str) -> None:
        """모델을 저장"""
        torch.save(self.state_dict(), path)
        self.logger.info(f"모델 저장: {path}")
    
    def load(self, path: str) -> None:
        """모델을 로드"""
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        self.logger.info(f"모델 로드: {path}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        raise NotImplementedError("서브클래스에서 구현해야 합니다")

# src/models/lstm_model.py - LSTM 모델 클래스 (일관된 인터페이스로 수정)
class LSTMModel(BaseModel):
    """시계열 센서 데이터 예측을 위한 LSTM 모델"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        output_size: int, 
        dropout_rate: float = 0.2
    ):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # 구현...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순방향 전파"""
        # 구현...
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """예측 수행"""
        # 구현...
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        # 구현...

# src/models/multivariate_model.py - 다변량 모델 클래스
class MultivariateLSTMClassifier(BaseModel):
    """다변량 시계열 데이터를 위한 LSTM 기반 분류 모델"""
    
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int, 
        num_layers: int, 
        num_classes: int, 
        dropout_rate: float = 0.2
    ):
        super(MultivariateLSTMClassifier, self).__init__()
        # 구현...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """순방향 전파"""
        # 구현...
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        # 구현...