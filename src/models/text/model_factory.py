"""
텍스트 모델 팩토리 모듈

이 모듈은 텍스트 분류를 위한 다양한 모델을 생성하는 팩토리 패턴을 구현합니다.
모델 유형과 파라미터에 따라 적절한 모델 인스턴스를 반환합니다.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, Optional
import logging

from src.models.text.transformer_model import TransformerEncoder, TextCNN, BiLSTMAttention

# 로깅 설정
logger = logging.getLogger(__name__)

class TextModelFactory:
    """
    텍스트 모델 팩토리 클래스
    
    다양한 텍스트 분류 모델을 생성하는 팩토리 메서드를 제공합니다.
    """
    
    @staticmethod
    def create_model(model_type: str, model_params: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
        """
        모델 생성
        
        Args:
            model_type: 모델 유형 ('transformer', 'cnn', 'bilstm')
            model_params: 모델 파라미터 딕셔너리
            device: 모델을 배치할 장치 (CPU/GPU)
            
        Returns:
            nn.Module: 생성된 모델 인스턴스
        """
        model_type = model_type.lower()
        
        # 필수 파라미터 확인
        if 'vocab_size' not in model_params:
            raise ValueError("모델 생성에 필요한 'vocab_size' 파라미터가 없습니다.")
        
        if 'num_classes' not in model_params:
            raise ValueError("모델 생성에 필요한 'num_classes' 파라미터가 없습니다.")
        
        # 트랜스포머 인코더 모델
        if model_type == 'transformer':
            model = TransformerEncoder(
                vocab_size=model_params['vocab_size'],
                embedding_dim=model_params.get('embedding_dim', 128),
                num_heads=model_params.get('num_heads', 8),
                hidden_dim=model_params.get('hidden_dim', 512),
                num_layers=model_params.get('num_layers', 4),
                num_classes=model_params['num_classes'],
                max_seq_length=model_params.get('max_seq_length', 512),
                dropout_rate=model_params.get('dropout_rate', 0.1)
            )
            logger.info(f"TransformerEncoder 모델 생성 완료: {model.get_model_info()}")
            
        # TextCNN 모델
        elif model_type == 'cnn':
            model = TextCNN(
                vocab_size=model_params['vocab_size'],
                embedding_dim=model_params.get('embedding_dim', 128),
                filter_sizes=model_params.get('filter_sizes', [3, 4, 5]),
                num_filters=model_params.get('num_filters', 100),
                num_classes=model_params['num_classes'],
                max_seq_length=model_params.get('max_seq_length', 512),
                dropout_rate=model_params.get('dropout_rate', 0.5),
                padding_idx=model_params.get('padding_idx', 0)
            )
            logger.info(f"TextCNN 모델 생성 완료: {model.get_model_info()}")
            
        # BiLSTM+Attention 모델
        elif model_type == 'bilstm':
            model = BiLSTMAttention(
                vocab_size=model_params['vocab_size'],
                embedding_dim=model_params.get('embedding_dim', 128),
                hidden_dim=model_params.get('hidden_dim', 256),
                num_layers=model_params.get('num_layers', 2),
                num_classes=model_params['num_classes'],
                dropout_rate=model_params.get('dropout_rate', 0.5),
                padding_idx=model_params.get('padding_idx', 0)
            )
            logger.info(f"BiLSTMAttention 모델 생성 완료: {model.get_model_info()}")
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 지정된 장치로 모델 이동
        if device is not None:
            model = model.to(device)
            logger.info(f"모델을 장치 {device}로 이동했습니다.")
        
        return model
    
    @staticmethod
    def load_model_from_path(model_path: str, model_info: Dict[str, Any], device: Optional[torch.device] = None) -> nn.Module:
        """
        저장된 모델 로드
        
        Args:
            model_path: 모델 가중치 파일 경로
            model_info: 모델 정보 딕셔너리
            device: 모델을 배치할 장치 (CPU/GPU)
            
        Returns:
            nn.Module: 로드된 모델 인스턴스
        """
        # 모델 유형 결정
        model_type = model_info.get('model_type', '').lower()
        
        if 'transformer' in model_type:
            # 트랜스포머 인코더 모델
            model = TransformerEncoder(
                vocab_size=model_info.get('vocab_size'),
                embedding_dim=model_info.get('embedding_dim', 128),
                num_heads=model_info.get('num_heads', 8),
                hidden_dim=model_info.get('hidden_dim', 512),
                num_layers=model_info.get('num_layers', 4),
                num_classes=model_info.get('num_classes'),
                max_seq_length=model_info.get('max_seq_length', 512),
                dropout_rate=model_info.get('dropout_rate', 0.1)
            )
            logger.info(f"TransformerEncoder 모델 로드: {model_info}")
            
        elif 'textcnn' in model_type or model_type == 'cnn':
            # TextCNN 모델
            model = TextCNN(
                vocab_size=model_info.get('vocab_size'),
                embedding_dim=model_info.get('embedding_dim', 128),
                filter_sizes=model_info.get('filter_sizes', [3, 4, 5]),
                num_filters=model_info.get('num_filters', 100),
                num_classes=model_info.get('num_classes'),
                max_seq_length=model_info.get('max_seq_length', 512),
                dropout_rate=model_info.get('dropout_rate', 0.5),
                padding_idx=model_info.get('padding_idx', 0)
            )
            logger.info(f"TextCNN 모델 로드: {model_info}")
            
        elif 'bilstm' in model_type:
            # BiLSTM+Attention 모델
            model = BiLSTMAttention(
                vocab_size=model_info.get('vocab_size'),
                embedding_dim=model_info.get('embedding_dim', 128),
                hidden_dim=model_info.get('hidden_dim', 256),
                num_layers=model_info.get('num_layers', 2),
                num_classes=model_info.get('num_classes'),
                dropout_rate=model_info.get('dropout_rate', 0.5),
                padding_idx=model_info.get('padding_idx', 0)
            )
            logger.info(f"BiLSTMAttention 모델 로드: {model_info}")
            
        else:
            raise ValueError(f"지원하지 않는 모델 유형: {model_type}")
        
        # 모델 가중치 로드
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        logger.info(f"모델 가중치를 '{model_path}'에서 로드했습니다.")
        
        # 지정된 장치로 모델 이동
        if device is not None:
            model = model.to(device)
            logger.info(f"모델을 장치 {device}로 이동했습니다.")
        
        # 평가 모드로 설정
        model.eval()
        
        return model

    @staticmethod
    def save_pretrained_embeddings(model: nn.Module, embedding_path: str) -> bool:
        """
        학습된 임베딩 저장
        
        Args:
            model: 학습된 모델
            embedding_path: 임베딩 저장 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 모델 유형에 따라 임베딩 레이어 추출
            if isinstance(model, TransformerEncoder):
                embedding_weights = model.token_embedding.weight.data
            elif isinstance(model, TextCNN) or isinstance(model, BiLSTMAttention):
                embedding_weights = model.embedding.weight.data
            else:
                logger.error(f"지원하지 않는 모델 유형: {type(model)}")
                return False
            
            # 임베딩 저장
            torch.save(embedding_weights, embedding_path)
            logger.info(f"학습된 임베딩을 '{embedding_path}'에 저장했습니다.")
            return True
            
        except Exception as e:
            logger.error(f"임베딩 저장 중 오류 발생: {str(e)}")
            return False