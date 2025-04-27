"""
트랜스포머 기반 텍스트 분류 모델 모듈

이 모듈은 텍스트 분류를 위한 트랜스포머 기반 모델 구현을 제공합니다:
- 기본 트랜스포머 인코더 모델
- 어텐션 시각화 기능
- 모델 평가 유틸리티
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """
    트랜스포머 위치 인코딩 레이어
    
    Args:
        d_model (int): 임베딩 차원
        max_seq_length (int): 최대 시퀀스 길이
        dropout (float): 드롭아웃 비율
    """
    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬 계산
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 사인 함수와 코사인 함수를 번갈아 적용
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 배치 차원 추가 [1, max_seq_length, d_model]
        pe = pe.unsqueeze(0)
        
        # 버퍼로 등록 (모델 매개변수가 아닌 상수)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            x: 입력 텐서, 형태 (batch_size, seq_length, d_model)
            
        Returns:
            torch.Tensor: 위치 인코딩이 추가된 텐서
        """
        # 위치 인코딩 더하기
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """
    텍스트 분류를 위한 트랜스포머 인코더 모델
    
    Args:
        vocab_size (int): 어휘 사전 크기
        embedding_dim (int): 임베딩 차원
        num_heads (int): 어텐션 헤드 수
        hidden_dim (int): 피드포워드 네트워크 은닉층 차원
        num_layers (int): 인코더 레이어 수
        num_classes (int): 분류할 클래스 수
        max_seq_length (int): 최대 시퀀스 길이
        dropout_rate (float): 드롭아웃 비율
    """
    def __init__(self, 
                vocab_size: int, 
                embedding_dim: int = 128, 
                num_heads: int = 8, 
                hidden_dim: int = 512, 
                num_layers: int = 4, 
                num_classes: int = 2, 
                max_seq_length: int = 512, 
                dropout_rate: float = 0.1):
        super(TransformerEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        
        # 토큰 임베딩 레이어
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 위치 인코딩 레이어
        self.pos_encoding = PositionalEncoding(
            embedding_dim, max_seq_length, dropout_rate
        )
        
        # 트랜스포머 인코더 레이어
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers
        )
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # 어텐션 가중치 저장용
        self.attention_weights = None
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self) -> None:
        """가중치 초기화"""
        init_range = 0.1
        self.token_embedding.weight.data.uniform_(-init_range, init_range)
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.uniform_(-init_range, init_range)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, 
               tokens: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None, 
               save_attention: bool = False) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            tokens: 입력 토큰 ID, 형태 (batch_size, seq_length)
            attention_mask: 어텐션 마스크, 형태 (batch_size, seq_length)
                           패딩 토큰을 무시하기 위한 마스크 (1: 실제 토큰, 0: 패딩 토큰)
            save_attention: 어텐션 가중치 저장 여부
            
        Returns:
            torch.Tensor: 클래스별 로짓, 형태 (batch_size, num_classes)
        """
        # 임베딩 및 위치 인코딩
        x = self.token_embedding(tokens)  # (batch_size, seq_length, embedding_dim)
        x = self.pos_encoding(x)
        
        # 어텐션 마스크 변환 (패딩 토큰을 무시하기 위함)
        if attention_mask is not None:
            # 마스크: 0은 True(어텐션 차단), 1은 False(어텐션 허용)로 변환
            src_key_padding_mask = (attention_mask == 0)
        else:
            src_key_padding_mask = None
        
        # 트랜스포머 인코더 통과
        if save_attention:
            # 어텐션 가중치를 저장하기 위한 훅 등록
            attention_weights = []
            
            def get_attention_hook(layer_idx):
                def hook(module, input, output):
                    # 멀티헤드 어텐션 모듈의 출력에서 어텐션 가중치 추출
                    attn_weights = module.self_attn.get_attn_weights()
                    attention_weights.append((layer_idx, attn_weights))
                return hook
            
            # 각 인코더 레이어에 훅 등록
            for i, layer in enumerate(self.transformer_encoder.layers):
                layer.register_forward_hook(get_attention_hook(i))
            
            # 트랜스포머 인코더 통과
            encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            
            # 어텐션 가중치 저장
            self.attention_weights = attention_weights
        else:
            # 트랜스포머 인코더 통과 (일반 추론)
            encoder_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        # 문장 표현 추출 (첫 번째 토큰 [CLS] 또는 평균)
        sentence_repr = encoder_output[:, 0, :]  # [CLS] 토큰 사용
        
        # 분류
        logits = self.classifier(sentence_repr)
        
        return logits
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """
        저장된 어텐션 가중치 반환
        
        Returns:
            List[torch.Tensor]: 레이어별 어텐션 가중치 리스트
        """
        if self.attention_weights is None:
            logger.warning("어텐션 가중치가 저장되지 않았습니다. forward 호출 시 save_attention=True로 설정하세요.")
            return []
        
        return self.attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 구성 정보
        """
        return {
            "model_type": "TransformerEncoder",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "max_seq_length": self.max_seq_length,
            "dropout_rate": self.dropout_rate,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }


class TextCNN(nn.Module):
    """
    텍스트 분류를 위한 CNN 모델
    
    Args:
        vocab_size (int): 어휘 사전 크기
        embedding_dim (int): 임베딩 차원
        filter_sizes (List[int]): 컨볼루션 필터 크기 목록
        num_filters (int): 각 필터 크기별 필터 수
        num_classes (int): 분류할 클래스 수
        max_seq_length (int): 최대 시퀀스 길이
        dropout_rate (float): 드롭아웃 비율
        padding_idx (int): 패딩 토큰 인덱스
    """
    def __init__(self, 
                vocab_size: int, 
                embedding_dim: int = 128, 
                filter_sizes: List[int] = [3, 4, 5], 
                num_filters: int = 100, 
                num_classes: int = 2, 
                max_seq_length: int = 512, 
                dropout_rate: float = 0.5,
                padding_idx: int = 0):
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        
        # 토큰 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # 컨볼루션 레이어
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=filter_size
            )
            for filter_size in filter_sizes
        ])
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout_rate)
        
        # 분류 레이어
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self) -> None:
        """가중치 초기화"""
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            if conv.bias is not None:
                nn.init.constant_(conv.bias, 0)
        
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            tokens: 입력 토큰 ID, 형태 (batch_size, seq_length)
            
        Returns:
            torch.Tensor: 클래스별 로짓, 형태 (batch_size, num_classes)
        """
        # 임베딩 레이어
        embedded = self.embedding(tokens)  # (batch_size, seq_length, embedding_dim)
        
        # 컨볼루션 레이어 입력을 위한 차원 변환
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # 각 필터 크기별 컨볼루션 및 풀링
        pooled_outputs = []
        for conv in self.convs:
            # 컨볼루션
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, seq_length - filter_size + 1)
            
            # 최대 풀링
            pooled = F.max_pool1d(
                conv_out, 
                kernel_size=conv_out.shape[2]
            )  # (batch_size, num_filters, 1)
            
            pooled = pooled.squeeze(2)  # (batch_size, num_filters)
            pooled_outputs.append(pooled)
        
        # 모든 풀링 결과 연결
        cat = torch.cat(pooled_outputs, dim=1)  # (batch_size, num_filters * len(filter_sizes))
        
        # 드롭아웃 적용
        cat = self.dropout(cat)
        
        # 분류 레이어
        logits = self.fc(cat)  # (batch_size, num_classes)
        
        return logits
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 구성 정보
        """
        return {
            "model_type": "TextCNN",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "filter_sizes": self.filter_sizes,
            "num_filters": self.num_filters,
            "num_classes": self.num_classes,
            "max_seq_length": self.max_seq_length,
            "dropout_rate": self.dropout_rate,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }


class BiLSTMAttention(nn.Module):
    """
    텍스트 분류를 위한 양방향 LSTM + 어텐션 모델
    
    Args:
        vocab_size (int): 어휘 사전 크기
        embedding_dim (int): 임베딩 차원
        hidden_dim (int): LSTM 은닉층 차원
        num_layers (int): LSTM 레이어 수
        num_classes (int): 분류할 클래스 수
        dropout_rate (float): 드롭아웃 비율
        padding_idx (int): 패딩 토큰 인덱스
    """
    def __init__(self, 
                vocab_size: int, 
                embedding_dim: int = 128, 
                hidden_dim: int = 256, 
                num_layers: int = 2, 
                num_classes: int = 2, 
                dropout_rate: float = 0.5,
                padding_idx: int = 0):
        super(BiLSTMAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # 토큰 임베딩 레이어
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        # 양방향 LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # 어텐션 레이어
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 드롭아웃 레이어
        self.dropout = nn.Dropout(dropout_rate)
        
        # 분류 레이어
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
        # 어텐션 가중치 저장용
        self.attention_weights = None
        
        # 가중치 초기화
        self._init_weights()
    
    def _init_weights(self) -> None:
        """가중치 초기화"""
        nn.init.xavier_uniform_(self.embedding.weight)
        
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        for module in self.attention.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        순방향 전파
        
        Args:
            tokens: 입력 토큰 ID, 형태 (batch_size, seq_length)
            attention_mask: 어텐션 마스크, 형태 (batch_size, seq_length)
                           패딩 토큰을 무시하기 위한 마스크 (1: 실제 토큰, 0: 패딩 토큰)
            
        Returns:
            torch.Tensor: 클래스별 로짓, 형태 (batch_size, num_classes)
        """
        # 임베딩 레이어
        embedded = self.embedding(tokens)  # (batch_size, seq_length, embedding_dim)
        
        # BiLSTM
        lstm_output, _ = self.lstm(embedded)  # (batch_size, seq_length, hidden_dim*2)
        
        # 어텐션 스코어 계산
        attention_scores = self.attention(lstm_output)  # (batch_size, seq_length, 1)
        
        # 마스킹 적용 (패딩 토큰 무시)
        if attention_mask is not None:
            # 마스크 확장: (batch_size, seq_length) -> (batch_size, seq_length, 1)
            mask_expanded = attention_mask.unsqueeze(-1).float()
            # 마스킹: 패딩 토큰에 매우 작은 값 할당
            attention_scores = attention_scores * mask_expanded + (1 - mask_expanded) * -10000.0
        
        # 어텐션 가중치 계산 (소프트맥스)
        attention_weights = F.softmax(attention_scores, dim=1)  # (batch_size, seq_length, 1)
        self.attention_weights = attention_weights
        
        # 가중 합 계산
        context = torch.sum(attention_weights * lstm_output, dim=1)  # (batch_size, hidden_dim*2)
        
        # 드롭아웃 적용
        context = self.dropout(context)
        
        # 분류 레이어
        logits = self.fc(context)  # (batch_size, num_classes)
        
        return logits
    
    def get_attention_weights(self) -> torch.Tensor:
        """
        저장된 어텐션 가중치 반환
        
        Returns:
            torch.Tensor: 어텐션 가중치, 형태 (batch_size, seq_length, 1)
        """
        if self.attention_weights is None:
            logger.warning("어텐션 가중치가 저장되지 않았습니다. forward를 먼저 호출하세요.")
            return torch.tensor([])
        
        return self.attention_weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        모델 정보 반환
        
        Returns:
            Dict[str, Any]: 모델 구성 정보
        """
        return {
            "model_type": "BiLSTMAttention",
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.num_classes,
            "dropout_rate": self.dropout_rate,
            "parameter_count": sum(p.numel() for p in self.parameters())
        }