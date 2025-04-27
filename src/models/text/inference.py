"""
텍스트 모델 추론 모듈

이 모듈은 훈련된 텍스트 분류 모델을 사용한 추론 기능을 제공합니다:
- 텍스트 입력 전처리
- 모델 예측 수행
- 결과 해석 및 시각화
"""

import torch
import numpy as np
import logging
import json
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.models.text.model_factory import TextModelFactory, BiLSTMAttention
from src.data.text.preprocessor import TextPreprocessor
from src.data.text.tokenizer import Tokenizer, WordPieceTokenizer, BPETokenizer

# 로깅 설정
logger = logging.getLogger(__name__)

class TextInferenceEngine:
    """
    텍스트 모델 추론 엔진
    
    훈련된 텍스트 분류 모델을 사용한 추론을 수행합니다.
    """
    
    def __init__(self, 
                model_path: str, 
                model_info_path: str,
                tokenizer_path: Optional[str] = None,
                device: Optional[torch.device] = None):
        """
        추론 엔진 초기화
        
        Args:
            model_path: 모델 가중치 파일 경로
            model_info_path: 모델 정보 파일 경로
            tokenizer_path: 토크나이저 파일 경로 (없으면 기본 토크나이저 사용)
            device: 추론에 사용할 장치 (CPU/GPU)
        """
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 모델 정보 로드
        with open(model_info_path, 'r', encoding='utf-8') as f:
            self.model_info = json.load(f)
        
        # 모델 로드
        self.model = TextModelFactory.load_model_from_path(
            model_path=model_path,
            model_info=self.model_info,
            device=self.device
        )
        
        # 토크나이저 로드
        if tokenizer_path and os.path.exists(tokenizer_path):
            # 토크나이저 타입 확인
            if 'wordpiece' in tokenizer_path.lower():
                self.tokenizer = WordPieceTokenizer.load_from_file(tokenizer_path)
            elif 'bpe' in tokenizer_path.lower():
                self.tokenizer = BPETokenizer.load_from_file(tokenizer_path)
            else:
                self.tokenizer = Tokenizer.load_from_file(tokenizer_path)
            
            logger.info(f"토크나이저를 '{tokenizer_path}'에서 로드했습니다.")
        else:
            # 기본 토크나이저 생성
            self.tokenizer = TextPreprocessor()
            logger.warning(f"토크나이저 파일이 없습니다. 기본 텍스트 전처리기를 사용합니다.")
        
        # 클래스 레이블 매핑
        self.class_mapping = self.model_info.get('class_mapping', {})
        self.inverse_class_mapping = {int(idx): label for idx, label in self.class_mapping.items()}
        
        # 최대 시퀀스 길이
        self.max_seq_length = self.model_info.get('max_seq_length', 512)
        
        # 평가 모드로 설정
        self.model.eval()
        
        logger.info(f"텍스트 추론 엔진 초기화 완료: {self.model_info.get('model_type')} 모델")
    
    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        입력 텍스트 전처리
        
        Args:
            text: 입력 텍스트
            
        Returns:
            Dict[str, torch.Tensor]: 전처리된 텐서 (토큰, 마스크 등)
        """
        # 텍스트 정제 및 토큰화
        if hasattr(self.tokenizer, 'tokenize') and callable(self.tokenizer.tokenize):
            # Tokenizer 또는 서브워드 토크나이저 사용
            tokens = self.tokenizer.tokenize(text)
            token_ids = self.tokenizer.encode(text)
            
            # 시퀀스 길이 제한
            if len(token_ids) > self.max_seq_length:
                token_ids = token_ids[:self.max_seq_length]
            
            # 패딩 및 어텐션 마스크 생성
            padding_length = self.max_seq_length - len(token_ids)
            attention_mask = [1] * len(token_ids) + [0] * padding_length
            token_ids = token_ids + [0] * padding_length  # 0은 PAD 토큰
            
        else:
            # TextPreprocessor 사용
            tokens = self.tokenizer.tokenize(text)
            
            # 시퀀스 길이 제한
            if len(tokens) > self.max_seq_length:
                tokens = tokens[:self.max_seq_length]
            
            # 단어->인덱스 변환 (어휘 사전이 있는 경우)
            if hasattr(self.tokenizer, 'word_index') and self.tokenizer.word_index:
                token_ids = [self.tokenizer.word_index.get(token, 1) for token in tokens]  # 1은 UNK 토큰
            else:
                # 간단한 어휘 사전이 없는 경우, 일련번호로 변환
                unique_tokens = sorted(set(tokens))
                token_to_id = {token: idx for idx, token in enumerate(unique_tokens, start=1)}
                token_ids = [token_to_id.get(token, 0) for token in tokens]
            
            # 패딩 및 어텐션 마스크 생성
            padding_length = self.max_seq_length - len(token_ids)
            attention_mask = [1] * len(token_ids) + [0] * padding_length
            token_ids = token_ids + [0] * padding_length  # 0은 PAD 토큰
        
        # 텐서 변환
        tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(self.device)
        mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(self.device)
        
        return {
            'tokens': tokens_tensor,
            'attention_mask': mask_tensor,
            'original_tokens': tokens
        }
    
    def predict(self, text: str, return_probs: bool = False) -> Union[int, Dict[str, Any]]:
        """
        텍스트 분류 예측 수행
        
        Args:
            text: 입력 텍스트
            return_probs: 각 클래스별 확률 반환 여부
            
        Returns:
            Union[int, Dict[str, Any]]: 예측 클래스 또는 예측 정보
        """
        # 텍스트 전처리
        inputs = self.preprocess_text(text)
        
        # 어텐션 사용 여부에 따라 모델 호출
        with torch.no_grad():
            if hasattr(self.model, 'get_attention_weights'):
                # 트랜스포머 또는 BiLSTM+Attention 모델
                model_output = self.model(
                    inputs['tokens'], 
                    inputs['attention_mask'], 
                    save_attention=True if hasattr(self.model, 'get_attention_weights') else False
                )
            else:
                # TextCNN 등 어텐션이 없는 모델
                model_output = self.model(inputs['tokens'])
        
        # 예측 클래스 및 확률 계산
        probabilities = torch.softmax(model_output, dim=1)[0].cpu().numpy()
        predicted_class = int(torch.argmax(model_output, dim=1).item())
        
        # 클래스 레이블 가져오기
        if predicted_class in self.inverse_class_mapping:
            predicted_label = self.inverse_class_mapping[predicted_class]
        else:
            predicted_label = f"Class_{predicted_class}"
        
        # 결과 반환
        if return_probs:
            class_probs = {
                self.inverse_class_mapping.get(i, f"Class_{i}"): float(prob) 
                for i, prob in enumerate(probabilities)
            }
            
            return {
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'probability': float(probabilities[predicted_class]),
                'class_probabilities': class_probs,
                'tokens': inputs['original_tokens']
            }
        else:
            return predicted_class
    
    def visualize_attention(self, text: str, output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """
        어텐션 가중치 시각화
        
        Args:
            text: 입력 텍스트
            output_path: 시각화 이미지 저장 경로
            
        Returns:
            Optional[plt.Figure]: 시각화 그림 객체 (None: 어텐션 가중치 없음)
        """
        # 어텐션 지원 확인
        if not hasattr(self.model, 'get_attention_weights'):
            logger.warning("현재 모델은 어텐션 시각화를 지원하지 않습니다.")
            return None
        
        # 텍스트 전처리 및 예측
        inputs = self.preprocess_text(text)
        self.predict(text, return_probs=True)  # 어텐션 가중치 계산을 위한 예측
        
        # 어텐션 가중치 가져오기
        if isinstance(self.model, BiLSTMAttention):
            # BiLSTM+Attention 모델
            attention_weights = self.model.get_attention_weights()
            if attention_weights is None or len(attention_weights) == 0:
                logger.warning("어텐션 가중치를 가져올 수 없습니다.")
                return None
            
            attention_weights = attention_weights.squeeze(2).cpu().numpy()[0]
            
            # 토큰 길이에 맞게 어텐션 가중치 자르기
            tokens = inputs['original_tokens']
            attention_weights = attention_weights[:len(tokens)]
            
            # 시각화
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(tokens)), attention_weights, align='center', alpha=0.7)
            plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
            plt.xlabel('Tokens')
            plt.ylabel('Attention Weight')
            plt.title('BiLSTM Attention Weights')
            plt.tight_layout()
            
        elif hasattr(self.model, 'get_attention_weights') and self.model.__class__.__name__ == 'TransformerEncoder':
            # 트랜스포머 모델
            attention_layers = self.model.get_attention_weights()
            if not attention_layers:
                logger.warning("어텐션 가중치를 가져올 수 없습니다.")
                return None
            
            # 토큰 길이에 맞게 어텐션 가중치 자르기
            tokens = inputs['original_tokens']
            
            # 여러 레이어의 어텐션 가중치 시각화
            num_layers = len(attention_layers)
            fig, axes = plt.subplots(num_layers, 1, figsize=(10, 3 * num_layers))
            
            for i, (layer_idx, attn_weights) in enumerate(attention_layers):
                # 여러 헤드의 어텐션 가중치 평균
                head_mean = attn_weights[0].mean(dim=0).cpu().numpy()
                
                # 유효한 토큰에 대한 어텐션 가중치 추출
                head_mean = head_mean[:len(tokens), :len(tokens)]
                
                # 히트맵 그리기
                ax = axes[i] if num_layers > 1 else axes
                sns.heatmap(head_mean, annot=False, cmap='viridis', ax=ax)
                ax.set_title(f'Layer {layer_idx} Attention')
                ax.set_xlabel('Token Position (Target)')
                ax.set_ylabel('Token Position (Source)')
                ax.set_xticklabels(tokens, rotation=45, ha='right')
                ax.set_yticklabels(tokens, rotation=0)
                
            plt.tight_layout()
        
        else:
            logger.warning("알 수 없는 모델 유형의 어텐션 가중치입니다.")
            return None
        
        # 저장
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"어텐션 시각화 저장 완료: {output_path}")
        
        return plt.gcf()
    
    def batch_predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        배치 예측 수행
        
        Args:
            texts: 입력 텍스트 목록
            
        Returns:
            List[Dict[str, Any]]: 각 텍스트별 예측 결과
        """
        results = []
        for text in texts:
            result = self.predict(text, return_probs=True)
            results.append(result)
        
        return results


