#!/usr/bin/env python
"""
텍스트 데이터 처리 CLI 모듈

이 스크립트는 텍스트 데이터 처리, 토큰화, 모델 학습, 평가를 위한
대화형 명령줄 인터페이스를 제공합니다.
"""

import os
import sys
import time
import logging
import argparse
import torch
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import shutil

# 프로젝트 루트 경로 추가
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 모듈 임포트
from src.data.text.preprocessor import TextPreprocessor
from src.data.text.tokenizer import Tokenizer, WordPieceTokenizer, BPETokenizer
from src.data.common.utils import split_dataset

# 기본 CLI 클래스 임포트
from src.cli.base_cli import BaseCLI

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(project_root, 'logs', 'text_cli.log'))
    ]
)
logger = logging.getLogger(__name__)

class TextCLI(BaseCLI):
    """
    텍스트 데이터 처리 CLI 클래스
    
    이 클래스는 텍스트 데이터 전처리, 토큰화, 모델 훈련을 위한
    대화형 인터페이스를 제공합니다.
    """
    
    def __init__(self):
        """텍스트 CLI 초기화"""
        super().__init__(title="텍스트 데이터 처리 시스템")
        
        # 상태 정보 초기화
        self.state = {
            'preprocessor': None,
            'tokenizer': None,
            'dataset': None,
            'preprocessed_texts': None,
            'tokenized_data': None,
            'model': None,
            'train_data': None,
            'val_data': None,
            'test_data': None
        }
        
        # 전처리 설정
        self.preprocess_params = {
            'lowercase': True,
            'remove_punctuation': True,
            'remove_numbers': False,
            'remove_stopwords': True,
            'stemming': False,
            'lemmatization': False
        }
        
        # 토큰화 설정
        self.tokenize_params = {
            'tokenizer_type': 'basic',  # 'basic', 'wordpiece', 'bpe'
            'vocab_size': 10000,
            'min_frequency': 2
        }
        
        # 모델 설정
        self.model_params = {
            'embed_dim': 128,
            'hidden_dim': 256,
            'num_layers': 2,
            'dropout_rate': 0.2
        }
        
        # 학습 설정
        self.training_params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 10
        }
        
        # 기본 경로 설정
        self.paths = {
            'data_dir': os.path.join(project_root, 'data', 'text'),
            'output_dir': os.path.join(project_root, 'data', 'processed', 'text'),
            'model_dir': os.path.join(project_root, 'models', 'text'),
            'plot_dir': os.path.join(project_root, 'plots', 'text')
        }
        
        # 필요한 디렉토리 생성
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)
        
        logger.info("텍스트 데이터 처리 CLI 초기화 완료")
    
    def main_menu(self) -> None:
        """메인 메뉴 표시"""
        while True:
            self.print_header()
            print("텍스트 데이터 처리 시스템에 오신 것을 환영합니다.")
            print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
            
            menu_options = [
                "데이터 로드",
                "전처리 설정",
                "텍스트 전처리",
                "토큰화 설정",
                "텍스트 토큰화",
                "데이터 분할",
                "데이터 시각화",
                "토크나이저 저장/로드",
                "시스템 설정",
                "종료"
            ]
            
            self.print_status()
            choice = self.show_menu(menu_options, "메인 메뉴")
            
            if choice == 0:
                self.load_data_menu()
            elif choice == 1:
                self.preprocess_settings_menu()
            elif choice == 2:
                self.preprocess_text_menu()
            elif choice == 3:
                self.tokenize_settings_menu()
            elif choice == 4:
                self.tokenize_text_menu()
            elif choice == 5:
                self.split_data_menu()
            elif choice == 6:
                self.visualize_data_menu()
            elif choice == 7:
                self.tokenizer_save_load_menu()
            elif choice == 8:
                self.system_settings_menu()
            elif choice == 9:
                print("\n프로그램을 종료합니다. 감사합니다!")
                break
    
    def print_status(self) -> None:
        """현재 상태 출력"""
        print("\n현재 상태:")
        print("-" * 40)
        
        if self.state['dataset'] is not None:
            print(f"✅ 데이터: {len(self.state['dataset'])}개 텍스트 로드됨")
        else:
            print("❌ 데이터: 로드되지 않음")
        
        if self.state['preprocessor'] is not None:
            print(f"✅ 전처리기: 설정됨 (소문자화={self.preprocess_params['lowercase']}, "
                  f"구두점 제거={self.preprocess_params['remove_punctuation']})")
        else:
            print("❌ 전처리기: 설정되지 않음")
        
        if self.state['preprocessed_texts'] is not None:
            print(f"✅ 전처리된 텍스트: {len(self.state['preprocessed_texts'])}개")
        else:
            print("❌ 전처리된 텍스트: 없음")
        
        if self.state['tokenizer'] is not None:
            print(f"✅ 토크나이저: {self.tokenize_params['tokenizer_type']} "
                  f"(어휘 크기={self.tokenize_params['vocab_size']})")
        else:
            print("❌ 토크나이저: 설정되지 않음")
        
        if self.state['tokenized_data'] is not None:
            print(f"✅ 토큰화된 데이터: 사용 가능")
        else:
            print("❌ 토큰화된 데이터: 없음")
        
        if self.state['train_data'] is not None:
            print(f"✅ 분할된 데이터: 훈련={len(self.state['train_data'])}, "
                  f"검증={len(self.state['val_data'])}, 테스트={len(self.state['test_data'])}")
        else:
            print("❌ 분할된 데이터: 없음")
        
        print("-" * 40)
    
    def load_data_menu(self) -> None:
        """데이터 로드 메뉴"""
        self.print_header("텍스트 데이터 로드")
        
        print("텍스트 데이터를 로드합니다.")
        print("파일 또는 디렉토리에서 텍스트를 로드할 수 있습니다.\n")
        
        # 로드 방식 선택
        load_options = ["파일에서 로드", "디렉토리에서 로드", "CSV/TSV 파일에서 로드"]
        load_choice = self.show_menu(load_options, "로드 방식")
        
        try:
            if load_choice == 0:  # 파일에서 로드
                file_path = self.get_input("텍스트 파일 경로")
                
                if not os.path.exists(file_path):
                    self.show_error(f"파일이 존재하지 않습니다: {file_path}")
                    self.wait_for_user()
                    return
                
                # 파일에서 텍스트 로드
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 텍스트 분할 방식 선택
                split_options = ["줄바꿈으로 분할", "문단으로 분할", "전체 텍스트를 하나로 유지"]
                split_choice = self.show_menu(split_options, "텍스트 분할 방식")
                
                if split_choice == 0:  # 줄바꿈으로 분할
                    texts = [line.strip() for line in text.split('\n') if line.strip()]
                elif split_choice == 1:  # 문단으로 분할
                    texts = [para.strip() for para in text.split('\n\n') if para.strip()]
                else:  # 전체 텍스트
                    texts = [text]
                
                self.state['dataset'] = texts
                self.show_success(f"{len(texts)}개의 텍스트를 로드했습니다.")
                
            elif load_choice == 1:  # 디렉토리에서 로드
                dir_path = self.get_input("텍스트 파일 디렉토리 경로")
                
                if not os.path.isdir(dir_path):
                    self.show_error(f"디렉토리가 존재하지 않습니다: {dir_path}")
                    self.wait_for_user()
                    return
                
                # 파일 확장자 필터
                file_ext = self.get_input("파일 확장자 (예: .txt, .md)", ".txt")
                
                # 디렉토리에서 파일 목록 가져오기
                files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
                        if f.endswith(file_ext) and os.path.isfile(os.path.join(dir_path, f))]
                
                if not files:
                    self.show_error(f"디렉토리에 {file_ext} 파일이 없습니다.")
                    self.wait_for_user()
                    return
                
                # 파일에서 텍스트 로드
                texts = []
                for file_path in files:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_text = f.read().strip()
                            texts.append(file_text)
                    except Exception as e:
                        self.show_warning(f"파일 로드 중 오류 발생: {file_path} - {str(e)}")
                
                self.state['dataset'] = texts
                self.show_success(f"{len(texts)}개의 텍스트 파일을 로드했습니다.")
                
            elif load_choice == 2:  # CSV/TSV 파일에서 로드
                file_path = self.get_input("CSV/TSV 파일 경로")
                
                if not os.path.exists(file_path):
                    self.show_error(f"파일이 존재하지 않습니다: {file_path}")
                    self.wait_for_user()
                    return
                
                # 구분자 및 텍스트 컬럼 선택
                delimiter = self.get_input("구분자 (CSV: ',', TSV: '\\t')", ",")
                if delimiter == '\\t':
                    delimiter = '\t'
                
                # CSV 파일 로드
                import pandas as pd
                df = pd.read_csv(file_path, delimiter=delimiter)
                
                # 컬럼 목록 표시
                columns = df.columns.tolist()
                print("\n사용 가능한 컬럼:")
                for i, col in enumerate(columns):
                    print(f"{i+1}. {col}")
                
                # 텍스트 컬럼 선택
                while True:
                    col_idx = self.get_numeric_input("\n텍스트 컬럼 번호", 1, min_val=1, max_val=len(columns))
                    col_name = columns[col_idx-1]
                    
                    # 레이블 컬럼 선택 여부
                    has_label = self.get_yes_no_input("레이블 컬럼이 있습니까?")
                    
                    if has_label:
                        label_idx = self.get_numeric_input("레이블 컬럼 번호", 2, min_val=1, max_val=len(columns))
                        label_name = columns[label_idx-1]
                        
                        # 텍스트와 레이블 추출
                        texts = df[col_name].fillna("").tolist()
                        labels = df[label_name].tolist()
                        
                        # 상태 업데이트
                        self.state['dataset'] = texts
                        self.state['labels'] = labels
                        self.show_success(f"{len(texts)}개의 텍스트와 레이블을 로드했습니다.")
                    else:
                        # 텍스트만 추출
                        texts = df[col_name].fillna("").tolist()
                        
                        # 상태 업데이트
                        self.state['dataset'] = texts
                        self.show_success(f"{len(texts)}개의 텍스트를 로드했습니다.")
                    
                    break
            
            # 데이터 미리보기
            if self.state['dataset'] and len(self.state['dataset']) > 0:
                preview_count = min(5, len(self.state['dataset']))
                print("\n데이터 미리보기:")
                for i in range(preview_count):
                    preview_text = self.state['dataset'][i]
                    # 너무 긴 텍스트는 일부만 표시
                    if len(preview_text) > 100:
                        preview_text = preview_text[:100] + "..."
                    print(f"{i+1}. {preview_text}")
                
                if hasattr(self.state, 'labels') and self.state['labels'] is not None:
                    print("\n레이블 미리보기:")
                    for i in range(preview_count):
                        print(f"{i+1}. {self.state['labels'][i]}")
            
        except Exception as e:
            self.show_error(f"데이터 로드 중 오류 발생: {str(e)}")
            logger.exception("데이터 로드 중 예외 발생")
        
        self.wait_for_user()
    
    def preprocess_settings_menu(self) -> None:
        """전처리 설정 메뉴"""
        self.print_header("텍스트 전처리 설정")
        
        print("텍스트 전처리 설정을 변경합니다.\n")
        
        # 현재 설정 표시
        print("현재 설정:")
        for param, value in self.preprocess_params.items():
            print(f"- {param}: {value}")
        
        print("\n설정 변경:")
        
        # 소문자 변환
        self.preprocess_params['lowercase'] = self.get_yes_no_input(
            "소문자 변환", self.preprocess_params['lowercase']
        )
        
        # 구두점 제거
        self.preprocess_params['remove_punctuation'] = self.get_yes_no_input(
            "구두점 제거", self.preprocess_params['remove_punctuation']
        )
        
        # 숫자 제거
        self.preprocess_params['remove_numbers'] = self.get_yes_no_input(
            "숫자 제거", self.preprocess_params['remove_numbers']
        )
        
        # 불용어 제거
        self.preprocess_params['remove_stopwords'] = self.get_yes_no_input(
            "불용어 제거", self.preprocess_params['remove_stopwords']
        )
        
        # 어간 추출
        self.preprocess_params['stemming'] = self.get_yes_no_input(
            "어간 추출 (Porter Stemmer)", self.preprocess_params['stemming']
        )
        
        # 표제어 추출
        self.preprocess_params['lemmatization'] = self.get_yes_no_input(
            "표제어 추출 (WordNet Lemmatizer)", self.preprocess_params['lemmatization']
        )
        
        # 전처리기 생성
        try:
            self.state['preprocessor'] = TextPreprocessor(
                lowercase=self.preprocess_params['lowercase'],
                remove_punctuation=self.preprocess_params['remove_punctuation'],
                remove_numbers=self.preprocess_params['remove_numbers'],
                remove_stopwords=self.preprocess_params['remove_stopwords'],
                stemming=self.preprocess_params['stemming'],
                lemmatization=self.preprocess_params['lemmatization']
            )
            
            self.show_success("전처리 설정이 변경되었습니다.")
            
        except Exception as e:
            self.show_error(f"전처리기 생성 중 오류 발생: {str(e)}")
            logger.exception("전처리기 생성 중 예외 발생")
        
        self.wait_for_user()
    
    def preprocess_text_menu(self) -> None:
        """텍스트 전처리 메뉴"""
        self.print_header("텍스트 전처리")
        
        # 데이터 확인
        if not self.state['dataset']:
            self.show_error("텍스트 데이터가 로드되지 않았습니다. 먼저 데이터를 로드하세요.")
            self.wait_for_user()
            return
        
        # 전처리기 확인
        if not self.state['preprocessor']:
            self.show_message("전처리기가 설정되지 않았습니다. 기본 설정으로 생성합니다.")
            self.state['preprocessor'] = TextPreprocessor(
                lowercase=self.preprocess_params['lowercase'],
                remove_punctuation=self.preprocess_params['remove_punctuation'],
                remove_numbers=self.preprocess_params['remove_numbers'],
                remove_stopwords=self.preprocess_params['remove_stopwords'],
                stemming=self.preprocess_params['stemming'],
                lemmatization=self.preprocess_params['lemmatization']
            )
        
        try:
            # 전처리 진행
            self.show_message(f"전처리 시작: {len(self.state['dataset'])}개 텍스트")
            
            preprocessed_texts = []
            for i, text in enumerate(self.state['dataset']):
                # 진행상황 표시 (10% 단위)
                if i % max(1, len(self.state['dataset']) // 10) == 0:
                    progress = i / len(self.state['dataset']) * 100
                    self.show_message(f"진행 중: {progress:.0f}% ({i}/{len(self.state['dataset'])})")
                
                # 텍스트 정제
                cleaned_text = self.state['preprocessor'].clean_text(text)
                preprocessed_texts.append(cleaned_text)
            
            # 상태 업데이트
            self.state['preprocessed_texts'] = preprocessed_texts
            
            self.show_success(f"전처리 완료: {len(preprocessed_texts)}개 텍스트")
            
            # 전처리 결과 미리보기
            preview_count = min(5, len(preprocessed_texts))
            print("\n전처리 결과 미리보기:")
            for i in range(preview_count):
                print(f"\n원본: {self.state['dataset'][i][:100]}...")
                print(f"전처리: {preprocessed_texts[i][:100]}...")
            
            # 결과 저장 여부 확인
            save_result = self.get_yes_no_input("\n전처리 결과를 파일로 저장하시겠습니까?")
            if save_result:
                output_path = self.get_input(
                    "저장 파일 경로", 
                    os.path.join(self.paths['output_dir'], "preprocessed_texts.txt")
                )
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    for text in preprocessed_texts:
                        f.write(text + '\n')
                
                self.show_success(f"전처리 결과가 '{output_path}'에 저장되었습니다.")
        
        except Exception as e:
            self.show_error(f"텍스트 전처리 중 오류 발생: {str(e)}")
            logger.exception("텍스트 전처리 중 예외 발생")
        
        self.wait_for_user()
    
    def tokenize_settings_menu(self) -> None:
        """토큰화 설정 메뉴"""
        self.print_header("토큰화 설정")
        
        print("텍스트 토큰화 설정을 변경합니다.\n")
        
        # 현재 설정 표시
        print("현재 설정:")
        for param, value in self.tokenize_params.items():
            print(f"- {param}: {value}")
        
        print("\n설정 변경:")
        
        # 토크나이저 유형 선택
        tokenizer_options = ["기본 토크나이저", "WordPiece 토크나이저", "BPE 토크나이저"]
        tokenizer_choice = self.show_menu(tokenizer_options, "토크나이저 유형")
        
        if tokenizer_choice == 0:
            self.tokenize_params['tokenizer_type'] = 'basic'
        elif tokenizer_choice == 1:
            self.tokenize_params['tokenizer_type'] = 'wordpiece'
        else:
            self.tokenize_params['tokenizer_type'] = 'bpe'
        
        # 어휘 크기
        self.tokenize_params['vocab_size'] = self.get_numeric_input(
            "어휘 사전 크기", self.tokenize_params['vocab_size'], 
            min_val=100, max_val=100000
        )
        
        # 최소 단어 빈도
        self.tokenize_params['min_frequency'] = self.get_numeric_input(
            "최소 단어 빈도", self.tokenize_params['min_frequency'], 
            min_val=1, max_val=100
        )
        
        # 토크나이저 생성
        try:
            if self.tokenize_params['tokenizer_type'] == 'basic':
                self.state['tokenizer'] = Tokenizer(vocab_size=self.tokenize_params['vocab_size'])
            elif self.tokenize_params['tokenizer_type'] == 'wordpiece':
                self.state['tokenizer'] = WordPieceTokenizer(vocab_size=self.tokenize_params['vocab_size'])
            else:  # 'bpe'
                self.state['tokenizer'] = BPETokenizer(vocab_size=self.tokenize_params['vocab_size'])
            
            self.show_success("토큰화 설정이 변경되었습니다.")
            
        except Exception as e:
            self.show_error(f"토크나이저 생성 중 오류 발생: {str(e)}")
            logger.exception("토크나이저 생성 중 예외 발생")
        
        self.wait_for_user()
    
    def tokenize_text_menu(self) -> None:
        """텍스트 토큰화 메뉴"""
        self.print_header("텍스트 토큰화")
        
        # 전처리된 텍스트 확인
        if not self.state['preprocessed_texts']:
            if not self.state['dataset']:
                self.show_error("텍스트 데이터가 로드되지 않았습니다. 먼저 데이터를 로드하세요.")
                self.wait_for_user()
                return
            
            # 전처리되지 않은 데이터 사용 여부 확인
            use_raw = self.get_yes_no_input(
                "전처리된 텍스트가 없습니다. 원본 텍스트를 사용하시겠습니까?", True
            )
            
            if not use_raw:
                self.show_message("먼저 텍스트 전처리를 수행하세요.")
                self.wait_for_user()
                return
            
            texts_to_tokenize = self.state['dataset']
        else:
            texts_to_tokenize = self.state['preprocessed_texts']
        
        # 토크나이저 확인
        if not self.state['tokenizer']:
            self.show_message("토크나이저가 설정되지 않았습니다. 기본 설정으로 생성합니다.")
            self.state['tokenizer'] = Tokenizer(vocab_size=self.tokenize_params['vocab_size'])
        
        try:
            # 어휘 사전 구축
            self.show_message(f"어휘 사전 구축 시작: {len(texts_to_tokenize)}개 텍스트")
            
            self.state['tokenizer'].train_from_texts(
                texts=texts_to_tokenize,
                min_frequency=self.tokenize_params['min_frequency']
            )
            
            vocab_size = len(self.state['tokenizer'].vocab)
            self.show_message(f"어휘 사전 구축 완료: {vocab_size}개 토큰")
            
            # 토큰화 진행
            self.show_message("텍스트 토큰화 시작...")
            
            token_sequences = []
            for i, text in enumerate(texts_to_tokenize):
                # 진행상황 표시 (10% 단위)
                if i % max(1, len(texts_to_tokenize) // 10) == 0:
                    progress = i / len(texts_to_tokenize) * 100
                    self.show_message(f"진행 중: {progress:.0f}% ({i}/{len(texts_to_tokenize)})")
                
                # 텍스트 토큰화
                tokens = self.state['tokenizer'].tokenize(text)
                token_sequences.append(tokens)
            
            # 인코딩 진행
            self.show_message("시퀀스 인코딩 시작...")
            
            encoded_sequences = []
            for i, tokens in enumerate(token_sequences):
                # 진행상황 표시 (10% 단위)
                if i % max(1, len(token_sequences) // 10) == 0:
                    progress = i / len(token_sequences) * 100
                    self.show_message(f"진행 중: {progress:.0f}% ({i}/{len(token_sequences)})")
                
                # 토큰 시퀀스를 ID로 인코딩
                token_ids = self.state['tokenizer'].encode(tokens, add_special_tokens=True)
                encoded_sequences.append(token_ids)
            
            # 상태 업데이트
            self.state['tokenized_data'] = {
                'token_sequences': token_sequences,
                'encoded_sequences': encoded_sequences
            }
            
            self.show_success(f"토큰화 완료: {len(token_sequences)}개 시퀀스")
            
            # 토큰화 결과 미리보기
            preview_count = min(5, len(token_sequences))
            print("\n토큰화 결과 미리보기:")
            for i in range(preview_count):
                print(f"\n원본: {texts_to_tokenize[i][:100]}...")
                token_preview = ' '.join(token_sequences[i][:20])
                print(f"토큰: {token_preview}...")
                id_preview = str(encoded_sequences[i][:20])
                print(f"인코딩: {id_preview}...")
            
            # 결과 저장 여부 확인
            save_result = self.get_yes_no_input("\n토큰화 결과를 파일로 저장하시겠습니까?")
            if save_result:
                output_dir = self.get_input(
                    "저장 디렉토리 경로", 
                    self.paths['output_dir']
                )
                
                os.makedirs(output_dir, exist_ok=True)
                
                # 토큰 시퀀스 저장
                tokens_path = os.path.join(output_dir, "token_sequences.json")
                with open(tokens_path, 'w', encoding='utf-8') as f:
                    # 리스트의 리스트를 JSON으로 직렬화
                    json.dump(token_sequences, f, ensure_ascii=False, indent=2)
                
                # 인코딩 시퀀스 저장
                encodings_path = os.path.join(output_dir, "encoded_sequences.json")
                with open(encodings_path, 'w', encoding='utf-8') as f:
                    json.dump(encoded_sequences, f, ensure_ascii=False, indent=2)
                
                self.show_success(f"토큰화 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
        
        except Exception as e:
            self.show_error(f"텍스트 토큰화 중 오류 발생: {str(e)}")
            logger.exception("텍스트 토큰화 중 예외 발생")
        
        self.wait_for_user()
    
    def split_data_menu(self) -> None:
        """데이터 분할 메뉴"""
        self.print_header("데이터 분할")
        
        # 토큰화된 데이터 확인
        if not self.state['tokenized_data']:
            self.show_error("토큰화된 데이터가 없습니다. 먼저 텍스트 토큰화를 수행하세요.")
            self.wait_for_user()
            return
        
        try:
            # 분할 비율 설정
            print("데이터 분할 비율 설정:")
            
            while True:
                train_ratio = self.get_numeric_input("학습 데이터 비율", 0.7, min_val=0.1, max_val=0.9)
                val_ratio = self.get_numeric_input("검증 데이터 비율", 0.15, min_val=0.0, max_val=0.5)
                test_ratio = self.get_numeric_input("테스트 데이터 비율", 0.15, min_val=0.0, max_val=0.5)
                
                # 합이 1이 되는지 확인
                total = train_ratio + val_ratio + test_ratio
                if abs(total - 1.0) < 0.001:  # 부동소수점 오차 허용
                    break
                else:
                    self.show_error(f"비율의 합은 1이어야 합니다. 현재 합: {total:.2f}")
            
            # 셔플 여부 확인
            shuffle = self.get_yes_no_input("데이터를 셔플하시겠습니까?", True)
            
            # 데이터 준비
            encoded_sequences = self.state['tokenized_data']['encoded_sequences']
            
            # 레이블 데이터 확인
            if hasattr(self.state, 'labels') and self.state['labels'] is not None:
                # 레이블이 있는 경우 함께 분할
                labels = self.state['labels']
                
                # 데이터-레이블 길이 확인
                if len(encoded_sequences) != len(labels):
                    self.show_error(f"데이터와 레이블 길이가 일치하지 않습니다: {len(encoded_sequences)} vs {len(labels)}")
                    self.wait_for_user()
                    return
                
                # 분할 수행
                split_result = split_dataset(
                    data=np.array(encoded_sequences, dtype=object),
                    labels=np.array(labels),
                    train_ratio=train_ratio,
                    valid_ratio=val_ratio,
                    test_ratio=test_ratio,
                    shuffle=shuffle
                )
                
                # 결과 추출
                train_data = split_result['train_data'].tolist()
                val_data = split_result['valid_data'].tolist()
                test_data = split_result['test_data'].tolist()
                
                train_labels = split_result['train_labels'].tolist()
                val_labels = split_result['valid_labels'].tolist()
                test_labels = split_result['test_labels'].tolist()
                
                # 상태 업데이트
                self.state['train_data'] = train_data
                self.state['val_data'] = val_data
                self.state['test_data'] = test_data
                
                self.state['train_labels'] = train_labels
                self.state['val_labels'] = val_labels
                self.state['test_labels'] = test_labels
                
                self.show_success(
                    f"데이터 분할 완료: 학습={len(train_data)}개, "
                    f"검증={len(val_data)}개, 테스트={len(test_data)}개"
                )
                
            else:
                # 레이블이 없는 경우 데이터만 분할
                split_result = split_dataset(
                    data=np.array(encoded_sequences, dtype=object),
                    train_ratio=train_ratio,
                    valid_ratio=val_ratio,
                    test_ratio=test_ratio,
                    shuffle=shuffle
                )
                
                # 결과 추출
                train_data = split_result['train_data'].tolist()
                val_data = split_result['valid_data'].tolist()
                test_data = split_result['test_data'].tolist()
                
                # 상태 업데이트
                self.state['train_data'] = train_data
                self.state['val_data'] = val_data
                self.state['test_data'] = test_data
                
                self.show_success(
                    f"데이터 분할 완료: 학습={len(train_data)}개, "
                    f"검증={len(val_data)}개, 테스트={len(test_data)}개"
                )
            
            # 결과 저장 여부 확인
            save_result = self.get_yes_no_input("\n분할 결과를 파일로 저장하시겠습니까?")
            if save_result:
                output_dir = self.get_input(
                    "저장 디렉토리 경로", 
                    self.paths['output_dir']
                )
                
                os.makedirs(output_dir, exist_ok=True)
                
                # 데이터 저장
                train_path = os.path.join(output_dir, "train_data.json")
                with open(train_path, 'w', encoding='utf-8') as f:
                    json.dump(train_data, f, ensure_ascii=False, indent=2)
                
                val_path = os.path.join(output_dir, "val_data.json")
                with open(val_path, 'w', encoding='utf-8') as f:
                    json.dump(val_data, f, ensure_ascii=False, indent=2)
                
                test_path = os.path.join(output_dir, "test_data.json")
                with open(test_path, 'w', encoding='utf-8') as f:
                    json.dump(test_data, f, ensure_ascii=False, indent=2)
                
                # 레이블 저장
                if hasattr(self.state, 'train_labels') and self.state['train_labels'] is not None:
                    train_labels_path = os.path.join(output_dir, "train_labels.json")
                    with open(train_labels_path, 'w', encoding='utf-8') as f:
                        json.dump(train_labels, f, ensure_ascii=False, indent=2)
                    
                    val_labels_path = os.path.join(output_dir, "val_labels.json")
                    with open(val_labels_path, 'w', encoding='utf-8') as f:
                        json.dump(val_labels, f, ensure_ascii=False, indent=2)
                    
                    test_labels_path = os.path.join(output_dir, "test_labels.json")
                    with open(test_labels_path, 'w', encoding='utf-8') as f:
                        json.dump(test_labels, f, ensure_ascii=False, indent=2)
                
                self.show_success(f"분할 결과가 '{output_dir}' 디렉토리에 저장되었습니다.")
        
        except Exception as e:
            self.show_error(f"데이터 분할 중 오류 발생: {str(e)}")
            logger.exception("데이터 분할 중 예외 발생")
        
        self.wait_for_user()
    
    def visualize_data_menu(self) -> None:
        """데이터 시각화 메뉴"""
        self.print_header("데이터 시각화")
        
        # 데이터 확인
        if not self.state['dataset']:
            self.show_error("텍스트 데이터가 로드되지 않았습니다. 먼저 데이터를 로드하세요.")
            self.wait_for_user()
            return
        
        try:
            # 필요한 패키지 임포트
            import matplotlib.pyplot as plt
            from collections import Counter
            import numpy as np
            import pandas as pd
            
            # 시각화 유형 선택
            viz_options = [
                "텍스트 길이 분포",
                "단어 빈도 분포",
                "토큰 길이 분포",
                "가장 빈번한 단어/토큰",
                "단어 길이 분포"
            ]
            
            viz_choice = self.show_menu(viz_options, "시각화 유형")
            
            if viz_choice == 0:  # 텍스트 길이 분포
                # 텍스트 길이 계산
                if self.state['preprocessed_texts']:
                    text_lengths = [len(text.split()) for text in self.state['preprocessed_texts']]
                    title = "전처리된 텍스트 길이 분포 (단어 수)"
                else:
                    text_lengths = [len(text.split()) for text in self.state['dataset']]
                    title = "원본 텍스트 길이 분포 (단어 수)"
                
                # 히스토그램 생성
                plt.figure(figsize=(10, 6))
                plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue')
                plt.xlabel('텍스트 길이 (단어 수)')
                plt.ylabel('빈도')
                plt.title(title)
                plt.grid(True, alpha=0.3)
                
                # 주요 통계 정보
                mean_length = np.mean(text_lengths)
                median_length = np.median(text_lengths)
                
                plt.axvline(mean_length, color='red', linestyle='--', label=f'평균: {mean_length:.1f}')
                plt.axvline(median_length, color='green', linestyle='--', label=f'중앙값: {median_length:.1f}')
                plt.legend()
                
                # 길이 통계 출력
                print(f"\n텍스트 길이 통계:")
                print(f"- 최소: {min(text_lengths)}")
                print(f"- 최대: {max(text_lengths)}")
                print(f"- 평균: {mean_length:.2f}")
                print(f"- 중앙값: {median_length:.2f}")
                
                # 저장 경로
                output_path = os.path.join(self.paths['plot_dir'], "text_length_distribution.png")
                
            elif viz_choice == 1:  # 단어 빈도 분포
                # 데이터 선택
                if self.state['preprocessed_texts']:
                    texts = self.state['preprocessed_texts']
                    title = "전처리된 텍스트 단어 빈도 분포"
                else:
                    texts = self.state['dataset']
                    title = "원본 텍스트 단어 빈도 분포"
                
                # 모든 단어 추출 및 카운트
                all_words = []
                for text in texts:
                    words = text.split()
                    all_words.extend(words)
                
                word_counts = Counter(all_words)
                
                # 가장 빈번한 단어 추출 (상위 30개)
                top_words = word_counts.most_common(30)
                words, counts = zip(*top_words)
                
                # 막대 그래프 생성
                plt.figure(figsize=(12, 8))
                plt.bar(range(len(words)), counts, align='center', alpha=0.7, color='skyblue')
                plt.xticks(range(len(words)), words, rotation=45, ha='right')
                plt.xlabel('단어')
                plt.ylabel('빈도')
                plt.title(title)
                plt.tight_layout()
                
                # 단어 빈도 통계 출력
                print(f"\n단어 빈도 통계:")
                print(f"- 총 단어 수: {len(all_words)}")
                print(f"- 고유 단어 수: {len(word_counts)}")
                print(f"- 가장 빈번한 단어: {top_words[:10]}")
                
                # 저장 경로
                output_path = os.path.join(self.paths['plot_dir'], "word_frequency_distribution.png")
                
            elif viz_choice == 2:  # 토큰 길이 분포
                # 토큰화된 데이터 확인
                if not self.state['tokenized_data']:
                    self.show_error("토큰화된 데이터가 없습니다. 먼저 텍스트 토큰화를 수행하세요.")
                    self.wait_for_user()
                    return
                
                # 토큰 시퀀스 길이 계산
                token_lengths = [len(seq) for seq in self.state['tokenized_data']['token_sequences']]
                
                # 히스토그램 생성
                plt.figure(figsize=(10, 6))
                plt.hist(token_lengths, bins=50, alpha=0.7, color='skyblue')
                plt.xlabel('토큰 시퀀스 길이')
                plt.ylabel('빈도')
                plt.title('토큰 시퀀스 길이 분포')
                plt.grid(True, alpha=0.3)
                
                # 주요 통계 정보
                mean_length = np.mean(token_lengths)
                median_length = np.median(token_lengths)
                
                plt.axvline(mean_length, color='red', linestyle='--', label=f'평균: {mean_length:.1f}')
                plt.axvline(median_length, color='green', linestyle='--', label=f'중앙값: {median_length:.1f}')
                plt.legend()
                
                # 길이 통계 출력
                print(f"\n토큰 시퀀스 길이 통계:")
                print(f"- 최소: {min(token_lengths)}")
                print(f"- 최대: {max(token_lengths)}")
                print(f"- 평균: {mean_length:.2f}")
                print(f"- 중앙값: {median_length:.2f}")
                
                # 저장 경로
                output_path = os.path.join(self.paths['plot_dir'], "token_length_distribution.png")
                
            elif viz_choice == 3:  # 가장 빈번한 단어/토큰
                # 토크나이저 확인
                if not self.state['tokenizer'] or 'vocab' not in dir(self.state['tokenizer']):
                    self.show_error("어휘 사전이 구축된 토크나이저가 없습니다. 먼저 토크나이저를 학습하세요.")
                    self.wait_for_user()
                    return
                
                # 단어-빈도 사전 추출
                vocab = self.state['tokenizer'].vocab
                inv_vocab = self.state['tokenizer'].inv_vocab
                
                # 특수 토큰 제외
                special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
                token_freqs = []
                
                for token, idx in vocab.items():
                    if token not in special_tokens:
                        token_freqs.append((token, idx))
                
                # 인덱스 기준 정렬 (빈도 순)
                token_freqs.sort(key=lambda x: x[1])
                
                # 상위 토큰 추출 (인덱스가 낮을수록 빈도가 높음)
                top_tokens = token_freqs[:50]
                tokens, indices = zip(*top_tokens)
                
                # 막대 그래프 생성
                plt.figure(figsize=(12, 8))
                plt.bar(range(len(tokens)), [1.0/(idx+1) for idx in indices], align='center', alpha=0.7, color='skyblue')
                plt.xticks(range(len(tokens)), tokens, rotation=45, ha='right')
                plt.xlabel('토큰')
                plt.ylabel('상대 빈도')
                plt.title('가장 빈번한 토큰')
                plt.tight_layout()
                
                # 토큰 통계 출력
                print(f"\n어휘 사전 통계:")
                print(f"- 총 토큰 수: {len(vocab)}")
                print(f"- 가장 빈번한 토큰: {tokens[:10]}")
                
                # 저장 경로
                output_path = os.path.join(self.paths['plot_dir'], "token_frequency.png")
                
            elif viz_choice == 4:  # 단어 길이 분포
                # 데이터 선택
                if self.state['preprocessed_texts']:
                    texts = self.state['preprocessed_texts']
                    title = "전처리된 텍스트 단어 길이 분포"
                else:
                    texts = self.state['dataset']
                    title = "원본 텍스트 단어 길이 분포"
                
                # 모든 단어 추출
                all_words = []
                for text in texts:
                    words = text.split()
                    all_words.extend(words)
                
                # 단어 길이 계산
                word_lengths = [len(word) for word in all_words]
                
                # 히스토그램 생성
                plt.figure(figsize=(10, 6))
                plt.hist(word_lengths, bins=20, alpha=0.7, color='skyblue')
                plt.xlabel('단어 길이 (문자 수)')
                plt.ylabel('빈도')
                plt.title(title)
                plt.grid(True, alpha=0.3)
                
                # 주요 통계 정보
                mean_length = np.mean(word_lengths)
                median_length = np.median(word_lengths)
                
                plt.axvline(mean_length, color='red', linestyle='--', label=f'평균: {mean_length:.1f}')
                plt.axvline(median_length, color='green', linestyle='--', label=f'중앙값: {median_length:.1f}')
                plt.legend()
                
                # 길이 통계 출력
                print(f"\n단어 길이 통계:")
                print(f"- 최소: {min(word_lengths)} 문자")
                print(f"- 최대: {max(word_lengths)} 문자")
                print(f"- 평균: {mean_length:.2f} 문자")
                print(f"- 중앙값: {median_length:.2f} 문자")
                
                # 저장 경로
                output_path = os.path.join(self.paths['plot_dir'], "word_length_distribution.png")
            
            else:
                self.show_error("알 수 없는 시각화 유형입니다.")
                self.wait_for_user()
                return
            
            # 시각화 결과 저장
            os.makedirs(self.paths['plot_dir'], exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            self.show_success(f"시각화 결과가 '{output_path}'에 저장되었습니다.")
            
            # 그래프 표시 (옵션)
            show_graph = self.get_yes_no_input("\n그래프를 표시하시겠습니까?", False)
            if show_graph:
                plt.show()
            else:
                plt.close()
        
        except Exception as e:
            self.show_error(f"데이터 시각화 중 오류 발생: {str(e)}")
            logger.exception("데이터 시각화 중 예외 발생")
        
        self.wait_for_user()
    
    def tokenizer_save_load_menu() -> None:
        """토크나이저 저장/로드 메뉴"""
        print_header("토크나이저 저장/로드")
    
        # 메뉴 옵션
        options = ["토크나이저 저장", "토크나이저 로드", "뒤로 가기"]
    
        while True:
            print("토크나이저를 저장하거나 로드합니다.\n")
        
            # 현재 상태 출력
            if STATE['tokenizer'] is not None:
                tokenizer_type = STATE['tokenizer_type'].upper()
                vocab_size = len(STATE['tokenizer'].vocab) if hasattr(STATE['tokenizer'], 'vocab') else "알 수 없음"
                print(f"현재 토크나이저: {tokenizer_type} (어휘 크기: {vocab_size})")
            else:
                print("현재 토크나이저: 없음")
            
            if STATE['current_tokenizer_path'] is not None:
                print(f"저장된 토크나이저 경로: {STATE['current_tokenizer_path']}")

            # 옵션 선택
            print("\n옵션을 선택하세요:")
            for i, option in enumerate(options):
                print(f"{i+1}. {option}")
            
            choice = get_input("\n선택", "3")
        
            if choice == "1":
                save_tokenizer()
            elif choice == "2":
                load_tokenizer()
            elif choice == "3":
                break
            else:
                print("잘못된 선택입니다. 다시 시도하세요.")
        
        print("\n")

def save_tokenizer() -> None:
    """토크나이저 저장 기능"""
    # 토크나이저 확인
    if STATE['tokenizer'] is None:
        print("❌ 오류: 저장할 토크나이저가 없습니다. 먼저 토크나이저를 초기화하세요.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 저장 디렉토리 확인
    tokenizer_dir = DEFAULT_DIRS['tokenizer_dir']
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    # 파일명 설정
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    tokenizer_type = STATE['tokenizer_type']
    vocab_size = STATE['tokenizer_params']['vocab_size']
    default_filename = f"{tokenizer_type}_tokenizer_{vocab_size}_{timestamp}.pkl"
    
    tokenizer_path = get_input("저장할 파일 경로", os.path.join(tokenizer_dir, default_filename))
    
    try:
        # 토크나이저 저장
        if hasattr(STATE['tokenizer'], 'save_to_file'):
            # 내장 저장 메서드가 있는 경우
            STATE['tokenizer'].save_to_file(tokenizer_path)
        else:
            # pickle로 직접 저장
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(STATE['tokenizer'], f)
        
        # 상태 업데이트
        STATE['current_tokenizer_path'] = tokenizer_path
        
        print(f"\n✅ 토크나이저가 '{tokenizer_path}'에 저장되었습니다.")
        
        # 메타데이터 저장 (선택적)
        save_metadata = get_yes_no_input("토크나이저 메타데이터도 저장하시겠습니까?")
        if save_metadata:
            metadata_path = tokenizer_path.replace('.pkl', '_metadata.json')
            metadata = {
                'tokenizer_type': STATE['tokenizer_type'],
                'vocab_size': STATE['tokenizer_params']['vocab_size'],
                'min_frequency': STATE['tokenizer_params']['min_frequency'],
                'max_sequence_length': STATE['tokenizer_params']['max_sequence_length'],
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'vocab_count': len(STATE['tokenizer'].vocab) if hasattr(STATE['tokenizer'], 'vocab') else "unknown"
            }
            
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
            
            print(f"메타데이터가 '{metadata_path}'에 저장되었습니다.")
        
    except Exception as e:
        print(f"\n❌ 오류: 토크나이저 저장 중 예외가 발생했습니다: {str(e)}")
        logger.exception("토크나이저 저장 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def load_tokenizer() -> None:
    """토크나이저 로드 기능"""
    # 저장 디렉토리 확인
    tokenizer_dir = DEFAULT_DIRS['tokenizer_dir']
    
    if not os.path.exists(tokenizer_dir):
        print(f"❌ 오류: 토크나이저 디렉토리 '{tokenizer_dir}'가 존재하지 않습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 토크나이저 파일 목록
    tokenizer_files = [f for f in os.listdir(tokenizer_dir) if f.endswith('.pkl') and 'tokenizer' in f]
    
    if not tokenizer_files:
        print(f"❌ 오류: '{tokenizer_dir}' 디렉토리에 토크나이저 파일이 없습니다.")
        input("\n계속하려면 Enter 키를 누르세요...")
        return
    
    # 파일 목록 출력
    print("\n저장된 토크나이저 목록:")
    tokenizer_files.sort(reverse=True)  # 최신 파일 먼저
    
    for i, file in enumerate(tokenizer_files):
        # 메타데이터 파일이 있으면 추가 정보 표시
        metadata_path = os.path.join(tokenizer_dir, file.replace('.pkl', '_metadata.json'))
        info = ""
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                info = f"(타입: {metadata.get('tokenizer_type', '알 수 없음').upper()}, " \
                       f"어휘 크기: {metadata.get('vocab_count', '알 수 없음')})"
            except:
                pass
        
        # 파일 크기 및 수정 날짜
        file_path = os.path.join(tokenizer_dir, file)
        file_size = os.path.getsize(file_path) // 1024  # KB
        mod_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file_path)))
        
        print(f"{i+1}. {file} {info} ({file_size}KB, {mod_time})")
    
    # 파일 선택
    while True:
        try:
            choice = get_input("\n로드할 토크나이저 번호", "1")
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(tokenizer_files):
                selected_file = tokenizer_files[choice_idx]
                break
            else:
                print(f"유효한 번호를 입력하세요 (1-{len(tokenizer_files)})")
        except ValueError:
            print("숫자를 입력하세요")
    
    # 토크나이저 타입 결정
    selected_path = os.path.join(tokenizer_dir, selected_file)
    
    # 메타데이터에서 타입 정보 추출 시도
    tokenizer_type = 'basic'  # 기본값
    metadata_path = selected_path.replace('.pkl', '_metadata.json')
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            tokenizer_type = metadata.get('tokenizer_type', 'basic')
        except:
            pass
    
    try:
        # 토크나이저 로드 방법 결정
        if 'wordpiece' in selected_file:
            tokenizer_type = 'wordpiece'
        elif 'bpe' in selected_file:
            tokenizer_type = 'bpe'
        
        # 토크나이저 타입에 따라 로드
        if hasattr(Tokenizer, 'load_from_file'):
            # 내장 로드 메서드가 있는 경우
            if tokenizer_type == 'wordpiece':
                tokenizer = WordPieceTokenizer.load_from_file(selected_path)
            elif tokenizer_type == 'bpe':
                tokenizer = BPETokenizer.load_from_file(selected_path)
            else:
                tokenizer = Tokenizer.load_from_file(selected_path)
        else:
            # pickle로 직접 로드
            with open(selected_path, 'rb') as f:
                tokenizer = pickle.load(f)
        
        # 상태 업데이트
        STATE['tokenizer'] = tokenizer
        STATE['tokenizer_type'] = tokenizer_type
        STATE['current_tokenizer_path'] = selected_path
        
        print(f"\n✅ 토크나이저가 '{selected_path}'에서 로드되었습니다.")
        
        # 토크나이저 정보 출력
        vocab_size = len(tokenizer.vocab) if hasattr(tokenizer, 'vocab') else "알 수 없음"
        print(f"토크나이저 타입: {tokenizer_type.upper()}")
        print(f"어휘 크기: {vocab_size}")
        
    except Exception as e:
        print(f"\n❌ 오류: 토크나이저 로드 중 예외가 발생했습니다: {str(e)}")
        logger.exception("토크나이저 로드 중 예외 발생")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def system_config_menu() -> None:
    """시스템 설정 메뉴"""
    print_header("시스템 설정")
    
    print("시스템 설정을 변경합니다.\n")
    
    # 디렉토리 설정
    print("디렉토리 설정:")
    for dir_name, dir_path in DEFAULT_DIRS.items():
        new_path = get_input(f"{dir_name} 디렉토리", dir_path)
        DEFAULT_DIRS[dir_name] = new_path
        os.makedirs(new_path, exist_ok=True)
    
    # 로깅 레벨 설정
    print("\n로깅 레벨 설정:")
    log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    print("옵션을 선택하세요:")
    for i, level in enumerate(log_levels):
        print(f"{i+1}. {level}")
    
    try:
        choice = get_input("\n로깅 레벨", "2")  # INFO가 기본값
        choice_idx = int(choice) - 1
        
        if 0 <= choice_idx < len(log_levels):
            selected_level = log_levels[choice_idx]
            
            # 로깅 레벨 설정
            numeric_level = getattr(logging, selected_level)
            logger.setLevel(numeric_level)
            
            # 모든 핸들러의 레벨 설정
            for handler in logger.handlers:
                handler.setLevel(numeric_level)
            
            print(f"\n✅ 로깅 레벨이 '{selected_level}'로 설정되었습니다.")
        else:
            print(f"유효한 번호를 입력하세요 (1-{len(log_levels)})")
    except ValueError:
        print("숫자를 입력하세요")
    
    print(f"\n✅ 시스템 설정이 변경되었습니다.")
    for dir_name, dir_path in DEFAULT_DIRS.items():
        print(f"- {dir_name} 디렉토리: {dir_path}")
    
    input("\n계속하려면 Enter 키를 누르세요...")

def main_menu() -> None:
    """메인 메뉴 표시"""
    menu_options = [
        "데이터 로드",
        "전처리 설정",
        "텍스트 전처리",
        "토큰화 설정",
        "텍스트 토큰화",
        "데이터 분할",
        "데이터 시각화",
        "토크나이저 저장/로드",
        "시스템 설정",
        "종료"
    ]
    
    while True:
        print_header("텍스트 데이터 처리 시스템")
        print("텍스트 데이터 처리 시스템에 오신 것을 환영합니다.")
        print("아래 메뉴에서 원하는 작업을 선택하세요.\n")
        
        for i, option in enumerate(menu_options):
            print(f"{i+1}. {option}")
        
        print_status()
        
        choice = get_input("\n메뉴 선택", "10")  # 기본값은 종료
        
        if choice == "1":
            load_data_menu()
        elif choice == "2":
            preprocessing_settings_menu()
        elif choice == "3":
            preprocess_text_menu()
        elif choice == "4":
            tokenization_settings_menu()
        elif choice == "5":
            tokenize_text_menu()
        elif choice == "6":
            split_data_menu()
        elif choice == "7":
            visualize_data_menu()
        elif choice == "8":
            tokenizer_save_load_menu()
        elif choice == "9":
            system_config_menu()
        elif choice == "10":
            print("\n프로그램을 종료합니다. 감사합니다!")
            break
        else:
            print("\n유효하지 않은 선택입니다. 다시 시도하세요.")
            input("계속하려면 Enter 키를 누르세요...")

def main():
    """메인 함수"""
    try:
        # 필요한 디렉토리 생성
        ensure_dirs()
        
        # 로깅 설정
        log_dir = os.path.join(project_root, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # 메인 메뉴 실행
        main_menu()
        
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
        logger.exception("예상치 못한 오류 발생")
        sys.exit(1)

if __name__ == "__main__":
    main()