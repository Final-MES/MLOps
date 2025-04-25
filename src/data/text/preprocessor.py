"""
텍스트 데이터 전처리 모듈

이 모듈은 텍스트 데이터 전처리를 위한 기능을 제공합니다:
- 텍스트 정제 (특수 문자 제거, 소문자 변환 등)
- 토큰화 (단어, 문장, 문자)
- 불용어 제거
- 어간 추출 / 표제어 추출
- 인코딩 (원-핫, 정수 시퀀스 등)
"""

import re
import string
import logging
import os
import pickle
from typing import List, Dict, Tuple, Any, Optional, Union, Callable, Set
import numpy as np
from collections import Counter

# 로깅 설정
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """텍스트 데이터 전처리를 위한 클래스"""
    
    def __init__(self, 
                 lowercase: bool = True,
                 remove_punctuation: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = True,
                 stemming: bool = False,
                 lemmatization: bool = False,
                 max_sequence_length: Optional[int] = None,
                 vocabulary_size: Optional[int] = None):
        """
        텍스트 전처리기 초기화
        
        Args:
            lowercase: 소문자 변환 여부
            remove_punctuation: 구두점 제거 여부
            remove_numbers: 숫자 제거 여부
            remove_stopwords: 불용어 제거 여부
            stemming: 어간 추출 여부
            lemmatization: 표제어 추출 여부
            max_sequence_length: 최대 시퀀스 길이
            vocabulary_size: 어휘 사전 크기
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.stemming = stemming
        self.lemmatization = lemmatization
        self.max_sequence_length = max_sequence_length
        self.vocabulary_size = vocabulary_size
        
        # 불용어 목록 (기본)
        self.stopwords = set([
            'a', 'an', 'the', 'and', 'but', 'or', 'if', 'because', 'as', 'what',
            'which', 'this', 'that', 'these', 'those', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'to', 'from', 'in', 'out', 'on', 'off', 'over', 'under', 'of', 'at',
            'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through',
            'during', 'before', 'after', 'above', 'below', 'up', 'down'
        ])
        
        # 어휘 사전 (학습 시 생성)
        self.word_index = {}  # 단어 -> 인덱스
        self.index_word = {}  # 인덱스 -> 단어
        
        # 어간 추출기, 표제어 추출기 (필요한 경우만 초기화)
        self.stemmer = None
        self.lemmatizer = None
        
        if self.stemming:
            try:
                from nltk.stem import PorterStemmer
                self.stemmer = PorterStemmer()
            except ImportError:
                logger.warning("NLTK가 설치되지 않았습니다. 어간 추출이 비활성화됩니다.")
                self.stemming = False
        
        if self.lemmatization:
            try:
                from nltk.stem import WordNetLemmatizer
                self.lemmatizer = WordNetLemmatizer()
            except ImportError:
                logger.warning("NLTK가 설치되지 않았습니다. 표제어 추출이 비활성화됩니다.")
                self.lemmatization = False
    
    def load_stopwords(self, stopwords_file: str = None, language: str = 'english') -> None:
        """
        불용어 로드
        
        Args:
            stopwords_file: 불용어 파일 경로 (없으면 NLTK 불용어 사용)
            language: 불용어 언어 (NLTK 사용 시)
        """
        if stopwords_file and os.path.exists(stopwords_file):
            # 파일에서 불용어 로드
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                self.stopwords = set(line.strip() for line in f if line.strip())
            logger.info(f"불용어 파일에서 {len(self.stopwords)}개 불용어 로드 완료")
        else:
            # NLTK 불용어 사용
            try:
                from nltk.corpus import stopwords
                try:
                    import nltk
                    nltk.download('stopwords', quiet=True)
                    self.stopwords = set(stopwords.words(language))
                    logger.info(f"NLTK에서 {len(self.stopwords)}개 불용어 로드 완료")
                except:
                    logger.warning("NLTK 불용어를 다운로드할 수 없습니다. 기본 불용어를 사용합니다.")
            except ImportError:
                logger.warning("NLTK가 설치되지 않았습니다. 기본 불용어를 사용합니다.")
    
    def clean_text(self, text: str) -> str:
        """
        텍스트 정제
        
        Args:
            text: 입력 텍스트
            
        Returns:
            str: 정제된 텍스트
        """
        # 소문자 변환
        if self.lowercase:
            text = text.lower()
        
        # 구두점 제거
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 숫자 제거
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트 토큰화 (단어 단위)
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[str]: 토큰 목록
        """
        # 텍스트 정제
        cleaned_text = self.clean_text(text)
        
        # 공백 기준 토큰화
        tokens = cleaned_text.split()
        
        # 불용어 제거
        if self.remove_stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]
        
        # 어간 추출
        if self.stemming and self.stemmer:
            tokens = [self.stemmer.stem(token) for token in tokens]
        
        # 표제어 추출
        if self.lemmatization and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        문장 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[str]: 문장 목록
        """
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            return nltk.sent_tokenize(text)
        except ImportError:
            logger.warning("NLTK가 설치되지 않았습니다. 간단한 규칙 기반 문장 토큰화를 사용합니다.")
            # 간단한 규칙 기반 문장 토큰화
            sentences = re.split(r'(?<=[.!?])\s+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def build_vocabulary(self, texts: List[str], min_frequency: int = 1) -> Dict[str, int]:
        """
        어휘 사전 구축
        
        Args:
            texts: 텍스트 목록
            min_frequency: 최소 단어 빈도
            
        Returns:
            Dict[str, int]: 단어-인덱스 매핑
        """
        # 모든 토큰 추출
        all_tokens = []
        for text in texts:
            tokens = self.tokenize(text)
            all_tokens.extend(tokens)
        
        # 단어 빈도 계산
        word_counts = Counter(all_tokens)
        
        # 최소 빈도 필터링
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= min_frequency}
        
        # 어휘 크기 제한
        if self.vocabulary_size is not None and len(word_counts) > self.vocabulary_size:
            word_counts = dict(word_counts.most_common(self.vocabulary_size))
        
        # 단어-인덱스 매핑 생성
        self.word_index = {'<PAD>': 0, '<UNK>': 1}  # 특수 토큰
        for i, word in enumerate(word_counts.keys(), start=2):
            self.word_index[word] = i
        
        # 인덱스-단어 매핑 생성
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        
        logger.info(f"어휘 사전 구축 완료: {len(self.word_index)}개 단어")
        
        return self.word_index
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        텍스트 목록을 정수 시퀀스로 변환
        
        Args:
            texts: 텍스트 목록
            
        Returns:
            List[List[int]]: 정수 시퀀스 목록
        """
        if not self.word_index:
            logger.warning("어휘 사전이 구축되지 않았습니다. 먼저 build_vocabulary를 호출하세요.")
            return []
        
        sequences = []
        for text in texts:
            tokens = self.tokenize(text)
            sequence = [self.word_index.get(token, 1) for token in tokens]  # 1은 <UNK> 토큰
            sequences.append(sequence)
        
        return sequences
    
    def pad_sequences(self, sequences: List[List[int]], max_length: Optional[int] = None, 
                     padding: str = 'post', truncating: str = 'post') -> np.ndarray:
        """
        시퀀스 패딩
        
        Args:
            sequences: 정수 시퀀스 목록
            max_length: 최대 시퀀스 길이 (None이면 설정값 또는 최대 길이 사용)
            padding: 패딩 위치 ('pre' 또는 'post')
            truncating: 자르기 위치 ('pre' 또는 'post')
            
        Returns:
            np.ndarray: 패딩된 시퀀스
        """
        if not max_length:
            max_length = self.max_sequence_length
        
        if not max_length:
            # 최대 길이 자동 계산
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = np.zeros((len(sequences), max_length), dtype=np.int32)
        
        for i, seq in enumerate(sequences):
            if len(seq) > max_length:
                # 시퀀스 자르기
                if truncating == 'pre':
                    seq = seq[-max_length:]
                else:
                    seq = seq[:max_length]
            
            # 패딩 적용
            if padding == 'pre':
                padded_sequences[i, -len(seq):] = seq
            else:
                padded_sequences[i, :len(seq)] = seq
        
        return padded_sequences
    
    def create_one_hot_encoding(self, sequences: List[List[int]], vocab_size: Optional[int] = None) -> np.ndarray:
        """
        원-핫 인코딩 생성
        
        Args:
            sequences: 정수 시퀀스 목록
            vocab_size: 어휘 크기 (None이면 자동 계산)
            
        Returns:
            np.ndarray: 원-핫 인코딩 배열
        """
        # 어휘 크기 결정
        if not vocab_size:
            vocab_size = len(self.word_index)
        
        if not vocab_size:
            vocab_size = max(max(seq) for seq in sequences) + 1
        
        # 패딩
        padded_sequences = self.pad_sequences(sequences)
        
        # 원-핫 인코딩
        one_hot = np.zeros((len(sequences), padded_sequences.shape[1], vocab_size), dtype=np.int32)
        
        for i, seq in enumerate(padded_sequences):
            for j, idx in enumerate(seq):
                if idx < vocab_size:
                    one_hot[i, j, idx] = 1
        
        return one_hot
    
    def create_ngrams(self, text: str, n: int) -> List[str]:
        """
        문자 N-그램 생성
        
        Args:
            text: 입력 텍스트
            n: N-그램 크기
            
        Returns:
            List[str]: N-그램 목록
        """
        text = self.clean_text(text)
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    def create_word_ngrams(self, text: str, n: int) -> List[str]:
        """
        단어 N-그램 생성
        
        Args:
            text: 입력 텍스트
            n: N-그램 크기
            
        Returns:
            List[str]: 단어 N-그램 목록
        """
        tokens = self.tokenize(text)
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def save(self, file_path: str) -> bool:
        """
        전처리기 저장
        
        Args:
            file_path: 저장 파일 경로
            
        Returns:
            bool: 저장 성공 여부
        """
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'lowercase': self.lowercase,
                    'remove_punctuation': self.remove_punctuation,
                    'remove_numbers': self.remove_numbers,
                    'remove_stopwords': self.remove_stopwords,
                    'stemming': self.stemming,
                    'lemmatization': self.lemmatization,
                    'max_sequence_length': self.max_sequence_length,
                    'vocabulary_size': self.vocabulary_size,
                    'stopwords': self.stopwords,
                    'word_index': self.word_index,
                    'index_word': self.index_word
                }, f)
            logger.info(f"전처리기 저장 완료: {file_path}")
            return True
        except Exception as e:
            logger.error(f"전처리기 저장 중 오류 발생: {e}")
            return False
    
    @classmethod
    def load(cls, file_path: str) -> 'TextPreprocessor':
        """
        전처리기 로드
        
        Args:
            file_path: 로드할 파일 경로
            
        Returns:
            TextPreprocessor: 로드된 전처리기
        """
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            
            preprocessor = cls(
                lowercase=data['lowercase'],
                remove_punctuation=data['remove_punctuation'],
                remove_numbers=data['remove_numbers'],
                remove_stopwords=data['remove_stopwords'],
                stemming=data['stemming'],
                lemmatization=data['lemmatization'],
                max_sequence_length=data['max_sequence_length'],
                vocabulary_size=data['vocabulary_size']
            )
            
            preprocessor.stopwords = data['stopwords']
            preprocessor.word_index = data['word_index']
            preprocessor.index_word = data['index_word']
            
            logger.info(f"전처리기 로드 완료: {file_path}")
            return preprocessor
        except Exception as e:
            logger.error(f"전처리기 로드 중 오류 발생: {e}")
            return None