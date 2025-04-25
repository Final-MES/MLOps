"""
텍스트 토큰화 모듈

이 모듈은 고급 토큰화 기능을 제공합니다:
- 서브워드 토큰화 (WordPiece, BPE)
- 문맥 고려 토큰화
- 토큰화 방식 최적화
"""

import re
import os
import json
import logging
import collections
from typing import List, Dict, Tuple, Any, Optional, Union, Set, Iterator
from pathlib import Path

# 로깅 설정
logger = logging.getLogger(__name__)

class Tokenizer:
    """기본 토큰화 클래스"""
    
    def __init__(self, 
                vocab_size: int = 30000, 
                pad_token: str = "[PAD]",
                unk_token: str = "[UNK]",
                cls_token: str = "[CLS]",
                sep_token: str = "[SEP]",
                mask_token: str = "[MASK]"):
        """
        토크나이저 초기화
        
        Args:
            vocab_size: 어휘 크기
            pad_token: 패딩 토큰
            unk_token: 알 수 없는 토큰
            cls_token: 문장 시작 토큰
            sep_token: 문장 구분 토큰
            mask_token: 마스킹 토큰
        """
        self.vocab_size = vocab_size
        
        # 특수 토큰
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        
        # 어휘 사전
        self.vocab = {}
        self.inv_vocab = {}
        
        # 특수 토큰 등록
        self._add_special_tokens()
    
    def _add_special_tokens(self) -> None:
        """특수 토큰을 어휘 사전에 추가"""
        special_tokens = [
            self.pad_token,
            self.unk_token,
            self.cls_token,
            self.sep_token,
            self.mask_token
        ]
        
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inv_vocab[i] = token
    
    def tokenize(self, text: str) -> List[str]:
        """
        텍스트 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[str]: 토큰 목록
        """
        # 기본 토큰화 (공백 기준)
        return text.split()
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        텍스트를 토큰 ID로 인코딩
        
        Args:
            text: 입력 텍스트
            add_special_tokens: 특수 토큰 추가 여부
            
        Returns:
            List[int]: 토큰 ID 목록
        """
        tokens = self.tokenize(text)
        
        # 특수 토큰 추가
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # 토큰 ID 변환
        token_ids = []
        for token in tokens:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self.vocab[self.unk_token])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        """
        토큰 ID를 텍스트로 디코딩
        
        Args:
            token_ids: 토큰 ID 목록
            skip_special_tokens: 특수 토큰 건너뛰기 여부
            
        Returns:
            str: 디코딩된 텍스트
        """
        tokens = []
        
        special_tokens_ids = {
            self.vocab[self.pad_token],
            self.vocab[self.cls_token],
            self.vocab[self.sep_token],
            self.vocab[self.mask_token]
        }
        
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_tokens_ids:
                continue
                
            if token_id in self.inv_vocab:
                tokens.append(self.inv_vocab[token_id])
            else:
                tokens.append(self.unk_token)
        
        # 공백으로 연결
        return " ".join(tokens)
    
    def train_from_texts(self, texts: List[str], min_frequency: int = 2) -> None:
        """
        텍스트 목록으로부터 어휘 사전 구축
        
        Args:
            texts: 텍스트 목록
            min_frequency: 최소 토큰 빈도
        """
        # 토큰 카운트
        token_counts = collections.Counter()
        
        for text in texts:
            tokens = self.tokenize(text)
            token_counts.update(tokens)
        
        # 빈도 필터링
        token_counts = {token: count for token, count in token_counts.items() 
                       if count >= min_frequency}
        
        # 어휘 크기 제한
        token_counts = dict(sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:self.vocab_size - 5])
        
        # 어휘 사전 초기화 (특수 토큰만 유지)
        self.vocab = {token: idx for token, idx in self.vocab.items() 
                     if token in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]}
        
        # 새 토큰 추가
        for token in token_counts:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        
        # 역 매핑 업데이트
        self.inv_vocab = {idx: token for token, idx in self.vocab.items()}
        
        logger.info(f"어휘 사전 구축 완료: {len(self.vocab)}개 토큰")
    
    def save_to_file(self, filepath: str) -> None:
        """
        토크나이저를 파일로 저장
        
        Args:
            filepath: 저장 경로
        """
        save_data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "pad_token": self.pad_token,
            "unk_token": self.unk_token,
            "cls_token": self.cls_token,
            "sep_token": self.sep_token,
            "mask_token": self.mask_token
        }
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # JSON 파일로 저장
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"토크나이저 저장 완료: {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Tokenizer':
        """
        파일에서 토크나이저 로드
        
        Args:
            filepath: 로드할 파일 경로
            
        Returns:
            Tokenizer: 로드된 토크나이저
        """
        # JSON 파일 로드
        with open(filepath, 'r', encoding='utf-8') as f:
            load_data = json.load(f)
        
        # 토크나이저 초기화
        tokenizer = cls(
            vocab_size=load_data["vocab_size"],
            pad_token=load_data["pad_token"],
            unk_token=load_data["unk_token"],
            cls_token=load_data["cls_token"],
            sep_token=load_data["sep_token"],
            mask_token=load_data["mask_token"]
        )
        
        # 어휘 사전 설정
        tokenizer.vocab = load_data["vocab"]
        tokenizer.inv_vocab = {int(idx): token for token, idx in tokenizer.vocab.items()}
        
        logger.info(f"토크나이저 로드 완료: {filepath}")
        
        return tokenizer


class WordPieceTokenizer(Tokenizer):
    """WordPiece 알고리즘을 사용한 서브워드 토크나이저"""
    
    def __init__(self, 
                vocab_size: int = 30000, 
                pad_token: str = "[PAD]",
                unk_token: str = "[UNK]",
                cls_token: str = "[CLS]",
                sep_token: str = "[SEP]",
                mask_token: str = "[MASK]",
                word_token_pattern: str = r'[a-zA-Z0-9]+|[^a-zA-Z0-9\s]',
                subword_prefix: str = "##"):
        """
        WordPiece 토크나이저 초기화
        
        Args:
            vocab_size: 어휘 크기
            pad_token: 패딩 토큰
            unk_token: 알 수 없는 토큰
            cls_token: 문장 시작 토큰
            sep_token: 문장 구분 토큰
            mask_token: 마스킹 토큰
            word_token_pattern: 단어 토큰화 패턴
            subword_prefix: 서브워드 접두사
        """
        super().__init__(
            vocab_size=vocab_size,
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token
        )
        
        self.word_token_pattern = word_token_pattern
        self.subword_prefix = subword_prefix
    
    def tokenize(self, text: str) -> List[str]:
        """
        WordPiece 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[str]: 서브워드 토큰 목록
        """
        # 텍스트 정규화
        text = text.lower().strip()
        
        # 단어 토큰화
        words = re.findall(self.word_token_pattern, text)
        
        tokens = []
        
        for word in words:
            # 어휘 사전에 있는 단어는 그대로 사용
            if word in self.vocab:
                tokens.append(word)
                continue
            
            # 서브워드 토큰화
            subwords = self._tokenize_word(word)
            tokens.extend(subwords)
        
        return tokens
    
    def _tokenize_word(self, word: str) -> List[str]:
        """
        단어를 서브워드로 토큰화
        
        Args:
            word: 입력 단어
            
        Returns:
            List[str]: 서브워드 목록
        """
        # 어휘 사전이 비어있으면 알 수 없는 토큰 반환
        if not self.vocab or len(self.vocab) <= 5:  # 특수 토큰만 있는 경우
            return [self.unk_token]
        
        # 단어가 어휘에 있으면 그대로 반환
        if word in self.vocab:
            return [word]
        
        # 단어 길이가 1이면 알 수 없는 토큰 반환
        if len(word) == 1:
            return [self.unk_token]
        
        # 가능한 모든 분할 시도
        # 첫 번째 서브워드는 접두사 없이
        subwords = []
        start = 0
        while start < len(word):
            end = len(word)
            curr_subword = None
            
            while start < end:
                subword = word[start:end]
                if start > 0:
                    subword = self.subword_prefix + subword
                
                if subword in self.vocab:
                    curr_subword = subword
                    break
                    
                end -= 1
            
            if curr_subword is None:
                # 적절한 서브워드를 찾을 수 없음
                return [self.unk_token]
            
            subwords.append(curr_subword)
            start = end
        
        return subwords
    
    def train_from_texts(self, texts: List[str], min_frequency: int = 2) -> None:
        """
        WordPiece 어휘 사전 구축
        
        Args:
            texts: 텍스트 목록
            min_frequency: 최소 토큰 빈도
        """
        # 모든 텍스트에서 단어 추출
        all_words = []
        for text in texts:
            text = text.lower().strip()
            words = re.findall(self.word_token_pattern, text)
            all_words.extend(words)
        
        # 단어 빈도 계산
        word_counts = collections.Counter(all_words)
        
        # 서브워드 생성 및 빈도 계산
        subword_counts = collections.defaultdict(int)
        
        # 특수 토큰 추가
        for token in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
            subword_counts[token] = len(all_words)  # 높은 빈도로 설정
        
        # 모든 문자를 서브워드로 추가
        for word in all_words:
            for i in range(len(word)):
                if i == 0:
                    subword = word[0]
                    subword_counts[subword] += word_counts[word]
                else:
                    subword = self.subword_prefix + word[i]
                    subword_counts[subword] += word_counts[word]
        
        # 단어 전체를 서브워드로 추가
        for word, count in word_counts.items():
            if count >= min_frequency:
                subword_counts[word] += count
        
        # 가능한 서브워드 조합 생성
        for word, count in word_counts.items():
            if count < min_frequency:
                continue
                
            # 길이 2 이상의 서브워드
            for i in range(len(word)):
                for j in range(i+2, len(word)+1):
                    if i == 0:
                        subword = word[i:j]
                    else:
                        subword = self.subword_prefix + word[i:j]
                    
                    subword_counts[subword] += count
        
        # 빈도 기준 상위 서브워드 선택
        sorted_subwords = sorted(subword_counts.items(), key=lambda x: x[1], reverse=True)
        selected_subwords = sorted_subwords[:self.vocab_size]
        
        # 어휘 사전 구축
        self.vocab = {}
        self.inv_vocab = {}
        
        # 특수 토큰 추가
        self._add_special_tokens()
        
        # 선택된 서브워드 추가
        for subword, _ in selected_subwords:
            if subword not in self.vocab:
                self.vocab[subword] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = subword
        
        logger.info(f"WordPiece 어휘 사전 구축 완료: {len(self.vocab)}개 서브워드")


class BPETokenizer(Tokenizer):
    """BPE(Byte-Pair Encoding) 알고리즘을 사용한 서브워드 토크나이저"""
    
    def __init__(self, 
                vocab_size: int = 30000, 
                pad_token: str = "[PAD]",
                unk_token: str = "[UNK]",
                cls_token: str = "[CLS]",
                sep_token: str = "[SEP]",
                mask_token: str = "[MASK]"):
        """
        BPE 토크나이저 초기화
        
        Args:
            vocab_size: 어휘 크기
            pad_token: 패딩 토큰
            unk_token: 알 수 없는 토큰
            cls_token: 문장 시작 토큰
            sep_token: 문장 구분 토큰
            mask_token: 마스킹 토큰
        """
        super().__init__(
            vocab_size=vocab_size,
            pad_token=pad_token,
            unk_token=unk_token,
            cls_token=cls_token,
            sep_token=sep_token,
            mask_token=mask_token
        )
        
        # BPE 병합 규칙
        self.merges = {}
    
    def tokenize(self, text: str) -> List[str]:
        """
        BPE 토큰화
        
        Args:
            text: 입력 텍스트
            
        Returns:
            List[str]: 서브워드 토큰 목록
        """
        # 텍스트 정규화
        text = text.lower().strip()
        
        # 공백으로 단어 분리
        words = text.split()
        
        tokens = []
        
        for word in words:
            # 어휘 사전에 있는 단어는 그대로 사용
            if word in self.vocab:
                tokens.append(word)
                continue
            
            # 단어를 문자 시퀀스로 변환
            chars = list(word)
            
            # BPE 병합 적용
            while len(chars) > 1:
                pairs = self._get_pairs(chars)
                
                # 병합할 쌍 없음
                if not pairs:
                    break
                
                # 가장 높은 우선순위의 쌍 찾기
                best_pair = None
                best_rank = float('inf')
                
                for pair in pairs:
                    pair_str = pair[0] + pair[1]
                    if pair_str in self.merges:
                        rank = self.merges[pair_str]
                        if rank < best_rank:
                            best_rank = rank
                            best_pair = pair
                
                # 병합할 쌍 없음
                if best_pair is None:
                    break
                
                # 쌍 병합
                chars = self._merge_pair(best_pair[0], best_pair[1], chars)
            
            # 서브워드 추가
            for subword in chars:
                if subword in self.vocab:
                    tokens.append(subword)
                else:
                    tokens.append(self.unk_token)
        
        return tokens
    
    def _get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        """
        연속된 문자 쌍 추출
        
        Args:
            word: 문자 목록
            
        Returns:
            List[Tuple[str, str]]: 문자 쌍 목록
        """
        pairs = []
        prev_char = word[0]
        
        for char in word[1:]:
            pairs.append((prev_char, char))
            prev_char = char
        
        return pairs
    
    def _merge_pair(self, pair1: str, pair2: str, word: List[str]) -> List[str]:
        """
        쌍 병합
        
        Args:
            pair1: 첫 번째 문자
            pair2: 두 번째 문자
            word: 문자 목록
            
        Returns:
            List[str]: 병합된 문자 목록
        """
        merged = pair1 + pair2
        result = []
        i = 0
        
        while i < len(word):
            if i < len(word) - 1 and word[i] == pair1 and word[i+1] == pair2:
                result.append(merged)
                i += 2
            else:
                result.append(word[i])
                i += 1
        
        return result
    
    def train_from_texts(self, texts: List[str], min_frequency: int = 2, 
                        num_merges: int = 10000) -> None:
        """
        BPE 어휘 사전 및 병합 규칙 구축
        
        Args:
            texts: 텍스트 목록
            min_frequency: 최소 토큰 빈도
            num_merges: 병합 횟수
        """
        # 모든 텍스트에서 단어 추출
        word_counts = collections.Counter()
        
        for text in texts:
            text = text.lower().strip()
            words = text.split()
            word_counts.update(words)
        
        # 빈도 필터링
        word_counts = {word: count for word, count in word_counts.items() 
                      if count >= min_frequency}
        
        # 단어를 문자로 분리
        vocab = collections.defaultdict(int)
        
        # 특수 토큰 추가
        for token in [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token]:
            vocab[token] = sum(word_counts.values())  # 높은 빈도로 설정
        
        # 모든 문자 추가
        for word, count in word_counts.items():
            for char in word:
                vocab[char] += count
        
        # 초기 병합 사전
        self.merges = {}
        
        # 단어를 문자 시퀀스로 분리
        splits = {word: list(word) for word in word_counts}
        
        # BPE 알고리즘 실행
        for i in range(num_merges):
            # 모든 쌍의 빈도 계산
            pair_counts = collections.Counter()
            
            for word, chars in splits.items():
                count = word_counts[word]
                for pair in self._get_pairs(chars):
                    pair_counts[pair] += count
            
            # 더 이상 병합할 쌍이 없으면 종료
            if not pair_counts:
                break
            
            # 가장 빈도가 높은 쌍 선택
            best_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
            
            # 병합 규칙 추가
            merged = best_pair[0] + best_pair[1]
            self.merges[merged] = i
            
            # 새로운 토큰 추가
            vocab[merged] += pair_counts[best_pair]
            
            # 모든 단어에서 쌍 병합
            new_splits = {}
            for word, chars in splits.items():
                new_chars = self._merge_pair(best_pair[0], best_pair[1], chars)
                new_splits[word] = new_chars
            
            splits = new_splits
        
        # 빈도 기준 상위 토큰 선택
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        selected_tokens = sorted_vocab[:self.vocab_size]
        
        # 어휘 사전 구축
        self.vocab = {}
        self.inv_vocab = {}
        
        # 특수 토큰 추가
        self._add_special_tokens()
        
        # 선택된 토큰 추가
        for token, _ in selected_tokens:
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = token
        
        logger.info(f"BPE 어휘 사전 구축 완료: {len(self.vocab)}개 토큰, {len(self.merges)}개 병합 규칙")


# 토크나이저 사용 예시
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 기본 토크나이저 테스트
    texts = [
        "Hello world! This is a test.",
        "Natural language processing is fun.",
        "Tokenization is an important step in NLP."
    ]
    
    # 기본 토크나이저
    tokenizer = Tokenizer(vocab_size=1000)
    tokenizer.train_from_texts(texts)
    
    test_text = "Hello world! This is a tokenization test."
    tokens = tokenizer.tokenize(test_text)
    token_ids = tokenizer.encode(test_text)
    decoded_text = tokenizer.decode(token_ids)
    
    print("기본 토크나이저 테스트:")
    print(f"원본 텍스트: {test_text}")
    print(f"토큰: {tokens}")
    print(f"토큰 ID: {token_ids}")
    print(f"디코딩: {decoded_text}")
    
    # WordPiece 토크나이저 테스트
    wp_tokenizer = WordPieceTokenizer(vocab_size=1000)
    wp_tokenizer.train_from_texts(texts)
    
    wp_tokens = wp_tokenizer.tokenize(test_text)
    wp_token_ids = wp_tokenizer.encode(test_text)
    wp_decoded_text = wp_tokenizer.decode(wp_token_ids)
    
    print("\nWordPiece 토크나이저 테스트:")
    print(f"원본 텍스트: {test_text}")
    print(f"토큰: {wp_tokens}")
    print(f"토큰 ID: {wp_token_ids}")
    print(f"디코딩: {wp_decoded_text}")
    
    # BPE 토크나이저 테스트
    bpe_tokenizer = BPETokenizer(vocab_size=1000)
    bpe_tokenizer.train_from_texts(texts, num_merges=50)
    
    bpe_tokens = bpe_tokenizer.tokenize(test_text)
    bpe_token_ids = bpe_tokenizer.encode(test_text)
    bpe_decoded_text = bpe_tokenizer.decode(bpe_token_ids)
    
    print("\nBPE 토크나이저 테스트:")
    print(f"원본 텍스트: {test_text}")
    print(f"토큰: {bpe_tokens}")
    print(f"토큰 ID: {bpe_token_ids}")
    print(f"디코딩: {bpe_decoded_text}")