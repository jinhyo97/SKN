import pandas as pd
import re
import unicodedata
import torch


def remove_accent(sentence):
    '''주어진 문장에서 악센트(발음 기호)를 제거하는 함수'''
    
    # 'NFD' 형식으로 정규화하여 악센트를 분리
    return ''.join(
        char
        for char in unicodedata.normalize('NFD', sentence)  # 문자를 NFD 형식으로 변환
        if unicodedata.category(char) != 'Mn'  # 'Mn' (발음 기호) 범주에 속하지 않는 문자만 선택
    )


def preprocessing(sentence):
    '''문장을 전처리하는 함수'''
    
    # 1. 문장을 소문자로 변환
    sentence = sentence.lower()
    
    # 2. 악센트를 제거
    sentence = remove_accent(sentence)
    
    # 3. 문장 부호(!, ?, .) 앞에 공백 추가
    sentence = re.sub('([!,?.])', r' \1', sentence)
    
    # 4. 여러 개의 공백을 하나의 공백으로 치환
    sentence = re.sub('\s+', ' ', sentence)

    # 전처리된 문장을 반환
    return sentence


def char_to_idx(tokens: list, vocab: dict):
    '''주어진 토큰 리스트를 인덱스 리스트로 변환'''
    return [vocab.get(word) for word in tokens]


def idx_to_char(tokens: torch.Tensor, inverse_vocab: dict):
    '''주어진 인덱스 텐서를 원래의 토큰 리스트로 변환'''
    return [inverse_vocab.get(token.item()) for token in tokens]


def list_to_tensor(tokens: list):
    '''주어진 리스트를 PyTorch 텐서로 변환하고 정수형으로 변환'''
    return torch.Tensor(tokens).long()
