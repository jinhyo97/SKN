from src.data import TranslationDataset, TranslationDataModule
from src.model.seq2seq import Seq2Seq, Encoder, Decoder
from src.training import IMDBModule
from src.utils import (
    preprocessing,
    char_to_idx,
    idx_to_char,
    list_to_tensor,
)

import pandas as pd
import numpy as np
import random
import json
import nni
import itertools
from tqdm import tqdm

from tensorflow.keras.datasets import imdb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

tqdm.pandas()


def main(configs):
    # pandas 라이브러리를 사용하여 'fra.txt' 파일을 읽어온다.
    # 파일은 탭으로 구분되어 있으며, 헤더는 없음.
    data = pd.read_csv('./data/fra.txt', sep='\t', header=None).iloc[:, :2]
    data.columns = ['eng', 'fra']   # 읽어온 데이터의 열 이름을 'eng'와 'fra'로 설정

    # preprocessing
    data.eng = data.eng.progress_apply(lambda x: preprocessing(x))  # 영어 문장에 대해 전처리 함수를 적용하고 진행 상황을 표시
    data.fra = data.fra.progress_apply(lambda x: preprocessing(x))  # 프랑스어 문장에 대해 전처리 함수를 적용하고 진행 상황을 표시

    data['token_eng'] = data.eng.str.split()    # 영어 문장을 공백을 기준으로 분리하여 토큰화한 결과를 새로운 열에 저장
    data['token_fra'] = data.fra.str.split()    # 프랑스어 문장을 공백을 기준으로 분리하여 토큰화한 결과를 새로운 열에 저장

    data.token_fra = data.token_fra.apply(lambda x: ['<SOS>'] + x + ['<EOS>'])  # 프랑스어 토큰 리스트의 시작에 <SOS> (시작 토큰)과 끝에 <EOS> (종료 토큰)을 추가

    source = data.token_eng # 영어 토큰 리스트를 source 변수에 저장
    target = data.token_fra # 프랑스어 토큰 리스트를 target 변수에 저장

    # vocab 만들기
    eng_vocab = list(set(itertools.chain(*source.tolist())))    # 영어 단어 목록을 생성하기 위해 모든 토큰을 1차원으로 만들고 중복 제거
    eng_vocab = ['<PAD>'] + eng_vocab   # <PAD> 토큰을 어휘 목록의 첫 번째 항목으로 추가
    eng_vocab = dict(zip(eng_vocab, range(len(eng_vocab)))) # 각 단어에 고유한 인덱스를 부여하여 사전 생성
    eng_inverse_vocab = {value: key for key, value in eng_vocab.items()}    # 인덱스를 키로, 단어를 값으로 하는 반전된 사전 생성

    # eng token
    data['encoded_token_eng'] = data.token_eng.apply(
        lambda x: char_to_idx(x, eng_vocab)
    )   # 영어 토큰을 인덱스로 변환하여 새로운 열에 저장
    data.encoded_token_eng = data.encoded_token_eng.apply(
        lambda x: list_to_tensor(x)
    )   # 인덱스 리스트를 PyTorch 텐서로 변환
    source = pad_sequence(
        data.encoded_token_eng.tolist(),
        batch_first=True,
    )   # 변환된 텐서 리스트를 패딩하여 동일한 길이로 맞춤

    # fra vocab
    fra_vocab = ['<PAD>', '<SOS>', '<EOS>'] 
    fra_vocab = dict(zip(fra_vocab, range(len(fra_vocab)))) # 프랑스어 어휘 사전을 초기화하고 <PAD>, <SOS>, <EOS>에 인덱스 부여

    _fra_vocab = list(set(itertools.chain(*target.tolist())))   # 프랑스어 토큰 목록을 1차원으로 만들어 중복 제거한 후, <SOS>와 <EOS>를 제거
    _fra_vocab.remove('<SOS>')
    _fra_vocab.remove('<EOS>')    
    _fra_vocab = dict(zip(_fra_vocab, range(3, len(_fra_vocab)+3))) # 나머지 단어들에 인덱스를 부여 (3부터 시작)

    fra_vocab.update(_fra_vocab)    # 초기화한 프랑스어 어휘 사전에 나머지 단어들을 업데이트

    # fra token
    data['encoded_token_fra'] = data.token_fra.apply(
        lambda x: char_to_idx(x, fra_vocab)
    )   # 프랑스어 토큰을 인덱스로 변환하여 새로운 열에 저장    
    data.encoded_token_fra = data.encoded_token_fra.apply(
        lambda x: list_to_tensor(x)
    )   # 인덱스 리스트를 PyTorch 텐서로 변환
    target = pad_sequence(
        data.encoded_token_fra.tolist(),
        batch_first=True,
    )   # 변환된 텐서 리스트를 패딩하여 동일한 길이로 맞춤

    # 데이터셋을 학습용과 임시 데이터로 분할
    # temp 데이터를 검증용과 테스트용 데이터로 분할
    train_source, source_temp, train_target, target_temp = train_test_split(
        source,
        target,
        test_size=0.2,
        random_state=seed,
    )
    valid_source, test_source, valid_target, test_target = train_test_split(
        source_temp,
        target_temp,
        test_size=0.5,
        random_state=seed,
    )

    # 데이터셋 객체로 변환
    train_dataset = TranslationDataset(train_source, train_target)
    valid_dataset = TranslationDataset(valid_source, valid_target)
    test_dataset = TranslationDataset(test_source, test_target)

    # 데이터 모듈 생성 및 데이터 준비
    translation_data_module = TranslationDataModule(batch_size=configs.get('batch_size'))
    translation_data_module.prepare(train_dataset, valid_dataset, test_dataset)

