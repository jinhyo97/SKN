from src.data import StockDataset, StockDataModule
from src.model.rnn import Model
from src.training import StockModule

import pandas as pd
import numpy as np
import random
import json
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

import seaborn as sns


def main(configs):
    # 'samsung' 데이터셋 로드
    fine_dust = pd.read_csv(
        './data/서울시 대기질 자료 제공_2022.csv',
        encoding='cp949',
    )
    
    # preprocessing
    fine_dust = fine_dust.query('구분=="평균"')
    fine_dust = fine_dust.drop(columns=['구분'])
    fine_dust.일시 = pd.to_datetime(fine_dust.일시)
    fine_dust = fine_dust.sort_values(by=['일시'])
    fine_dust = fine_dust.set_index(['일시'])
    fine_dust = fine_dust.astype(np.float32)          # 데이터프레임을 float32로 변환

    # 미세먼지와 초미세먼지의 평균을 target으로 설정
    fine_dust['target'] = fine_dust.mean(axis=1)

    # 데이터셋을 학습용과 임시 데이터로 분할
    train, temp = train_test_split(
        fine_dust,
        test_size=0.4,
        random_state=seed,
        shuffle=False,
    )

    # 임시 데이터를 검증용과 테스트용 데이터로 분할
    valid, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=seed,
        shuffle=False,
    )

    standard_scaler = StandardScaler()

    # 훈련 데이터의 열을 표준화
    train.loc[:, fine_dust.columns] = \
        standard_scaler.fit_transform(train.loc[:, fine_dust.columns])

    # 검증 데이터와 테스트 데이터의 열을 훈련 데이터의 통계로 표준화
    valid.loc[:, fine_dust.columns] = \
        standard_scaler.transform(valid.loc[:, fine_dust.columns])

    test.loc[:, fine_dust.columns] = \
        standard_scaler.transform(test.loc[:, fine_dust.columns])
    
    # X, y 생성
    window_size = configs.get('window_size')
    output_dim = configs.get('output_dim')
    x_train = np.lib.stride_tricks.sliding_window_view(
        train.iloc[:-output_dim, :-1], window_size, axis=0).transpose(0, 2, 1)
    y_train = np.lib.stride_tricks.sliding_window_view(
        train.target.iloc[window_size:], output_dim
    )
    x_valid = np.lib.stride_tricks.sliding_window_view(
        valid.iloc[:-output_dim, :-1], window_size, axis=0).transpose(0, 2, 1)
    y_valid = np.lib.stride_tricks.sliding_window_view(
        valid.target.iloc[window_size:], output_dim, axis=0
    )  
    x_test = np.lib.stride_tricks.sliding_window_view(
        test.iloc[:-output_dim, :-1], window_size, axis=0).transpose(0, 2, 1)
    y_test = np.lib.stride_tricks.sliding_window_view(
        test.target.iloc[window_size:], output_dim, axis=0
    )

    # 데이터셋 객체로 변환
    train_dataset = StockDataset(x_train, y_train)
    valid_dataset = StockDataset(x_valid, y_valid)
    test_dataset = StockDataset(x_test, y_test)

    # 데이터 모듈 생성 및 데이터 준비
    stock_data_module = StockDataModule(batch_size=configs.get('batch_size'))
    stock_data_module.prepare(train_dataset, valid_dataset, test_dataset)

    # 모델 생성
    configs.update({
        'input_dim': x_train.shape[-1],
        'seq_len': x_train.shape[1],
        })
    model = Model(configs)

    # LightningModule 인스턴스 생성
    stock_module = StockModule(
        model=model,
        learning_rate=configs.get('learning_rate'),
    )

    # Trainer 인스턴스 생성 및 설정
    del configs['output_dim'], configs['seed'], configs['epochs'], configs['seq_len'], configs['input_dim']
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],
        'logger': TensorBoardLogger(
            'tensorboard',
            f'fine_dust/{exp_name}',
        ),
    }

    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=stock_module,
        datamodule=stock_data_module,
    )


if __name__ == '__main__':
    # 사용 가능한 GPU가 있는 경우 'cuda', 그렇지 않으면 'cpu' 사용
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # hyperparameter
    with open('./configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # seed 설정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    main(configs)