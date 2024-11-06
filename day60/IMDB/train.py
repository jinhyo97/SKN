from src.data import IMDBDataset, IMDBDataModule
from src.model.cnn_lstm import CNN_LSTM
from src.training import IMDBModule

import numpy as np
import json
import random

from sklearn.model_selection import train_test_split

from datasets import load_dataset
import torch
from torch.nn.utils.rnn import pad_sequence
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from tensorflow.keras.datasets import imdb


def main(configs):
    # cnn을 통해 imdb 리뷰 예측 모델
    # 1. dataset load and preprocessing
    # 2. CNN 모델 구축
    #   - max_pool
    #   - average_pool
    #   - max_pool + average_pool
    # 3. 모델 학습 및 성능 확인

    # 'imdb' 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    
    # preprocessing
    # 200보다 길이가 긴 문장은 200개로 잘라서 사용
    max_len = 200
    x_train = [torch.Tensor(sequence[:max_len]).long() for sequence in x_train]
    x_test = [torch.Tensor(sequence[:max_len]).long() for sequence in x_test]

    # max_len보다 작은 문장을 max_len으로 맞춤
    x_train = pad_sequence(x_train, batch_first=True)
    x_test = pad_sequence(x_test, batch_first=True)

    # 데이터셋을 학습용과 임시 데이터로 분할
    # test 데이터를 검증용과 테스트용 데이터로 분할
    x_valid, x_test = train_test_split(
        x_test,
        test_size=0.5,
        random_state=seed,
    )
    y_valid, y_test = train_test_split(
        y_test,
        test_size=0.5,
        random_state=seed,
    )

    # 데이터셋 객체로 변환
    train_dataset = IMDBDataset(x_train, y_train)
    valid_dataset = IMDBDataset(x_valid, y_valid)
    test_dataset = IMDBDataset(x_test, y_test)

    # DataModule 객체 생성하여 데이터 준비
    imdb_data_module = IMDBDataModule(
        batch_size=configs.get('batch_size'),
    )
    imdb_data_module.prepare(
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    # 학습을 위한 LightningModule 생성
    model = CNN_LSTM(configs)
    imdb_module = IMDBModule(
        model=model,
        configs=configs,
    )

    # Trainer 설정
    exp_name = 'test'  # 실험 이름 설정
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],  # 조기 종료 설정
        'logger': TensorBoardLogger(
            'tensorboard',
            f'IMDB/{exp_name}',
        ),  # 로그를 텐서보드로 설정
    }

    # GPU 사용 여부에 따라 Trainer 설정 업데이트
    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=imdb_module,
        datamodule=imdb_data_module,
    )
    trainer.test(
        datamodule=imdb_data_module,
    )


if __name__ == '__main__':
    # GPU가 사용 가능하면 'gpu', 아니면 'cpu'로 설정
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # 하이퍼파라미터 파일 로드
    with open(r'C:\Users\USER\.vscode\git\SKN\day60\IMDB\configs.json', 'r') as file:
        configs = json.load(file)
    configs.update({'device': device})

    # 시드 고정
    seed = configs.get('seed')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # GPU 시드 고정 및 성능 설정
    if device == 'gpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    
    # main 함수 실행
    main(configs)
