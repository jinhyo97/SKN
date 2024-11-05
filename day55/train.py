from src.data import IMDBDataset, IMDBDataModule
from src.model.auto_encoder import Encoder, Decoder, AutoEncoder
from src.training import IMDBModule

import numpy as np
import json
import random

from datasets import load_dataset

import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(configs):
    # 모델과 토크나이저 설정
    model_name = configs.get('model_name')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # 전처리 함수 정의: 각 텍스트 예시를 토큰화하고 패딩 적용
    def preprocessing(example: str):
        return tokenizer(
            example['text'],
            padding='max_length',
            max_length=256,
            truncation=True,
            return_tensors='pt',
        )

    # IMDB 데이터셋 로드
    data = load_dataset('imdb')

    # 데이터셋 나누기 및 샘플링
    train = data['train'].shuffle(seed=configs.get('seed')).select(range(300))
    valid = train.select(range(100))
    test = data['test'].shuffle(seed=configs.get('seed')).select(range(100))

    # 토큰화된 데이터셋 생성
    tokenized_train_dataset = train.map(preprocessing, batched=True)
    tokenized_valid_dataset = valid.map(preprocessing, batched=True)
    tokenized_test_dataset = test.map(preprocessing, batched=True)

    # 토큰화된 데이터를 Dataset 객체로 변환
    train_dataset = IMDBDataset(tokenized_train_dataset)
    valid_dataset = IMDBDataset(tokenized_valid_dataset)
    test_dataset = IMDBDataset(tokenized_test_dataset)

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
    imdb_module = IMDBModule(
        model=model,
        configs=configs,
    )

    # Trainer 설정
    configs.update({'step': len(train) // configs.get('batch_size')})  # 스텝 수 계산
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])  # 실험 이름 설정
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


if __name__ == '__main__':
    # GPU가 사용 가능하면 'gpu', 아니면 'cpu'로 설정
    device = 'gpu' if torch.cuda.is_available() else 'cpu'

    # 하이퍼파라미터 파일 로드
    with open('./configs.json', 'r') as file:
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
