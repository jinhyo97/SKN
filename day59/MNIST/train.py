from src.data import MNISTDataset, MNISTDataModule
from src.model.cnn import CNN
from src.training import MNISTModule

import numpy as np
import json
import random

from datasets import load_dataset

import torch
import torchvision.datasets as ds
import torchvision.transforms as transforms

from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(configs):
    transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),      # 50%확률로 랜덤하게 좌우 반전
        transforms.ToTensor(),  # tensor + scaling (0 ~ 1)
    ])

    # MNIST 
    train_dataset = ds.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = ds.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform,
    )

    # 데이터셋 나누기 및 샘플링
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset,
        [50000, 10000]
    )

    # DataModule 객체 생성하여 데이터 준비
    mnist_data_module = MNISTDataModule(
        batch_size=configs.get('batch_size'),
    )
    mnist_data_module.prepare(
        train_dataset,
        valid_dataset,
        test_dataset,
    )

    # 학습을 위한 LightningModule 생성
    model = CNN(configs)
    mnist_module = MNISTModule(
        model=model,
        configs=configs,
    )

    # Trainer 설정
    exp_name = ','.join([f'{key}={value}' for key, value in configs.items()])  # 실험 이름 설정
    trainer_args = {
        'max_epochs': configs.get('epochs'),
        'callbacks': [
            EarlyStopping(monitor='loss/val_loss', mode='min', patience=10)
        ],  # 조기 종료 설정
        'logger': TensorBoardLogger(
            'tensorboard',
            f'MNIST/{exp_name}',
        ),  # 로그를 텐서보드로 설정
    }

    # GPU 사용 여부에 따라 Trainer 설정 업데이트
    if configs.get('device') == 'gpu':
        trainer_args.update({'accelerator': configs.get('device')})

    trainer = Trainer(**trainer_args)

    # 모델 학습 시작
    trainer.fit(
        model=mnist_module,
        datamodule=mnist_data_module,
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
