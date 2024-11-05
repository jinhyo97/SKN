from src.data import IMDBDataset, IMDBDataModule
from src.model.lstm import Model
from src.training import IMDBModule

import pandas as pd
import numpy as np
import random
import json
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import lightning as L
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


def main(configs):
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

    # DataLoader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.get('batch_size'),
    )

    # 모델 생성
    configs.update({
        'input_dim': x_train.shape[-1],
        'seq_len': x_train.shape[1],
        })
    model = Model(configs)

    # LightningModule 인스턴스 생성: 모델과 학습률 설정
    imdb_module = IMDBModule.load_from_checkpoint(
        r'C:\Users\USER\Documents\git\2024-08-05_Encore\day34\tensorboard\IMDB\test\version_2\checkpoints\epoch=1-step=50.ckpt',
        model=model,
        learning_rate=configs.get('learning_rate'),  # 설정에서 학습률 가져오기
    )
    # 모델을 평가 모드로 전환: 드롭아웃 등 비활성화
    model.eval()

    # prediction
    preds = []
    gts = []
    for batch in test_dataloader:
        X = batch.get('X')
        y = batch.get('y')
        pred = imdb_module.model(X)
        preds.append(pred.argmax(axis=-1))
        gts.append(y)

    preds = torch.cat(preds)
    gts = torch.cat(gts)


    confusion_matrix_result = confusion_matrix(gts, preds)
    TP = confusion_matrix_result[0, 0]
    FN = confusion_matrix_result[0, 1]
    FP = confusion_matrix_result[1, 0]
    TN = confusion_matrix_result[1, 1]

    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1 = 2*precision*recall / (precision+recall)
    accuracy = (TP+TN) / (TP+TN+FP+FN)

    print(f'precision: {precision: .2f}, recall: {recall: .2f}, recall: {recall: .2f}, accuracy: {accuracy: .2f}')


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