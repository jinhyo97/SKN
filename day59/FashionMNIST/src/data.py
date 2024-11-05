import torch
from torch.utils.data import DataLoader, Dataset
import lightning as L

class FashionMNISTDataset(Dataset):
    def __init__(self, data):  # 생성자 메서드
        super().__init__()  # 부모 클래스의 생성자를 호출하여 초기화
        self.data = data  # 입력 데이터 저장

    def __len__(self):  # 데이터셋의 길이 반환
        return len(self.data)
    
    def __getitem__(self, idx):  # 인덱스를 통해 샘플 반환
        return {
            'input_ids': torch.Tensor(self.data[idx].get('input_ids')).long(),
            'token_type_ids': torch.Tensor(self.data[idx].get('token_type_ids')).long(),
            'attention_mask': torch.Tensor(self.data[idx].get('attention_mask')).long(),
            'label': self.data[idx].get('label'),
        }


class FashionMNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size: int):
        super().__init__()
        self.batch_size = batch_size  # 배치 크기 설정

    def prepare(self, train_dataset, valid_dataset, test_dataset):
        # 준비 단계에서 학습, 검증, 테스트 데이터셋을 저장
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset

    def setup(self, stage: str):
        # 단계에 맞게 데이터셋 설정
        if stage == "fit":
            # 학습 및 검증 단계 데이터셋 설정
            self.train_data = self.train_dataset
            self.valid_data = self.valid_dataset

        if stage == "test":
            # 테스트 단계 데이터셋 설정
            self.test_data = self.test_dataset

    def train_dataloader(self):
        # 학습 데이터 로더 반환
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,  # 배치 크기 설정
            shuffle=True,  # 데이터셋 셔플
            drop_last=True,  # 배치 부족 시 마지막 데이터 제외
            num_workers=8,
        )

    def val_dataloader(self):
        # 검증 데이터 로더 반환
        return DataLoader(
            dataset=self.valid_data,
            batch_size=self.batch_size,
            shuffle=False,  # 셔플하지 않음
            drop_last=True,
            num_workers=8,
        )

    def test_dataloader(self):
        # 테스트 데이터 로더 반환
        return DataLoader(
            dataset=self.test_data,
            batch_size=self.batch_size,
            shuffle=False,  # 셔플하지 않음
            drop_last=True,
            num_workers=8,
        )
