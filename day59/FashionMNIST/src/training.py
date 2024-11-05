import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class FashionMNISTModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,  # nn.Module을 상속받은 모델 객체
        configs: dict,     # 학습 설정을 담고 있는 딕셔너리
    ):
        super().__init__()
        self.model = model               # 모델 인스턴스 설정
        self.configs = configs
        self.learning_rate = self.configs.get('learning_rate')  # 학습률 설정

    def training_step(self, batch, batch_idx):
        X, y = batch
        preds = self.model(X)  # 모델 예측 수행

        # 예측값과 실제 레이블을 바탕으로 손실값 계산 (Cross Entropy Loss)
        self.loss = F.cross_entropy(preds, y)

        return self.loss  # 계산된 손실 반환

    def on_train_epoch_end(self, *args, **kwargs):
        # 매 에포크 종료 시 학습 손실을 기록하여 로그로 남김
        self.log_dict(
            {'loss/train_loss': self.loss},  # 'train_loss' 키로 손실 로그 기록
            on_epoch=True,
            prog_bar=True,  # 진행 막대에 손실값 표시
            logger=True,    # 로그에 기록
        )
    
    def validation_step(self, batch, batch_idx):
        # 검증 단계: 입력 데이터를 통해 손실을 계산하고 반환
        X, y = batch
        preds = self.model(X)  # 모델 예측 수행

        # 예측값과 실제 레이블을 바탕으로 검증 손실 계산
        self.val_loss = F.cross_entropy(preds, y)

        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        # 검증 에포크 종료 시 검증 손실과 학습률을 기록하여 로그로 남김
        self.log_dict(
            {'loss/val_loss': self.val_loss,  # 'val_loss' 키로 검증 손실 기록
             'learning_rate': self.learning_rate},  # 학습률을 로그에 기록
            on_epoch=True,
            prog_bar=True,  # 진행 막대에 표시
            logger=True,    # 로그에 기록
        )

    def test_step(self, batch, batch_idx):
        X, y = batch        
        preds = self.model(X)  # 모델 예측 수행

        # 최종 예측된 클래스 레이블 반환
        return (preds.argmax(axis=-1) == y).float()

    def configure_optimizers(self):
        optimizer = optim.Adam(           # Adam 옵티마이저 설정
            self.model.parameters(),
            lr=self.learning_rate,
        )

        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'interval': 'step',  # 스케줄러 업데이트 주기 (매 스텝)
            #     'frequency': 1,
            # }
        }
