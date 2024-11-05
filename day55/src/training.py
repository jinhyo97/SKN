import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning as L


class IMDBModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,  # nn.Module을 상속받은 모델 객체
        configs: dict,     # 학습 설정을 담고 있는 딕셔너리
    ):
        super().__init__()
        self.model = model               # 모델 인스턴스 설정
        self.configs = configs
        self.learning_rate = self.configs.get('learning_rate')  # 학습률 설정

        # 초기에는 모든 모델 파라미터를 학습 불가능 상태로 고정
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def training_step(self, batch, batch_idx):
        # 학습 단계: 입력 데이터와 레이블로 손실을 계산하고 반환
        inputs = {
            'input_ids': batch.get('input_ids'),  # 입력 ID
            'token_type_ids': batch.get('token_type_ids'),  # 토큰 타입 ID
            'attention_mask': batch.get('attention_mask'),  # 어텐션 마스크
        }
        y = batch.get('label')  # 정답 레이블

        preds = self.model(**inputs)  # 모델 예측 수행

        # 예측값과 실제 레이블을 바탕으로 손실값 계산 (Cross Entropy Loss)
        self.loss = F.cross_entropy(preds.logits, y)

        return self.loss  # 계산된 손실 반환

    def on_train_epoch_start(self, *args, **kwargs):
        # 에포크가 시작될 때 파라미터를 순차적으로 학습 가능하게 전환
        for name, parameter in self.model.named_parameters():
            # 학습 단계에 따라 'classifier' 층부터 'embedding' 층까지 단계별로 가중치를 학습 가능하게 설정
            if self.global_step == (0 * self.configs.get('step')) and 'classifier' in name:
                parameter.requires_grad = True
            
            if self.global_step == (1 * self.configs.get('step')) and 'pooler' in name:
                parameter.requires_grad = True
            
            if ((2 * self.configs.get('step')) <= self.global_step < (14 * self.configs.get('step'))
                and f'bert.encoder.layer.{13 - self.global_step // self.configs.get("step")}' in name
            ):
                parameter.requires_grad = True
            
            if self.global_step == (14 * self.configs.get('step')) and 'embedding' in name:
                parameter.requires_grad = True

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
        inputs = {
            'input_ids': batch.get('input_ids'),
            'token_type_ids': batch.get('token_type_ids'),
            'attention_mask': batch.get('attention_mask'),
        }
        y = batch.get('label')  # 검증 데이터의 레이블

        preds = self.model(**inputs)  # 모델 예측 수행

        # 예측값과 실제 레이블을 바탕으로 검증 손실 계산
        self.val_loss = F.cross_entropy(preds.logits, y)

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
        # 테스트 단계: 입력 데이터를 통해 예측 결과를 반환
        inputs = {
            'input_ids': batch.get('input_ids'),
            'token_type_ids': batch.get('token_type_ids'),
            'attention_mask': batch.get('attention_mask'),
        }
        y = batch.get('label')  # 테스트 데이터의 레이블

        preds = self.model(**inputs)  # 모델 예측 수행

        # 최종 예측된 클래스 레이블 반환
        return preds.logits.argmax(axis=-1)

    def configure_optimizers(self):
        # 최적화 설정: Adam 옵티마이저와 OneCycleLR 스케줄러를 구성
        params = [{
            'params': self.model.bert.embeddings.parameters(),
            'lr': self.learning_rate / (2.6 ** 14)  # 가장 작은 학습률로 임베딩 레이어 설정
        }]
        params.extend([
            {
                'params': self.model.bert.encoder.layer[-i].parameters(),
                'lr': self.learning_rate / (2.6 ** (i + 1))  # 레이어가 깊을수록 작은 학습률 할당
            }
            for i in range(12, 0, -1)
        ])
        params.extend([
            {
                'params': self.model.bert.pooler.parameters(),
                'lr': self.learning_rate / (2.6 ** 1)  # 학습률 설정
            },
            {
                'params': self.model.classifier.parameters(),
                'lr': self.learning_rate / (2.6 ** 0)  # 분류기의 학습률 설정
            },
        ])
        optimizer = optim.Adam(params)  # Adam 옵티마이저 설정

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.configs.get('max_lr'),  # 학습률 최대값
            pct_start=self.configs.get('pct_start'),  # 학습률 상승 비율
            total_steps=self.configs.get('step') * self.configs.get('epochs')  # 전체 학습 단계 수
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # 스케줄러 업데이트 주기 (매 스텝)
                'frequency': 1,
            }
        }
