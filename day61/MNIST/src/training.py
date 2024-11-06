import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L


class MNISTModule(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,          # 모델 객체 (nn.Module을 상속받은 모델)
        learning_rate: float,      # 학습률
    ):
        super().__init__()
        self.model = model         # 모델을 초기화
        self.learning_rate = learning_rate  # 학습률을 초기화

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        X = X.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴

        x, x_reconstructed, mu, log_var = self.model(X)  # 모델을 통해 예측값을 계산
        self.loss = self.elbo(x, x_reconstructed, mu, log_var)  # 예측값과 실제 값 간의 손실 계산 (MSE 손실 함수 사용)

        return self.loss  # 계산된 손실 반환
    
    def on_train_epoch_end(self, *args, **kwargs):
        # 학습 에포크가 끝날 때 호출되는 메서드
        self.log_dict(
            {'loss/train_loss': self.loss},  # 학습 손실을 로그에 기록
            on_epoch=True,
            prog_bar=True,  # 진행 막대에 표시
            logger=True,    # 로그에 기록
        )
    
    def validation_step(self, batch, batch_idx):
        # 검증 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        X = X.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴

        x, x_reconstructed, mu, log_var = self.model(X)  # 모델을 통해 예측값을 계산
        self.val_loss = self.elbo(x, x_reconstructed, mu, log_var)  # 예측값과 실제 값 간의 손실 계산 (MSE 손실 함수 사용)

        return self.val_loss  # 검증 손실 반환
    
    def on_validation_epoch_end(self):
        # 검증 에포크가 끝날 때 호출되는 메서드
        self.log_dict(
            {'loss/val_loss': self.val_loss,  # 검증 손실을 로그에 기록
             'learning_rate': self.learning_rate},  # 학습률도 로그에 기록
            on_epoch=True,
            prog_bar=True,  # 진행 막대에 표시
            logger=True,    # 로그에 기록
        )

    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 입력 데이터를 가져옴
        X = X.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴      

        x, x_reconstructed, mu, log_var = self.model(X)  # 모델을 통해 예측값을 계산

        return x, x_reconstructed, mu, log_var  # 예측된 레이블 반환

    def configure_optimizers(self):
        # 옵티마이저와 스케줄러를 설정하는 메서드
        optimizer = optim.Adam(
            self.model.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # 학습률 설정
        )

        return {
            'optimizer': optimizer,   # 옵티마이저 반환
            # 'scheduler': scheduler,   # 학습률 스케줄러 반환
        }


    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)

        return mu + sigma*epsilon
    

    def elbo(self, x, x_reconstructed, mu, log_var):
        # reconstructed error (BCE)
        BCE = F.binary_cross_entropy(x_reconstructed, x)

        # KL divergence
        KL = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()

        return BCE + KL   

