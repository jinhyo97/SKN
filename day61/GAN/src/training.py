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
        self.criterion = nn.BCELoss()


    def training_step(self, batch, batch_idx, optimizer_idx):
        # 학습 단계에서 호출되는 메서드
        real_image = batch.get('X')  # 입력 데이터를 가져옴
        real_image = real_image.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴

        real_labels = torch.ones(len(real_image), 1)
        fake_labels = torch.zeros(len(real_image), 1)

        # discriminator train
        if optimizer_idx == 0:
            # real images train
            real_pred = self.discriminator(real_image)
            d_loss_real = self.criterion(real_pred, real_labels)

            # fake images train
            z = torch.randn(len(real_image), self.latent_dim)
            fake_images = self.generator(z)
            fake_pred = self.discriminator(fake_images)
            d_loss_fake = self.criterion(fake_pred, fake_labels)
        
            self.d_loss = (d_loss_real + d_loss_fake) / 2

            return self.d_loss
    
        if optimizer_idx == 1:
            z = torch.randn(len(real_image), self.latent_dim)
            fake_images = self.generator(z)
            fake_pred = self.discriminator(fake_images)
            self.g_loss = self.criterion(fake_pred, real_labels)

            return self.g_loss

    def on_epoch_end(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        fake_images = self.generator(z).reshape(self.batch_size, 28, 28, 1)

        fig, axes = plt.subplots(4, 4, figsize=(4, 4))
        for ax, img in zip(axes.flatten(), fake_images):
            ax.imshow(img, cmap='gray')
            ax.axis('off')
        plt.show()

    def on_train_epoch_end(self, *args, **kwargs):
        # 학습 에포크가 끝날 때 호출되는 메서드
        self.log_dict(
            {
                'loss/train_generator_loss': self.g_loss,
                'loss/train_discriminator_loss': self.d_loss,
            },  # 학습 손실을 로그에 기록
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
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
            lr=self.learning_rate,    # 학습률 설정
        )

        g_optimizer = optim.Adam(
            self.generator.parameters(),  # 모델의 파라미터를 옵티마이저에 전달
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

