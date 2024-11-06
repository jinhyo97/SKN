import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import lightning as L

import matplotlib.pyplot as plt


class MNISTModule(L.LightningModule):
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        configs: dict
    ):
        super().__init__()
        self.automatic_optimization = False

        self.generator = generator          # 모델을 초기화
        self.discriminator = discriminator  # 모델을 초기화
        self.configs = configs
        self.learning_rate = self.configs.get('learning_rate')
        self.batch_size = self.configs.get('batch_size')  

        self.criterion = nn.BCELoss()


    def training_step(self, batch, batch_idx, optimizer_idx):
        # 학습 단계에서 호출되는 메서드
        real_image = batch.get('X')  # 입력 데이터를 가져옴
        real_image = real_image.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴

        real_labels = torch.ones(len(real_image), 1)
        fake_labels = torch.zeros(len(real_image), 1)

        d_optimizer, g_optimizer = self.optimizers()

        # discriminator train
        self.toggle_optimizer(d_optimizer)

        # real image train
        real_pred = self.discriminator(real_image)
        d_loss_real = self.criterion(real_pred, real_labels)

        # 가짜 이미지 훈련
        z = torch.randn(len(real_image), self.latent_dim)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images)
        d_loss_fake = self.criterion(fake_pred, fake_labels)
        d_loss = (d_loss_real+d_loss_fake) / 2
        self.log('d_loss', d_loss, prog_bar=True)
        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()

        self.untoggle_optimizer(d_optimizer)


        # generator train
        self.toggle_optimizer(g_optimizer)

        z = torch.randn(len(real_image), self.latent_dim)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images)
        self.g_loss = self.criterion(fake_pred, real_labels)

        self.untoggle_optimizer(g_optimizer)

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
        # 학습 단계에서 호출되는 메서드
        real_image = batch.get('X')  # 입력 데이터를 가져옴
        real_image = real_image.flatten(start_dim=1) # 입력 차원을 flatten
        y = batch.get('y')  # 레이블 데이터를 가져옴

        real_labels = torch.ones(len(real_image), 1)
        fake_labels = torch.zeros(len(real_image), 1)

        z = torch.randn(len(real_image), self.latent_dim)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images)
        self.g_loss = self.criterion(fake_pred, real_labels)

        return self.g_loss
    

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

        return [d_optimizer, g_optimizer], []