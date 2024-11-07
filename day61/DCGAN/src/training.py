import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import lightning as L
import matplotlib.pyplot as plt


class MNISTModule(L.LightningModule):
    def __init__(
        self,
        generator: nn.Module,      # 생성자 모델 (nn.Module을 상속받은 생성 모델)
        discriminator: nn.Module,  # 판별자 모델 (nn.Module을 상속받은 판별 모델)
        configs: dict,             # 설정값을 담고 있는 딕셔너리 (예: learning_rate, batch_size 등)
    ):
        super().__init__()
        
        # 옵티마이저를 수동으로 설정하기 위해 자동 최적화 사용을 끔
        self.automatic_optimization = False

        # 모델을 초기화 (Generator, Discriminator)
        self.generator = generator          
        self.discriminator = discriminator
        
        # 설정값(configs)을 받아옵니다
        self.configs = configs
        self.learning_rate = self.configs.get('learning_rate')  # 학습률
        self.batch_size = self.configs.get('batch_size')        # 배치 크기
        self.latent_dim = self.configs.get('generator').get('latent_dim')  # 생성자의 잠재 벡터 차원

        # Binary Cross-Entropy 손실 함수 (GAN에서는 주로 이 함수 사용)
        self.criterion = nn.BCELoss()

    def training_step(self, batch, batch_idx):
        # 학습 단계에서 호출되는 메서드
        
        # 배치에서 실제 이미지와 레이블을 가져옵니다
        real_image = batch.get('X')  # 실제 이미지 데이터
        # real_image = real_image.flatten(start_dim=1)  # 이미지를 1D 벡터로 펼칩니다 (MNIST의 경우 28*28 = 784)
        y = batch.get('y')  # 레이블 데이터를 가져옵니다 (GAN에서는 사용하지 않을 수 있지만, 보통 레이블 있음)

        # 진짜와 가짜 이미지에 대한 레이블 설정
        real_labels = torch.ones(len(real_image), 1)   # 진짜 이미지의 레이블: 1
        fake_labels = torch.zeros(len(real_image), 1)  # 가짜 이미지의 레이블: 0

        # 옵티마이저 가져오기
        d_optimizer, g_optimizer = self.optimizers()

        # **Discriminator 훈련** (진짜와 가짜 이미지를 구별하도록 학습)
        self.toggle_optimizer(d_optimizer)  # 판별자의 옵티마이저를 활성화

        # 진짜 이미지 훈련
        real_pred = self.discriminator(real_image)  # 실제 이미지 판별
        d_loss_real = self.criterion(real_pred, real_labels)  # 진짜 이미지에 대한 손실 계산

        # 가짜 이미지 훈련
        z = torch.randn(len(real_image), self.latent_dim)  # 잠재 벡터 샘플링
        fake_images = self.generator(z)  # 생성자에서 가짜 이미지 생성
        fake_pred = self.discriminator(fake_images)  # 가짜 이미지 판별
        d_loss_fake = self.criterion(fake_pred, fake_labels)  # 가짜 이미지에 대한 손실 계산
        
        # 총 판별자 손실 (진짜와 가짜 이미지를 동시에 고려)
        d_loss = (d_loss_real + d_loss_fake) / 2
        self.log('d_loss', d_loss, prog_bar=True)  # 판별자 손실 로그

        # 역전파 및 옵티마이저 업데이트
        self.manual_backward(d_loss)
        d_optimizer.step()
        d_optimizer.zero_grad()

        self.untoggle_optimizer(d_optimizer)  # 판별자의 옵티마이저 비활성화

        # **Generator 훈련** (판별자가 가짜 이미지를 진짜로 분류하도록 훈련)
        self.toggle_optimizer(g_optimizer)  # 생성자의 옵티마이저를 활성화

        # 생성된 가짜 이미지와 그에 대한 판별자 출력
        z = torch.randn(len(real_image), self.latent_dim)  # 새로운 잠재 벡터 샘플링
        fake_images = self.generator(z)  # 생성자에서 가짜 이미지 생성
        fake_pred = self.discriminator(fake_images)  # 생성된 이미지를 판별자에 통과시켜 예측
        g_loss = self.criterion(fake_pred, real_labels)  # 가짜 이미지에 대한 손실 계산 (판별자가 가짜를 진짜로 분류하도록)

        self.log('g_loss', g_loss, prog_bar=True)  # 생성자 손실 로그
        self.manual_backward(g_loss)
        g_optimizer.step()
        g_optimizer.zero_grad()

        self.untoggle_optimizer(g_optimizer)  # 생성자의 옵티마이저 비활성화

        # 훈련 중에 생성된 일부 샘플을 시각화
        samples = fake_images[:25]  # 첫 25개의 생성된 이미지를 샘플링
        grid = torchvision.utils.make_grid(samples)  # 이미지를 그리드로 변환
        grid = grid.unsqueeze(1)  # 그리드의 차원 변경
        self.logger.experiment.add_images(
            'train/generated_images',  # 로그에 생성된 이미지를 기록
            grid,
            self.current_epoch,
        )

    def on_epoch_end(self):
        # 에폭이 끝날 때마다 호출되어 샘플 이미지를 생성하고 시각화
        z = torch.randn(self.batch_size, self.latent_dim)  # 배치 크기만큼 잠재 벡터 샘플링
        fake_images = self.generator(z).reshape(self.batch_size, 28, 28, 1)  # 생성된 이미지를 28x28x1 형태로 변환

        # 이미지를 4x4 격자에 시각화
        fig, axes = plt.subplots(4, 4, figsize=(4, 4))  # 4x4 그리드로 생성된 이미지를 표시
        for ax, img in zip(axes.flatten(), fake_images):
            ax.imshow(img.squeeze(), cmap='gray')  # 이미지를 그레이스케일로 표시
            ax.axis('off')  # 축 제거
        plt.show()  # 시각화된 이미지를 화면에 출력
    
    def validation_step(self, batch, batch_idx):
        # 검증 단계에서 호출되는 메서드
        real_image = batch.get('X')  # 실제 이미지 데이터 가져오기
        real_image = real_image.flatten(start_dim=1)  # 이미지를 1D 벡터로 펼칩니다
        y = batch.get('y')  # 레이블 데이터 가져오기

        # 진짜와 가짜 이미지에 대한 레이블 설정
        real_labels = torch.ones(len(real_image), 1)
        fake_labels = torch.zeros(len(real_image), 1)

        # 생성자에서 가짜 이미지 생성
        z = torch.randn(len(real_image), self.latent_dim)
        fake_images = self.generator(z)
        fake_pred = self.discriminator(fake_images)  # 판별자를 통해 예측값 계산
        
        # 생성자 손실 계산
        g_loss = self.criterion(fake_pred, real_labels)
        return g_loss  # 검증 중에는 생성자 손실만 반환

    def test_step(self, batch, batch_idx):
        # 테스트 단계에서 호출되는 메서드
        X = batch.get('X')  # 실제 이미지 데이터 가져오기
        X = X.flatten(start_dim=1)  # 이미지를 1D 벡터로 펼칩니다
        y = batch.get('y')  # 레이블 데이터 가져오기

        x, x_reconstructed, mu, log_var = self.model(X)  # 모델을 통해 예측값 계산

        return x, x_reconstructed, mu, log_var  # 예측된 결과 반환

    def configure_optimizers(self):
        # 옵티마이저와 학습률 스케줄러를 설정하는 메서드
        d_optimizer = optim.Adam(
            self.discriminator.parameters(),  # 판별자의 파라미터로 옵티마이저 설정
            lr=self.learning_rate,  # 학습률 설정
            betas=(0.5, 0.999),  # Adam의 beta 파라미터 설정
        )
        
        g_optimizer = optim.Adam(
            self.generator.parameters(),  # 생성자의 파라미터로 옵티마이저 설정
            lr=self.learning_rate,  # 학습률 설정
            betas=(0.5, 0.999),  # Adam의 beta 파라미터 설정
        )

        return [d_optimizer, g_optimizer], []  # 두 옵티마이저 반환 (판별자와 생성자)
