import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # 설정값(configs)에서 입력 채널(input_channel)을 가져옵니다
        self.configs = configs.get('discriminator')
        self.input_dim = self.configs.get('input_channel')  # 입력 채널 수 (예: 흑백 이미지 -> 1, RGB -> 3)

        # 첫 번째 Convolutional Layer: Conv2d, BatchNorm, LeakyReLU 활성화, Dropout2d
        self.cnn1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, kernel_size=2, stride=2),  # Conv2d: 64개의 출력 채널, 커널 크기 2, 스트라이드 2
            nn.BatchNorm2d(64),  # Batch Normalization
            nn.LeakyReLU(0.2),   # Leaky ReLU 활성화 함수 (음수 부분에 0.2 비율로 기울기)
            nn.Dropout2d(),      # Dropout: 2D 레이어에서 노드 일부를 랜덤으로 비활성화
        )
        
        # 두 번째 Convolutional Layer: Conv2d, BatchNorm, LeakyReLU 활성화, Dropout2d
        self.cnn2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=2),  # 128개의 출력 채널
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
        )

        # 세 번째 Convolutional Layer
        self.cnn3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=2, stride=2),  # 256개의 출력 채널
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
        )

        # 네 번째 Convolutional Layer
        self.cnn4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=2, stride=2),  # 512개의 출력 채널
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(),
        )

        # 최종적으로 512차원에서 1차원으로 변환 (판별 결과는 실수 1개)
        self.fc = nn.Linear(512, 1)  # 512차원 -> 1차원
        self.sigmoid = nn.Sigmoid()  # Sigmoid 활성화 함수: 0과 1 사이의 값으로 출력 (이진 분류)

    def forward(self, x):
        # 네 개의 Convolutional Layer를 통과한 후 Flatten
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)

        # 4D 텐서를 2D 텐서로 변환 (배치 크기 x 특성 수)
        x = x.flatten(start_dim=1)  # 배치 크기를 제외한 나머지 차원은 펼침

        # fully connected 레이어를 통과하고, Sigmoid를 적용해 결과값을 0과 1 사이로 변환
        x = self.fc(x)
        x = self.sigmoid(x)

        return x  # 0 또는 1을 출력, 이는 진짜인지 가짜인지를 나타냄 (확률)

# Generator 클래스는 생성된 이미지를 만드는 모델입니다.
class Generator(nn.Module):
    def __init__(self, configs):
        super().__init__()

        # 설정값(configs)에서 배치 크기(batch_size)와 잠재 벡터 차원(latent_dim)을 가져옵니다
        self.configs = configs.get('generator')
        self.batch_size = self.configs.get('batch_size')
        self.latent_dim = self.configs.get('latent_dim')

        # 첫 번째 Transposed Convolution Layer (업샘플링)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                100,  # 입력 채널 수 (latent vector의 차원, 100)
                256,  # 출력 채널 수
                kernel_size=4  # 커널 크기
            ),
            nn.BatchNorm2d(256),  # Batch Normalization
            nn.ReLU(),            # ReLU 활성화 함수
            nn.Dropout2d(),       # Dropout2d
        )

        # 두 번째 Transposed Convolution Layer
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(
                256,  # 입력 채널 수
                128,  # 출력 채널 수
                kernel_size=4,
                stride=2,  # 스트라이드 2
                padding=1,  # 패딩 1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        # 세 번째 Transposed Convolution Layer
        self.conv3 = nn.Sequential(
            nn.ConvTranspose2d(
                128,  # 입력 채널 수
                64,   # 출력 채널 수
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        # 네 번째 Transposed Convolution Layer
        self.conv4 = nn.Sequential(
            nn.ConvTranspose2d(
                64,   # 입력 채널 수
                1,    # 출력 채널 수 (1 채널, 흑백 이미지)
                kernel_size=4,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        # Tanh 활성화 함수 (출력 범위를 [-1, 1]로 제한)
        self.tanh = nn.Tanh()

    def forward(self, x):  # (배치 크기, 잠재 벡터 차원)
        # 잠재 벡터 x를 (배치 크기, latent_dim, 1, 1) 형태로 변형 (업샘플링을 위한 형상)
        x = x.reshape(self.batch_size, self.latent_dim, 1, 1)
        
        # 각 ConvTranspose2d 레이어를 차례로 통과시키며 이미지를 생성
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # 최종적으로 Tanh 활성화 함수로 출력 값 제한
        x = self.tanh(x)

        return x  # 생성된 이미지를 반환
