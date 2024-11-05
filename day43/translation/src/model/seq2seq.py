import torch
import torch.nn as nn


# 인코더 클래스 정의
class Encoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, latent_dim: int, num_layers: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        # 단어 임베딩 레이어
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        # LSTM 레이어 정의
        self.lstm = nn.LSTM(
            self.embedding_dim,
            self.latent_dim,
            num_layers=self.num_layers,
            batch_first=True,
        )
    
    def forward(self, x):
        # 입력을 임베딩
        x = self.embedding(x)
        # LSTM을 통해 hidden state 얻기
        x, _ = self.lstm(x)

        return x


# 디코더 클래스 정의
class Decoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, latent_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        # 단어 임베딩 레이어
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)

        # attention layer
        self.attention = Attention(self.latent_dim)

        # 여러 개의 LSTM 레이어 정의
        self.lstm1 = nn.LSTM(self.embedding_dim+self.latent_dim, self.latent_dim, batch_first=True)
        self.lstm2 = nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        self.lstm3 = nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        self.lstm4 = nn.LSTM(self.latent_dim, self.latent_dim, batch_first=True)
        # 최종 출력 레이어
        self.fc_out = nn.Linear(self.latent_dim, self.vocab_size)

    def forward(
        self,
        x,
        hidden_state_decoder,
        hidden_state_encoder,
        hidden_state,
        cell_state,
        ):
        x = x[:, np.newaxis]   # 차원 추가 (배치, 1)
        x = self.embedding(x)  # 입력 임베딩

        # attention을 통한 context vector
        context_vector, _ = self.attention(
            hidden_state_decoder,
            hidden_state_encoder,
            hidden_state_encoder,
        )
        context_vector = context_vector[:, np.newaxis, :]

        # context vector와 x의 embedding 결합
        x = torch.concat([x, context_vector], axis=-1)

        # 여러 LSTM 레이어를 순차적으로 통과
        x, _ = self.lstm1(x, (hidden_state, cell_state))
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        x, (h_n, c_n) = self.lstm4(x)  # 마지막 LSTM 레이어의 출력과 상태 반환
        x = self.fc_out(x)  # 최종 출력 생성 (단어 확률 분포)

        return x, (h_n, c_n)  # 출력 및 마지막 hidden state, cell state 반환


# Seq2Seq 모델 정의
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, teacher_forcing_ratio: float = 0.5):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio  # teacher forcing 비율 설정
    
    def forward(self, source, target):
        batch_size = len(source)  # 배치 크기
        target_length = target.shape[1]  # 목표 시퀀스의 길이
        target_vocab_size = self.decoder.vocab_size  # 출력 어휘 크기
        outputs = torch.zeros(batch_size, target_length, target_vocab_size)  # 출력을 저장할 텐서 초기화

        # 인코더를 통해 잠재 벡터 생성
        hidden_state_encoder = self.encoder(source)

        # <SOS> token의 hidden state 구하기
        x = target[:, 0]                # batch
        x = x[:, np.newaxis]            # batch, 1
        x = self.decoder.embedding(x)   # batch_size, 1, dim
        context_vector = torch.zeros_like(x)
        x = torch.concat([x, context_vector], axis=-1)
        x, _ = self.decoder.lstm1(x)    # batch_size, 1, dim
        x, _ = self.decoder.lstm2(x)    # batch_size, 1, dim
        x, _ = self.decoder.lstm3(x)    # batch_size, 1, dim
        x, (h_n, c_n) = self.decoder.lstm4(x)
        hidden_state_decoder = h_n[0]

        # 첫 번째 token (sos제외) 입력
        x = target[:, 1]

        # 목표 시퀀스의 각 타임스텝에 대해 반복
        for t in range(1, target_length):
            # 디코더를 통해 출력 및 다음 hidden/cell state 얻기
            output, (h_n, c_n) = self.decoder(
                x,
                hidden_state_decoder,
                hidden_state_encoder,
                h_n,
                c_n,
            )
            hidden_state_decoder = h_n[0]
            outputs[:, t - 1, :] = output[:, 0, :]  # 현재 타임스텝의 출력 저장

            # teacher forcing 사용 여부 결정
            if (np.random.random() < self.teacher_forcing_ratio):
                x = output[:, 0, :].argmax(axis=-1)  # 모델의 출력을 다음 입력으로 사용
            else:
                if t < target_length-2:
                    x = target[:, t+1]  # 실제 타겟을 다음 입력으로 사용
        
        return outputs  # 모든 출력 반환
