import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        dropout_ratio: float = 0.5,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropout_ratio = dropout_ratio

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(500, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.output = nn.Linear(300, self.latent_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        dropout_ratio: float = 0.5,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.dropot_ratio = dropout_ratio

        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dim, 300),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(),
            nn.Dropout(self.dropot_ratio),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(300, 500),
            nn.BatchNorm1d(500),
            nn.LeakyReLU(),
            nn.Dropout(self.dropot_ratio),
        )

        self.output = nn.Linear(500, self.input_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, x):
        latent_vector = self.encoder(x)
        x_reconstructed = self.decoder(latent_vector)

        return x_reconstructed, latent_vector
