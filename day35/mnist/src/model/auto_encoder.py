import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        configs,
        ):
        super().__init__()
        self.configs = configs
        self.input_dim = configs.get('input_dim')
        self.hidden_dim1 = configs.get('hidden_dim1')
        self.hidden_dim2 = configs.get('hidden_dim2')
        self.latent_dim = configs.get('latent_dim')
        self.dropout_ratio = configs.get('dropout_ratio')

        self.layer1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.output = nn.Linear(self.hidden_dim2, self.latent_dim)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.output(x)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        configs
        ):
        super().__init__()
        self.configs = configs
        self.input_dim = configs.get('input_dim')
        self.hidden_dim1 = configs.get('hidden_dim1')
        self.hidden_dim2 = configs.get('hidden_dim2')
        self.latent_dim = configs.get('latent_dim')
        self.dropout_ratio = configs.get('dropout_ratio')

        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim2),
            nn.BatchNorm1d(self.hidden_dim2),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(self.hidden_dim2, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_ratio),
        )

        self.output = nn.Linear(self.hidden_dim1, self.input_dim)
    
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
