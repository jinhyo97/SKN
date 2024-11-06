import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, configs={}):
        super().__init__()
        self.configs = configs
        self.hidden_dim = self.configs.get('hidden_dim', 128)
        self.latent_dim = self.configs.get('latent_dim', 20)

        self.fc1 = nn.Linear(28*28, self.hidden_dim)
        self.mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.sigma = nn.Linear(self.hidden_dim, self.latent_dim)

    def forward(self, x):
        x = self.fc1(x)

        mu = self.mu(x)
        sigma = self.sigma(x)

        return mu, sigma
    

class Decoder(nn.Module):
    def __init__(self, configs={}):
        super().__init__()
        self.configs = configs
        self.hidden_dim = self.configs.get('hidden_dim', 128)
        self.latent_dim = self.configs.get('latent_dim', 20)

        self.fc1 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 28*28)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.sigmoid(x)

        return x
    

class VAE(nn.Module):
    def __init__(self, encoder, decoder, configs={}):
        super().__init__()
        self.configs = configs

        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(sigma)

        return mu + sigma*epsilon

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)

        return x, x_reconstructed, mu, log_var
        
    def loss(self, x, x_reconstructed, mu, log_var):
        # reconstructed error (BCE)
        BCE = F.mse_loss(x_reconstructed, x)

        # KL divergence
        KL = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum()

        return BCE + KL