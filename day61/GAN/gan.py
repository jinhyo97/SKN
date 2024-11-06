import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs.get('generator')
        self.latent_dim = configs.get('latent_dim', 10)

        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(256, 28*28),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, z):
        x = self.layer1(z)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)

        return x
    

class Discriminator(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs.get('discriminator')
        self.latent_dim = configs.get('input_dim')

        self.layer1 = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.layer3 = nn.Sequential(
            nn.Linear(64, 1),
            nn.ReLU(),
            nn.Dropout(),
        )

        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.sigmoid(x)

        return x