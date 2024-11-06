from collections import OrderedDict
import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.embedding = nn.Embedding(10000, 128)

        # feature learning
        self.conv = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=200,
                                   out_channels=100,
                                   kernel_size=5,
                                   padding='same')),
                ('layer_norm', nn.LayerNorm(128)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout1d()),
                ('pool', nn.MaxPool1d(kernel_size=2)),
            ])
        )

        self.lstm = nn.LSTM(
                64,
                64,
                batch_first=True
        )

        ## classification
        self.fc = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(6400, 256)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout1d()),
                ('fc2', nn.Linear(256, 2)),
            ])
        )
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.conv(x)
        x, _ = self.lstm(x)
        # x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
