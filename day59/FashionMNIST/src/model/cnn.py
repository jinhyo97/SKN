from collections import OrderedDict
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        # feature learning
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=1,
                                   out_channels=64,
                                   kernel_size=2,
                                   padding='same')),
                ('batch_norm', nn.BatchNorm2d(64)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout2d()),
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ])
        )

        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', nn.Conv2d(in_channels=64,
                                   out_channels=128,
                                   kernel_size=2,
                                   padding='same')),
                ('batch_norm', nn.BatchNorm2d(128)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout2d()),
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ])
        )

        self.conv3 = nn.Sequential(                                                             
            OrderedDict([                       
                ('conv', nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=2,
                                   padding='same')),
                ('batch_norm', nn.BatchNorm2d(256)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout2d()),
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ])
        )

        self.conv4 = nn.Sequential(                                                             
            OrderedDict([                       
                ('conv', nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=2,
                                   padding='same')),
                ('batch_norm', nn.BatchNorm2d(512)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout2d()),
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ])
        )

        ## classification
        self.fc = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(512, 256)),
                ('batch_norm', nn.BatchNorm1d(256)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout1d()),
                ('fc2', nn.Linear(256, 10)),
            ])
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
