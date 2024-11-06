from collections import OrderedDict
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs

        self.embedding = nn.Embedding(10000, 128)

        # feature learning
        self.convs=nn.ModuleList()

        for kernel_size in (3, 5, 7, 11, 13):
            self.convs.append(
                nn.Sequential(
                    OrderedDict([
                        ('conv', nn.Conv1d(
                            200,
                            64,
                            kernel_size=kernel_size,
                            padding='same')),
                        ('pooling', nn.MaxPool1d(kernel_size=128))
                        ])
                )
            )


        ## classification
        self.fc = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(3200, 256)),
                ('relu', nn.ReLU()),
                ('dropout', nn.Dropout1d()),
                ('fc2', nn.Linear(256, 2)),
            ])
        )
    
    def forward(self, x):
        x = self.embedding(x)
        max_outs = [conv(x) for conv in self.convs]
        x = torch.cat(max_outs, axis=-1)

        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x
