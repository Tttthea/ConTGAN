import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

#gan-cnn
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.net2 = nn.Sequential(
            nn.Linear(14400, 512),  #Helpdesk 14400 #sepsis 18496 #bpic12w 3136 #bp13c 4096 #bp13i 1600
            nn.ReLU(),
            nn.Linear(512, 256),  # Helpdesk
            nn.ReLU(),
            nn.Linear(256, 128),  # Helpdesk
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.flatten(out, 1)  # cnn
        out = self.net2(out)  # cnn
        return out

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Linear(16, 64),
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(num_features=128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.BatchNorm1d(num_features=1),
            nn.Sigmoid()  # Sigmoid activation to output a probability
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        return self.fc(out)


