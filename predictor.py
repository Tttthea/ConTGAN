import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, dim):
        super(Predictor, self).__init__()
        self.dim = dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),

        )
        self.net2 = nn.Sequential(
            nn.Linear(14400, 512), #Helpdesk 14400 #sepsis 18496 #bpic12w 3136 #bp13c 4096 #bp13i 1600
            nn.ReLU(),
            nn.Linear(512, 256),  # Helpdesk
            nn.ReLU(),
            nn.Linear(256, 128),  # Helpdesk
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.dim),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.net(x)
        out = torch.flatten(out, 1)  # cnn
        out = self.net2(out)  # cnn
        return out

