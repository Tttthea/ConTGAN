import torch
import torch.nn as nn
import pandas as pd
import copy
import os
from helper import *
from torch.autograd import Variable
from predictor import Predictor


def train2(
        dataset,
        loader,
        dim,
        device,
        epoch_num: int = 30,
        lr: int = 0.005,
        n_critic: int = 500,
        criterion=torch.nn.CrossEntropyLoss(),
):
    '''

    :param dataset: event log name for storing
    :param loader: train dataloader
    :param dim: n_act, number of unique activities
    :param device: "cuda" or "cpu"
    :param epoch_num: 30/50/100...
    :param lr: 0.005
    :param n_critic: 500
    :param criterion: loss function
    :return: initially trained backbone model, name of the model
    '''
    correct = 0
    total = 0

    p = Predictor(dim+1).to(device)
    optimizer = torch.optim.Adam(p.parameters(), lr=lr)

    for epoch in range(epoch_num):
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader): #x: 50*16*16
            x, label, adj_tar = x.to(device), label.to(device), adj_tar.to(device)
            x = x.view(x.shape[0], 1, dim + 1, dim + 1) #cnn-pre
            x.requires_grad
            y_hat = p(x)
            loss = criterion(y_hat, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            estimated = torch.argmax(y_hat, dim=1).to(device)
            real = torch.argmax(label, dim=1).to(device)
            if batch % n_critic == 0:
                print(
                    "Epoch: {:2d}/{:2d}   ".format(epoch + 1, epoch_num),
                    "Batch: [{:2d}/{:2d}]  ".format(batch * len(x), len(loader.dataset)),
                    "loss: {:.6f}   ".format(loss.item()),
                )
            total += label.size(0)
            correct += (estimated == real).sum().item()
    print("accuracy: {:.4f}".format(float(correct) / total))
    to_path = "p_" + str(epoch_num) + "e_" + dataset + ".pt"
    pre = os.path.join(to_path)
    torch.save(p.state_dict(), pre)
    return p, to_path
