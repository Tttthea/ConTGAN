import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from MyDataset import MyDataset
from encoding import *
from GAN import Generator, Discriminator
from ImagePPMiner import ImagePPMiner
from helper import *
from train import *
from test import *
from datetime import datetime
from train2 import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

#randomness control
torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
torch.manual_seed(42)
np.random.seed(42)

if __name__ == '__main__':
    dataset = "Helpdesk" #rename it to the dataset you want, only the name of the event log, no .xes no path
    pm = ImagePPMiner(dataset)
    log = pm.import_log()
    print(log)
    n_act = log['concept:name'].nunique()
    graphs = encoding2(log)
    data = MyDataset(graphs)

    #holdout 2/1 sequential
    lengths = len(data)
    indices = list(range(lengths))
    split = int(lengths * 2/3)
    batch_size = 100
    train_indices, test_indices = indices[split:], indices[:split]
    train_sampler = torch.utils.data.RandomSampler(train_indices)
    test_sampler = torch.utils.data.RandomSampler(test_indices)
    trainloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, sampler=train_sampler, collate_fn=lambda batch: my_collate(batch, dim=n_act))
    testloader = DataLoader(dataset=data, batch_size=batch_size, shuffle=False, sampler=test_sampler, collate_fn=lambda batch: my_collate(batch, dim=n_act))

    # choose the algorithm
    # 1. Backbone model: train2(), 2. GAN training: train(), 3. GAN-augmentation: train_aug(), 4. GAN-filtering: retrain()
    # must run 1 and 2 before 3 and 4,
    # or make sure exists backbone model.pt and generator/discriminator.pt
    # you need to change the input dimension of FNN in GAN.py and predictor.py according to the datasets you choose
    algorithm_num = 4 # change to the number
    if algorithm_num == 1:
        print("backbone model is training")
        p, p_model = train2(dataset, trainloader, dim=n_act, device=DEVICE)
        test(testloader, p, device=DEVICE, dim = n_act)
    elif algorithm_num == 2:
        print("GAN is training")
        g, d, g_model, d_model = train(dataset, trainloader, n_act, device=DEVICE)
    elif algorithm_num == 3:
        print("augmentation is working")
        g_model = "g_100_lo_helpdesk_cnn_trail.pt" # rename to the generated .pt file from 2
        p_model = "p_30e_5l_helpdesk.pt" # rename to the generated .pt file from 1
        p = train_aug(trainloader, n_act, DEVICE, g_model, p_model)
        test(testloader, p, device=DEVICE, dim=n_act)
    elif algorithm_num == 4:
        print("filtering is working")
        d_model = "d_100_lo_helpdesk_cnn_trail.pt" # rename to the generated .pt file from 2
        p_model = "p_30e_5l_helpdesk.pt" # rename to the generated .pt file from 1
        p = retrain(trainloader, n_act, d_model, p_model, DEVICE)
        test(testloader, p, DEVICE, n_act)
    else:
        print("please choose one of the number before")

    #statistics
    # statistics(graphs, g_model, n_act, DEVICE)





