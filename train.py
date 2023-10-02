import torch
import torch.nn as nn
import pandas as pd
import copy
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from helper import *
from torch.autograd import Variable
from GAN import Generator, Discriminator
from predictor import Predictor


def cal_gradient_penalty(D, device, real, fake, dim):
    sigma = torch.rand(real.size(0), 1, dim, dim).to(device) #cnn
    sigma = sigma.expand(real.size()).to(device)
    real = real.view(real.size(0), -1)
    fake = fake.view(fake.size(0), -1)
    sigma = sigma.view(sigma.size(0), -1)
    x_hat = (sigma * real + (torch.tensor(1.) - sigma) * fake).to(device)
    x_hat.requires_grad = True
    d_x_hat = D(x_hat).to(device)
    gradients = torch.autograd.grad(outputs=d_x_hat, inputs=x_hat,
                                    grad_outputs=torch.ones(d_x_hat.size()).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0].to(device)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean().to(device)
    return gradient_penalty

#wgan-gp
def train(
        dataset,
        loader,
        dim,
        device,
        epoch_num: int = 100,
        lambda_gp = 0.5,
        d_epoch = 20,
        d_lr = 0.001,
        g_lr = 0.001

):
    '''

    :param dataset: event log name for storing
    :param loader: train dataloader
    :param dim: n_act, number of unique activities
    :param device: "cuda" or "cpu"
    :param epoch_num: 30/50/100...
    :param lambda_gp: strength of the gradient penalty term in the loss function
    :param d_epoch: epoch of Discriminator more than Generator
    :param d_lr: Discriminator learning rate
    :param g_lr: Generator learning rate
    :return: generator, discriminator, g_path, d_path
    '''

    dim = dim+1
    data_dim = dim * dim
    g = Generator(input_dim=data_dim, output_dim=data_dim).to(device)
    d = Discriminator(input_dim=data_dim).to(device)
    optimizer_G = torch.optim.Adam(g.parameters(), lr=g_lr)
    optimizer_D = torch.optim.Adam(d.parameters(), lr=d_lr)
    g_losses = []
    d_losses = []
    # Training loop
    for epoch in range(epoch_num):
        g.train()
        d.train()
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader): # Train d multiple times
            batch_size = len(x)
            x, label, adj_tar = x.to(device), label.to(device), adj_tar.to(device)
            for _ in range(d_epoch):
                z = torch.randint(0, 2, (batch_size, 1, dim, dim)).float().to(device) #cnn
                generated_data = g(z)
                # Discriminator loss for real data
                real_output = d(adj_tar)
                # Discriminator loss for fake data
                fake_output = d(generated_data.detach())
                wasserstein_loss = real_output.mean() - fake_output.mean()
                gradient_penalty = cal_gradient_penalty(d, device, adj_tar, generated_data.detach(), dim) #cnn
                # Total discriminator loss
                d_loss = -wasserstein_loss + lambda_gp * gradient_penalty
                d_losses.append(d_loss.item())
                # Backpropagation and optimization
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            # Generator loss
            z = torch.randint(0, 2, (batch_size, 1, dim, dim)).float().to(device) #cnn
            generated_data = g(z)
            g_loss = -d(generated_data).mean()
            g_losses.append(g_loss.item())
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()
            if ((epoch + 1) % 1 == 0) and (batch % 60 == 0):
                print(
                    f"Epoch [{epoch + 1}/{epoch_num}] | Generator Loss: {g_loss.item():.4f} | Discriminator Loss: {d_loss.item():.4f}")
    g_topath = "g_" + str(epoch_num) + "e_" + dataset + ".pt"
    d_topath = "d_" + str(epoch_num) + "e_" + dataset + ".pt"
    g_path = os.path.join(g_topath)
    d_path = os.path.join(d_topath)
    torch.save(g.state_dict(), g_path)
    torch.save(d.state_dict(), d_path)
    return g, d, g_topath, d_topath


def train_aug(
        loader,
        dim,
        device,
        g_loc,
        p_loc,
        epoch_num: int = 30,
        lr = 0.005,
        n_critic: int = 500,
        criterion=torch.nn.CrossEntropyLoss(),
        num_samples = 5
):
    '''

    :param loader:
    :param dim:
    :param device:
    :param g_loc: generator model
    :param p_loc: predictor model
    :param epoch_num:
    :param lr:
    :param n_critic:
    :param criterion:
    :param num_samples: number of samples within each batch
    :return: predictor
    '''
    dim = dim + 1
    data_dim = dim * dim

    g = Generator(input_dim=data_dim, output_dim=data_dim).to(device)
    g_path = os.path.join(g_loc)
    if os.path.exists(g_path):
        loaded_paras = torch.load(g_path)
        g.load_state_dict(loaded_paras)
        g.eval()

    p = Predictor(dim).to(device)
    p_path = os.path.join(p_loc)
    if os.path.exists(p_path):
        loaded_paras = torch.load(p_path)
        p.load_state_dict(loaded_paras)
        p.eval()
    optimizer = torch.optim.Adam(p.parameters(), lr=lr)

    losses = []
    accs = []

    for epoch in range(epoch_num):
        correct = 0
        total = 0
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader):
            x, label, adj_tar = x.to(device), label.to(device), adj_tar.to(device)
            batch_size = len(x)

            real = torch.argmax(label, dim=1)
            index = torch.mode(real).values
            tolabel = torch.zeros_like(label[0])
            tolabel = tolabel.index_fill(0, index, 1)

            synthetic_data = []
            synthetic_label = []

            with torch.no_grad():
                for _ in range(num_samples):
                    # Generate random noise as input to the GAN generator
                    # z = torch.randint(0, 2, (1, dim, dim)).float().to(device)
                    z = torch.randint(0, 2, (1, 1, dim, dim)).float().to(device)  # cnn
                    synthetic_sample = g(z)
                    synthetic_data.append(synthetic_sample)
                    synthetic_label.append(tolabel)

            # Convert the list of synthetic samples to a PyTorch tensor
            synthetic_data = torch.cat(synthetic_data, dim=0) #5,17*17
            synthetic_label = torch.stack(synthetic_label, dim=0)#5, 17

            synthetic_data = synthetic_data.view(synthetic_data.shape[0], 1, dim, dim)
            x = torch.cat((synthetic_data, x), 0).to(device)
            x.requires_grad

            label = torch.cat((synthetic_label, label), 0).to(device)
            y_hat = p(x)

            loss = criterion(y_hat, label)
            losses.append(loss.item())
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
            accuracy = correct / total
            accs.append(accuracy)

    print("accuracy: {:.4f}".format(np.mean(accuracy)))

    return p



#新 filter
def retrain(
        loader,
        dim,
        d_loc,
        p_loc,
        device,
        epoch_num: int = 30,
        lr: int = 0.001,
        n_critic: int = 500,
        criterion=torch.nn.CrossEntropyLoss(),
):
    '''

    :param loader:
    :param dim:
    :param d_loc:
    :param p_loc:
    :param device:
    :param epoch_num:
    :param lr:
    :param n_critic:
    :param criterion: loss function
    :return: predictor
    '''
    correct = 0
    total = 0

    dim = dim + 1
    data_dim = dim * dim

    losses = []
    accs = []

    d = Discriminator(input_dim = data_dim).to(device)
    d_path = os.path.join(d_loc)
    if os.path.exists(d_path):
        loaded_paras = torch.load(d_path)
        d.load_state_dict(loaded_paras)
        d.eval()
    p = Predictor(dim).to(device)
    p_path = os.path.join(p_loc)
    if os.path.exists(p_path):
        loaded_paras = torch.load(p_path)
        p.load_state_dict(loaded_paras)
        p.eval()
    optimizer = torch.optim.Adam(p.parameters(), lr=lr)
    for epoch in range(epoch_num):
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader):
            x, label, adj_tar = x.to(device), label.to(device), adj_tar.to(device)
            y_hat = p(x)
            gen_next = torch.argmax(y_hat, dim=1).to(device)
            with torch.no_grad():
                flows = copy.deepcopy(cur_flow)
                for curs, next in zip(flows, gen_next):
                    curs.append(next)
                adj_input = []
                for flow in flows:
                    adj_input.append(plain_adj(flow, dim))
                adj_input = torch.from_numpy(np.array(adj_input)).to(device)
                adj_input = adj_input.float()
                adj_input = adj_input.view(adj_input.shape[0], 1, data_dim)
                flag = d(adj_input)
            loss = criterion(y_hat, label) - torch.sum(torch.log(flag + 1e-10))
            losses.append(loss.item())
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
            accuracy = correct / total
            accs.append(accuracy)

    print("accuracy: {:.4f}".format(np.mean(accuracy)))
    return p

