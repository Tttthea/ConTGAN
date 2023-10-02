import copy
import torch
import torch.nn.functional as F
from helper import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
from sklearn.decomposition import PCA
from GAN import Generator, Discriminator
import os
from torch.utils.data import DataLoader
from MyDataset import MyDataset

def test(
        loader,
        g,
        device,
        dim
):

    correct = 0
    total = 0

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader):
            x, label, adj_tar = x.to(device), label.to(device), adj_tar.to(device)
            next_act = torch.IntTensor(next_act).to(device)
            flows = copy.deepcopy(cur_flow)
            # next_prob = g(hidden_x)
            next_prob = g(x) #cnn
            gen_next = torch.argmax(next_prob, dim=1).to(device)
            for curs, next in zip(flows, gen_next):
                curs.append(next)
            total += label.size(0)
            correct += (gen_next == next_act).sum().item()

            predictions.extend(gen_next.tolist())
            true_labels.extend(next_act.tolist())

            if batch % 10000 == 0:
                print(flows)
                print(next_act)
                print(gen_next)

    acc = float(correct) / total
    print("accuracy: {:.4f}".format(acc))

    # Calculate and print precision
    precision = precision_score(true_labels, predictions, average='weighted')
    print(f'Precision: {precision:.4f}')

    # Calculate and print recall
    recall = recall_score(true_labels, predictions, average='weighted')
    print(f'Recall: {recall:.4f}')

    # Calculate and print F1-score
    f1 = f1_score(true_labels, predictions, average='weighted')
    print(f'F1-Score: {f1:.4f}')

    # print(torch.unique(torch.tensor(true_labels), return_counts=True))
    return acc


def statistics(
        graphs,
        g_loc,
        dim,
        device,
        num_samples=500,
):
    dim = dim + 1
    data_dim = dim * dim

    g = Generator(input_dim=data_dim, output_dim=data_dim).to(device)
    g_path = os.path.join(g_loc)
    if os.path.exists(g_path):
        loaded_paras = torch.load(g_path)
        g.load_state_dict(loaded_paras)
        g.eval()

    gan_samples = []
    original_samples = []

    graphs = graphs[:num_samples]
    data = MyDataset(graphs)
    loader = DataLoader(dataset=data, batch_size=500, shuffle=False,
                             collate_fn=lambda batch: my_collate(batch, dim=dim-1))

    with torch.no_grad():
        for batch, (x, label, adj_tar, cur_flow, next_act) in enumerate(loader):
            batch_size = len(x)
            original_samples.append(adj_tar.view(adj_tar.size(0), -1))#225
            # z = torch.randint(0, 2, (1, dim, dim)).float().to(device)
            z = torch.randint(0, 2, (batch_size, 1, dim, dim)).float().to(device)  # cnn
            generated_sample = g(z)
            # generated_sample = torch.where(generated_sample > 0.993, 1.0, 0.0)
            # a = generated_sample.view(generated_sample.size(0), 1, dim, dim)
            gan_samples.append(generated_sample)


    gan_samples = torch.cat(gan_samples, dim=0)
    original_samples = torch.cat(original_samples, dim=0)

    #element-wise diff
    dif = gan_samples - original_samples
    dif_mean = dif.mean(dim=0)
    print(dif_mean.size()) #17*17

    #heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(dif, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Difference')
    plt.title('Element-wise Differences Between Original Samples And Synthetic Samples')
    plt.xlabel('features')
    plt.ylabel('Samples')
    plt.show()

    return


