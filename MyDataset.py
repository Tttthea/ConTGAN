import os
import pandas as pd
import torch


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, transform=None, target_transform=None):
        self.graphs = graphs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs.iloc[idx]['graphs']
        graph = torch.Tensor(graph)
        label = self.graphs.iloc[idx]['onehot_target']
        label = torch.Tensor(label)
        adj_tar = self.graphs.iloc[idx]['adj_target']
        adj_tar = torch.Tensor(adj_tar)
        cur_flow = self.graphs.iloc[idx]['cur_flow']
        next_act = self.graphs.iloc[idx]['target']
        if self.transform:
            graph = self.transform(graph)

        return graph, label, adj_tar, cur_flow, next_act
