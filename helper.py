import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
import copy

def show_image(x):
    "Show one image as an image."
    plt.imshow(x.reshape(x.size()[0], -1), interpolation="none")
    plt.show()

def to_tensor(x):
    return torch.as_tensor(np.array(x))

def img2Eventlog(img):
    print(img)
    act_list = {}
    if torch.cuda.is_available():
        img = img.cpu()
    img = img.detach().numpy()

    index_sorted = np.argsort(img[:, -1, :, 1])  # -1, -2: [4,5,...8,9000] activity happens order
    index_acts = np.argwhere(img[:, -1, :, 0] > 0)  # [0,4,5] which activities happen
    act_dic = {}
    act_count = {}
    samples = set()
    print(index_acts)
    print(index_sorted)
    for index in index_acts:
        sample = index[0]
        act = index[1]
        if sample not in samples:
            samples.add(sample)
            act_dic[sample] = [act]
        else:
            act_dic[sample].append(act)
    print(act_dic)
    for s in samples:
        act_count[s] = len(act_dic[s])
    print(act_count)
    print("-----")
    for sample, count in act_count.items():
        list = index_sorted[sample, :]
        list = list[-(count-1):][::-1]
        act_list[sample] = list
        miss = np.where(lambda x: x in act_dic[sample] not in act_list[sample])
        act_list[sample] = np.append(act_list[sample], miss)
        act_list[sample] = act_list[sample] + np.ones_like(act_list[sample]) #9, 8, 7, 4, 3, 2, 6, 5, 1]
    print(act_list)
    return act_list




def predict(act):
    act = act.transpose(3, 1)
    act = pd.DataFrame(act.detach().numpy())
    i = 0
    list_label = []
    while i < len(act):
        j = 0
        while j < (len(act.iat[i, 0]) - 1):
            if j > 0:
                list_label.append(act.iat[i, 0][j + 1])
            j = j + 1
        i = i + 1
    return list_label

def diff_time(df):
    l = df['time:timestamp']
    durs = []
    for i in range(len(l)-1):
        dur = int((l[i+1] - l[i]).total_seconds())
        if dur == 0:
            dur += 1
        durs.append(dur)
    return durs

def build_graph(df, n):
    graph = np.zeros((n, n), dtype=int)
    flag = np.zeros((n, n), dtype=int)
    nodes = df['concept:name']
    weights = df['dur']
    for i in range(len(nodes)-1):
        node1 = nodes[i]
        node2 = nodes[i+1]
        if flag[node1][node2] == 1:
            graph[node1][node2] += weights[i]
            graph[node1][node2] /= 2
        else:
            graph[node1][node2] = weights[i]
            flag[node1][node2] = 1
    return graph

def plain_adj(x, n):
    adj = np.zeros((n, n), dtype=int)
    x[-1] = x[-1].item()
    flag = False
    if x[-1] == n:
        x = copy.deepcopy(x[:-1])
        flag = True
    for i in range(len(x) - 1):
        node1 = x[i]
        node2 = x[i + 1]
        adj[node1][node2] = 1
        if flag:
            adj[n-1][n-1] = 1
    return adj

def build_graph2(df, n):
    # plain adjacency
    graph = np.zeros((n, n), dtype=int)
    nodes = df['concept:name']
    for i in range(len(nodes) - 1):
        node1 = nodes[i]
        node2 = nodes[i + 1]
        graph[node1][node2] = 1
    return graph

def build_graph3(df, n):
    # plain adjacency prefix
    graphs = []
    j = 1
    while j < len(df['concept:name']):
        j += 1
        graph = np.zeros((n+1, n+1), dtype=int)
        nodes = df['concept:name'][:j]
        for i in range(len(nodes) - 1):
            node1 = nodes[i]
            node2 = nodes[i + 1]
            graph[node1][node2] = 1
        graphs.append(graph)
    return graphs

def build_target(x):
    #A_i+1 is the target for A_i, A_tao's target is all_1
    if not x['tar']:
        tar = x['graphs']
    else:
        tar = x['shifted']
    return tar

def label_y(x, n):
    tar = copy.deepcopy(x['adj_target'])
    if not x['tar']:
        tar[n, n] = 1
    return tar

def if_still(b):
    tar = False
    if b == True:
        tar = True
    return tar

def onehot(x, n):
    vect = np.zeros(n+1, dtype=int)

    vect[x] = 1
    return vect

def shifte(x, n):
    if not x['tar']:
        tar = n
    else:
        tar = x['next_last']
    if not np.isnan(tar):
        tar = tar
    else:
        tar = 0
    return int(tar)

def if_first(b):
    tar = False
    if b == False:
        tar = True
    return tar

def order_in_case(x):
    length = len(x['concept:name'])
    return np.arange(1, length, dtype=int)

#cnn-predictor
def my_collate(batch, dim):
    batch_size = len(batch)
    data = torch.cat([item[0] for item in batch]).reshape(batch_size, 1, dim+1, dim+1)
    target = torch.cat([item[1] for item in batch]).reshape(batch_size, -1)
    adj_tar = torch.cat([item[2] for item in batch]).reshape(batch_size, 1, dim+1, dim+1)
    cur_flow = [item[3] for item in batch]
    next_act = [item[4] for item in batch]
    return [data, target, adj_tar, cur_flow, next_act]

