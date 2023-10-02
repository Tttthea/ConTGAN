import numpy as np
import pandas as pd
from collections import defaultdict
from helper import *
pd.set_option('display.max_columns', None)


def encoding2(log):
    #no time
    n_case = log['case:concept:name'].nunique()  # 1050
    n_act = log['concept:name'].nunique()  # 16
    act = log.groupby('case:concept:name', sort=False).agg({'concept:name': lambda x: list(x)})
    act['order'] = act.apply(lambda x: order_in_case(x), axis=1)
    indices = act.explode("order")
    act['graphs'] = act.apply(lambda x: build_graph3(x, n_act), axis=1)
    graphs = act.explode("graphs") #14164
    graphs['tar'] = (graphs['concept:name'] == (graphs.shift(-1)['concept:name'])).apply(lambda x: if_still(x))
    act['list_no_first'] = act['concept:name'].apply(lambda x: x[1:])
    graphs['tar_first'] = (graphs['concept:name'] == (graphs.shift(1)['concept:name'])).apply(lambda x: if_first(x))
    cur_last = act.explode("list_no_first")
    next_last = cur_last.shift(-1)['list_no_first']
    next_last[-1] = n_act
    graphs['cur_last_act'] = cur_last['list_no_first']
    graphs['next_last'] = next_last
    graphs['target'] = graphs.apply(lambda x: shifte(x, n_act), axis=1) #visual_next_act

    graphs['onehot_target'] = graphs['target'].apply(lambda x: onehot(x, n_act)) # computed in Generator
    graphs['indices'] = indices['order']

    graphs['cur_flow'] = graphs.apply(lambda x: x['concept:name'][:int(x['indices'])+1] if pd.notna(x['indices']) else 0, axis=1) #visual_cur_acts
    graphs['shifted'] = graphs['graphs'].shift(-1)

    last_line = graphs.iloc[-1]['graphs']
    graphs['shifted'].iloc[-1] = last_line
    graphs['adj_target'] = graphs.apply(lambda x: build_target(x), axis=1) # computed in Discriminator
    graphs = graphs[graphs['adj_target'].notna()]
    graphs['adj_target'] = graphs.apply(lambda x: label_y(x, n_act), axis=1)

    return graphs

