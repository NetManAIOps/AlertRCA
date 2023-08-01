import networkx as nx
from sklearn import preprocessing
import numpy as np
# import matplotlib.pyplot as plt
import torch
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def deep_rca_rank(G: nx.DiGraph, deep_rule, limited_start_set=None, max_iter=5, 
                  end_weight_initial=0.1, end_weight_anneal=2, use_entity=True,
                  use_start=False, weight='weight', debug=None, **kwargs):
    if len(G) == 0:
        return {}

    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G

    def bfs(start_set):
        visited = set()
        current_list = []
        for n in start_set:
            current_list.append(n)
            visited.add(n)
        while len(current_list) > 0:
            n = current_list.pop()
            for nbr in D[n]:
                if nbr not in visited:
                    current_list.append(nbr)
                    visited.add(nbr)
        # print(f"bfs: {visited}")
        return list(visited)

    # service entrance
    if limited_start_set is not None:
        limited_nodes = bfs(limited_start_set)
        D = D.subgraph(limited_nodes)
    
    # step1: meta->cpgat->hidden
    for i, n in enumerate(D):
        D.nodes[n]['meta_vector'] = deep_rule.get_node_vector(n, D.nodes[n]['details'])

    x = [D.nodes[n]['meta_vector'] for i, n in enumerate(D)]
    x = torch.stack(x, dim=0)
    eg_index = nx.to_scipy_sparse_matrix(D, format='coo')
    eg_index = torch.from_numpy(
            np.asarray([eg_index.row, eg_index.col])).long().to(device)
    res = deep_rule.gnn['down'](x, eg_index)
    for i, n in enumerate(D):
        D.nodes[n]['hidden_vector'] = res[i]
    
    # step2: hidden->attention
    for u, v, d in D.edges(data='direction'):
        pu = D.nodes[u]['hidden_vector']
        pv = D.nodes[v]['hidden_vector']
        D[u][v][weight] = deep_rule.predict_likelihood(pu, pv, direction=d) 
        # use hidden vector to get attention

    end_weights = {}
    for n in D:
        end_weights[n] = deep_rule.predict_likelihood(
            D.nodes[n]['hidden_vector'], D.nodes[n]['hidden_vector'],
        )
    
    for n in D:
        nbr_list = []
        w_list = []
        for nbr in D[n]:
            nbr_list.append(nbr)
            w_list.append(D[n][nbr][weight])
        w_list.append(end_weights[n])
        w_list = torch.nn.functional.softmax(torch.stack(w_list), dim=0)
        for i, nbr in enumerate(nbr_list):
            D[n][nbr][weight] = w_list[i]
        end_weights[n] = w_list[-1]  
    
    def normalize(d):
        n_list = []
        d_list = []
        for n in d:
            n_list.append(n)
            d_list.append(d[n])
        d_list = torch.nn.functional.softmax(torch.stack(d_list), dim=0)
        for i, n in enumerate(n_list):
            d[n] = d_list[i]
        return d

    def forward_cwx(D, self_weights, start_probability=None):
        if start_probability is None:
            print('forward error: no start probability')
        x = start_probability
        end_weight = end_weight_initial
        for each_iter in range(max_iter):
            xlast = x
            x = dict.fromkeys(D, None)
            for n in D:
                for nbr in D[n]:
                    if x[nbr] is not None:
                        x[nbr] += xlast[n] * D[n][nbr][weight]
                    else:
                        x[nbr] = xlast[n] * D[n][nbr][weight]
            for n in D:
                if x[n] is not None:
                    x[n] = torch.nn.functional.elu(deep_rule.transform(x[n]))
                
                if each_iter < max_iter - 1:
                    if x[n] is None:
                        x[n] = start_probability[n].clone()
                    else:
                        x[n] += start_probability[n]
        
        res = dict.fromkeys(D, None)
        for n in D:
            if x[n] is None:
                res[n] = deep_rule.predict_root_cause(
                    torch.cat(
                        [
                            D.nodes[n]['meta_vector'], 
                            torch.zeros_like(D.nodes[n]['meta_vector'], device=device)
                        ], dim=-1
                    ) * self_weights[n]
                )
            else:
                res[n] = deep_rule.predict_root_cause(
                    torch.cat([D.nodes[n]['meta_vector'], x[n]], dim=-1) * self_weights[n]
                )
        return normalize(res)

    start_probability = {}
    for n in D:
        start_probability[n] = D.nodes[n]['meta_vector']
    return forward_cwx(D, end_weights, start_probability)
