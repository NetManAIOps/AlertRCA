import os
import copy
import pandas as pd
import numpy as np
import networkx as nx
from yaml import load, CLoader as Loader

from itertools import product, groupby
from pathlib import Path
from pprint import pformat
from typing import Union, Optional, Any, Dict, List
import logging as logger
from typing import Tuple, Dict, List, Set, Union, Callable, Optional

import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split

def forward_fill_na(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    if axis < 0:
        axis = len(arr.shape) + axis
    idx_shape = tuple([slice(None)] + [np.newaxis] * (len(arr.shape) - axis - 1))
    idx = np.where(~np.isnan(arr), np.arange(arr.shape[axis])[idx_shape], 0)
    np.maximum.accumulate(idx, axis=axis, out=idx)
    slc = [np.arange(k)[tuple([slice(None) if dim == i else np.newaxis
                               for dim in range(len(arr.shape))])]
           for i, k in enumerate(arr.shape)]
    slc[axis] = idx
    return arr[tuple(slc)]

def insert_global_params(keys: List[str], global_params: Dict[str, Any], target: Dict[str, Any]):
    for key in keys:
        assert key in global_params
        target[key] = global_params[key]
    return target

def parse_yaml_graph_config(path: Union[str, Path]) -> nx.DiGraph:
    def metrics_sorted(metrics):
        return sorted(metrics, key=lambda _: _.split("##")[1])

    path = Path(path)
    logger.debug(f"parsing Graph from {path!s}")
    with open(path) as f:
        input_data = load(f, Loader=Loader)
    g = nx.DiGraph()
    global_params = {}
    for obj in input_data:
        if obj.get('class', "") == "global_params":
            global_params = obj
            break
    # parse nodes in the first pass
    for obj in input_data:
        try:
            if obj['class'] != "node":
                continue
            if "params" in obj or "global_params" in obj:
                params = obj.get("params", {})
                insert_global_params(obj.get('global_params', []), global_params, params)
                keys = list(params.keys())
                if obj.get('product', False):
                    for values in product(*params.values()):
                        node_id = obj["id"].format(**{k: v for k, v in zip(keys, values)})
                        node_metrics = [_.format(**{k: v for k, v in zip(keys, values)}) for _ in obj["metrics"]]
                        g.add_node(node_id, **{"metrics": metrics_sorted(node_metrics), "type": obj["type"]})
                else:
                    values_list = list(zip(*params.values()))
                    for values in values_list:
                        node_id = obj["id"].format(**{k: v for k, v in zip(keys, values)})
                        node_metrics = [_.format(**{k: v for k, v in zip(keys, values)}) for _ in obj["metrics"]]
                        g.add_node(node_id, **{"metrics": metrics_sorted(node_metrics), "type": obj["type"]})
            else:
                g.add_node(obj['id'], **{"metrics": metrics_sorted(obj['metrics']), "type": obj["type"]})
        except KeyError as e:
            logger.error(f"{e!r} obj={pformat(obj)}")

    def add_edge(_src, _dst, _attrs):
        _attrs['direction'] = 'd' # for AlertRCA
        if _src in g and _dst in g:  # ignore edges that don't exist
            g.add_edge(_src, _dst, **_attrs)
        else:
            logger.warning(f"ignoring edge {_src} -> {_dst}")

    # parse edges in the second pass
    for obj in input_data:
        try:
            if obj['class'] != "edge":
                continue
            if "params" in obj or "global_params" in obj:
                params = obj.get('params', {})
                insert_global_params(obj.get('global_params', []), global_params, params)
                keys = list(params.keys())
                if obj.get('product', False):
                    for values in product(*params.values()):
                        src = obj["src"].format(**{k: v for k, v in zip(keys, values)})
                        dst = obj["dst"].format(**{k: v for k, v in zip(keys, values)})
                        add_edge(src, dst, {"type": obj["type"]})
                else:
                    values_list = list(zip(*params.values()))
                    for values in values_list:
                        src = obj["src"].format(**{k: v for k, v in zip(keys, values)})
                        dst = obj["dst"].format(**{k: v for k, v in zip(keys, values)})
                        add_edge(src, dst, {"type": obj["type"]})
            else:
                add_edge(obj["src"], obj["dst"], {"type": obj["type"]})
        except KeyError as e:
            logger.error(f"{e!r} obj={pformat(obj)}")
    if not len(list(nx.weakly_connected_components(g))) <= 1:
        logger.warning(f"{path!s} is not a DAG: {list(nx.weakly_connected_components(g))=}")
    return g

def read_faults(datadir):
    df = pd.read_csv(os.path.join(datadir, 'faults.csv'))
    return df

def read_graph(datadir, graphfile=None):
    if graphfile is None:
        yml_file = os.path.join(datadir, 'graph.yml')
    else:
        yml_file = os.path.join(datadir, graphfile)
    return parse_yaml_graph_config(yml_file)

def read_metrics(datadir):
    df = pd.read_csv(os.path.join(datadir, 'metrics.norm.csv'))
    return df

def get_metric(mdf, kpi_id):
    return mdf[mdf['name']==kpi_id].sort_values(by=['timestamp']).reset_index(drop=True)

class failure_graph():
    def __init__(
        self,
        graph: nx.DiGraph
    ):
        self.graph = graph
        self._node_list: Dict[str, List[str]] = dict(map(
            lambda pair: (pair[0], [_[0] for _ in pair[1]]),
            groupby(sorted(
                graph.nodes(data=True), key=lambda _: _[1]['type']
            ), key=lambda _: _[1]['type'])
        ))
        self._node_to_idx: Dict[str, Dict[str, int]] = {
            node_type: {node: idx for idx, node in enumerate(node_list)}
            for node_type, node_list in self._node_list.items()
        }
        self._node_to_class: Dict[str, str] = dict(sum(
            [[(node, fc) for node in nodes] for fc, nodes in self._node_list.items()],
            []
        ))
        
        self.failure_classes = []
        self.failure_class_to_id = {}
        for fc in self._node_list:
            self.failure_classes.append(fc)
        self.failure_classes.sort()
        for i, fc in enumerate(self.failure_classes):
            self.failure_class_to_id[fc] = i
        
        self._node_metrics_dict = {}
        self.metric_size_dict = {}
        for node, data in graph.nodes(data=True):
            self._node_metrics_dict[node] = data['metrics']
            self.metric_size_dict[data['type']] = len(data['metrics'])
            
        self._node_to_gid = {}
        for fc in self.failure_classes:
            for node in self._node_list[fc]:
                self._node_to_gid[node] = len(self._node_to_gid)

class feature_metrics():
    def __init__(
        self,
        metrics: pd.DataFrame,
        graph: failure_graph,  # compose_all(failure_graphs)
        start_ts: int, 
        end_ts: int,
        granularity: int = 60,
    ):
        self._granularity = granularity
        self._features_list, self._timestamp_2_idx = self.extract_features(
            metrics,
            graph,
            start_ts,
            (end_ts - start_ts) // granularity + 1,
            granularity
        )
    
    def extract_features(
        self,
        metrics: pd.DataFrame,
        graph: failure_graph,
        start_ts: int,
        length: int,
        granularity: int = 60,
        clip_value: float = 10.
    ):
        the_df = metrics[
            (metrics.timestamp >= start_ts) &
            (metrics.timestamp < start_ts + length * granularity)
        ]
        metric_mean_dict = metrics.groupby('name')['value'].mean().to_dict()
        
        timestamp_list = [start_ts + i * granularity for i in range(length)]
        timestamp_2_idx = {ts: idx for idx, ts in enumerate(timestamp_list)}
        features_list = []
        
        for failure_class in tqdm(graph.failure_classes, desc = 'preprocess metrics for all failure classes'):
            metric_2_node_idx = {}
            metric_2_metric_idx = {}
            node_type_metrics = set()
            
            for i, instance in enumerate(graph._node_list[failure_class]):
                for j, metric in enumerate(graph._node_metrics_dict[instance]):
                    metric_2_node_idx[metric] = i
                    metric_2_metric_idx[metric] = j
                    node_type_metrics.add(metric)
                    
            _feat = np.full(
                (
                    len(graph._node_list[failure_class]), #node
                    graph.metric_size_dict[failure_class], #metric
                    length
                ), 
                float('nan')
            )
            
            _df = the_df.loc[
                the_df.name.isin(node_type_metrics) & the_df.timestamp.isin(timestamp_list),
                ['timestamp', 'value', 'name']
            ]
            
            if (len(_df) > 0):
                _df['idx0'] = _df['name'].map(lambda _: metric_2_node_idx[_])
                _df['idx1'] = _df['name'].map(lambda _: metric_2_metric_idx[_])
                _df['idx2'] = _df.timestamp.map(lambda _: timestamp_2_idx[_])
                _feat[_df.idx0.values, _df.idx1.values, _df.idx2.values] = _df['value'].values
            
            _feat = forward_fill_na(_feat, axis=-1)
            for i, j in zip(*np.where(np.any(np.isnan(_feat), axis=-1))):         
                metric_name = graph._node_metrics_dict[graph._node_list[failure_class][i]][j]
                np.nan_to_num(
                    _feat[i, j, :],
                    copy=False,
                    nan=metric_mean_dict.get(metric_name, -10)
                )
            assert np.all(np.isfinite(_feat))
            _feat = np.clip(_feat, -clip_value, clip_value)
            
            features_list.append(torch.from_numpy(_feat).float())
        
        return features_list, timestamp_2_idx
    
    def get_timestamp_indices(
        self, 
        fault_ts: int, 
        window_size: Tuple[int, int]
    ):
        start_ts = fault_ts - window_size[0] * self._granularity
        length = sum(window_size)
        timestamp_list = [start_ts + i * self._granularity for i in range(length)]
        ts_idx = np.asarray([self._timestamp_2_idx[_] for _ in timestamp_list])
        return ts_idx
    
    def get_features_slice(
        self, 
        fault_ts: int, 
        window_size: Tuple[int, int] = (10, 10), 
        batch_normalization: bool = True
    ):
        if batch_normalization:
            def batch_rescale(feat):
                return feat - torch.nanmean(feat[..., :-window_size[1]], dim=-1, keepdim=True)
        else:
            def batch_rescale(feat):
                return feat

        ts_idx = self.get_timestamp_indices(fault_ts, window_size)

        features_list = [batch_rescale(_[..., ts_idx]) for _ in self._features_list]
        return features_list

class graph_dataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        data_ids: List[int],
        failures: pd.DataFrame,
        metrics: pd.DataFrame,
        graphs: List[nx.DiGraph],        
        window_size: Tuple[int, int] = (10, 10),
        granularity: int = 60,
    ):
        cgraph = nx.compose_all(graphs)
        self.cgraph = failure_graph(cgraph)
        
        self.metrics = feature_metrics(
            metrics = metrics,
            graph = self.cgraph,  
            start_ts = min(failures.timestamp) - 20 * granularity,
            end_ts =  max(failures.timestamp) + 20 * granularity,
            granularity = granularity
        )
        
        self.data_ids = data_ids
        self.failures = failures
        self.graphs = graphs
        self.window_size = window_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def __len__(self):
        return len(self.data_ids)
    
    def __getitem__(self, index):
        f_idx = self.data_ids[index]
        fault = dict(self.failures.loc[f_idx])
        graph = copy.deepcopy(self.graphs[f_idx])
        
        ori_features = self.metrics.get_features_slice(
            fault['timestamp'], 
            self.window_size
        ) # list of (#node, #metric, #time range)
        
        labels = fault['root_cause_node'].split(';')
        
        for i, n in enumerate(graph):
            tp = self.cgraph._node_to_class[n]
            graph.nodes[n]['details'] = {
                'type': tp,
                'metric': ori_features[self.cgraph.failure_class_to_id[tp]][self.cgraph._node_to_idx[tp][n]].to(self.device)
            }
            
        result = labels, fault['timestamp'], graph
        return result

def get_dejavu_dataset(datadir, ratio=0.5, graph_list=False, upsample=15):
    fdf = read_faults(datadir)
    mdf = read_metrics(datadir)
    if not graph_list:
        nxg = read_graph(datadir)
        nxg_list = [nxg.copy() for i in range(len(fdf))]
    else:
        nxg_list = []
        for i in range(len(fdf)):
            t = fdf.loc[i, 'timestamp']
            nxg_list.append(read_graph(datadir, f'graphs/graph_{t}.yml'))
    
    if not('node_type' in fdf):
        node_types = []
        for i in range(len(fdf)):
            node_types.append(nxg_list[i].nodes[fdf.loc[i, 'root_cause_node']]['type'])
        fdf['node_type'] = node_types
    
    
    out_d = {}
    for i in range(len(fdf)):
        tp = fdf.loc[i, 'node_type'].split(';')[0]
        out_d.setdefault(tp, 0)
        out_d[tp] += 1
    
    ids = []
    tps = []
    for i in range(len(fdf)):
        tp = fdf.loc[i, 'node_type'].split(';')[0]
        ids.append(i)
        if (out_d[tp] > 1):
            tps.append(tp)
        else:
            tps.append('outsider')
        
    # split dataset
    # print(ids)
    # print(tps)

    x_train, x_test, y_train, y_test = train_test_split(
        ids, tps,
        test_size=1-ratio, 
        random_state=1234, 
        stratify=tps
    )

    # create train_dataset
    
    type_d = {}
    for i in x_train:
        ntp = tps[i]
        type_d.setdefault(ntp, 0)
        type_d[ntp] += 1
    
    ids = []
    remains_d = {}
    for i in x_train:
        ntp = tps[i]
        if (type_d[ntp] >= upsample):
            ids.append(i)
        else:
            times = upsample // type_d[ntp]
            remains = upsample % type_d[ntp]
            ids.extend([i] * times)
            if remains_d.get(ntp, 0) < remains:
                ids.append(i)
            remains_d.setdefault(ntp, 0)
            remains_d[ntp] += 1
    
    train_csv = pd.read_csv(os.path.join(datadir, 'train.csv'))
    valid_csv = pd.read_csv(os.path.join(datadir, 'valid.csv'))
    test_csv = pd.read_csv(os.path.join(datadir, 'test.csv'))

    train_dataset = graph_dataset(
        list(train_csv['id']),
        fdf,
        mdf,
        nxg_list,
    )

    valid_dataset = graph_dataset(
        list(valid_csv['id']),
        fdf,
        mdf,
        nxg_list
    )

    test_dataset = graph_dataset(
        list(test_csv['id']),
        fdf,
        mdf,
        nxg_list
    )

    return train_dataset, valid_dataset, test_dataset