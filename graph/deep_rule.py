import os
import torch.nn as nn
import torch.nn.functional as F
# import groot.conf.constants as const
from dejavu_dataloader.models import GRUFeatureModule
import numpy as np
# from transformers import BertTokenizer, BertModel
import torch
# import gensim.downloader

# from .event_graph_search import DeepEventGraph
import networkx as nx

from .rca_algo import deep_rca_rank

from torch_geometric.nn import SAGEConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 500
batch_size = 16
learning_rate = 4e-3
clip_value = 1.0
attention_clip = 0.1
drop_p = 0.2
# lr_epoch = min(20, num_epochs * 0.1)
lr_epoch = 10000

class RuleDict(nn.Module):
    def __init__(self, catalog, num_features, get_node_vector, hid_channels=12,
                 attention_channels=12, heads=4,
                 attention_type='dagat', negative_slope=0.2, **kwargs):
        super(RuleDict, self).__init__()
        self.catalog = catalog
        self.hid_channels = hid_channels
        self.attention_channels = attention_channels
        if attention_type == 'dagat':
            self.heads = heads
        else:
            self.heads = 1
        self.attention_type = attention_type

        self.deep_transformer = torch.nn.Sequential(
            torch.nn.Linear(in_features=hid_channels, out_features=hid_channels, bias=False)
        )
        nn.init.xavier_uniform_(self.deep_transformer[0].weight, gain=1.414)

        self.start_embedding = torch.nn.Sequential(
            torch.nn.Linear(in_features=hid_channels * 2, out_features=hid_channels, bias=False)
        )
        nn.init.xavier_uniform_(self.start_embedding[0].weight, gain=1.414)

        self.root_cause_fc = torch.nn.Sequential(
            # torch.nn.Dropout(p=drop_p),
            torch.nn.Linear(in_features=hid_channels * 2, out_features=64),
            torch.nn.GELU(),
            # torch.nn.Dropout(p=drop_p),
            torch.nn.Linear(in_features=64, out_features=1, bias=False)
        )

        self.alpha_w_q = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hid_channels,
                out_features=attention_channels * self.heads, 
                bias=False)
        )
        self.alpha_w_k = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hid_channels,
                out_features=attention_channels * self.heads, 
                bias=False)
        )
        self.alpha_w_v = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hid_channels,
                out_features=hid_channels * self.heads, 
                bias=False)
        )
        if self.attention_type == 'add':
            self.attention_vector = torch.nn.Parameter(torch.Tensor(1, heads, attention_channels))
        else:
            self.attention_vector = torch.nn.Parameter(torch.Tensor(1, heads, attention_channels * 2))
        
        self.scale = np.sqrt(attention_channels)
        
    def map(self, x):
        if self.attention_type in ['dot', 'scale_dot', 'add', 'concat']:
            Q = K = V = x.unsqueeze(1).repeat([1, self.heads, 1])
        elif self.attention_type == 'bilinear':
            Q = V = x.unsqueeze(1).repeat([1, self.heads, 1])
            K = self.alpha_w_k(x).view(1, self.heads, self.attention_channels)
        elif self.attention_type in ['transformer', 'dagat']:
            Q = self.alpha_w_q(x).view(1, self.heads, self.attention_channels)
            K = self.alpha_w_k(x).view(1, self.heads, self.attention_channels)
            V = self.alpha_w_v(x).view(1, self.heads, self.hid_channels)
        return Q, K, V

    def attention(self, a, b):
        a = a.unsqueeze(0)
        b = b.unsqueeze(0)
        # print(a, b)
        Qa, Ka, Va = self.map(a)
        Qb, Kb, Vb = self.map(b)
        # print(f"Qa: {Qa.size()}, Kb: {Kb.size()}")
        a = Qa
        b = Kb
        # print(a, b)

        # ['dot', 'add', 'concat', 'scale_dot', 'bilinear', 'transformer']
        if self.attention_type in ['dot', 'bilinear']:
            return (a * b).sum(dim=-1)  # [edge_num, heads]
        if self.attention_type == 'add':
            return (self.attention_vector * (a + b).tanh()).sum(dim=-1)
        if self.attention_type == 'concat':
            return (self.attention_vector * torch.cat([a, b], dim=-1)).sum(dim=-1)
        if self.attention_type in ['scale_dot', 'transformer', 'dagat']:
            return (a * b).sum(dim=-1) / self.scale # (1, heads)
        return None

    def forward(self, event0, event1, direction='d'):
        '''
            event0: [1, h_dim]
            event1: [1, h_dim]
        
        '''
        attention = self.attention(event0, event1).sum()
        return attention

        attention = torch.clamp(attention, min=-10, max=10)
        # print('attention shape', attention.size())
        minus = 0.5 if self.attention_type == 'dagat' else 0.0
        res = torch.clamp(torch.exp(attention) - minus, min=0.0)
        return res

    def get_start_probability(self, a):
        res = self.start_embedding(a)
        return res

    def get_root_cause(self, a):
        res = self.root_cause_fc(a)
        return res

gnn_cell = SAGEConv
class GNN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GNN, self).__init__()
        self.drop_p = drop_p
        self.conv1 = gnn_cell(num_features, hidden_channels)
        self.conv2 = gnn_cell(hidden_channels, hidden_channels)
        self.conv3 = gnn_cell(hidden_channels, hidden_channels)
        self.conv4 = gnn_cell(hidden_channels, hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, 0.2)
        # x = F.tanh(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.conv2(x, edge_index)
        # x = F.tanh(x)
        # x = F.leaky_relu(x, 0.2)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.conv3(x, edge_index)
        # x = F.tanh(x)
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.conv4(x, edge_index)
        return x

class DeepRule:
    def __init__(self, catalog, rca_parameters, token_type='bert'): 
        print('In deep rule, device is:', device)
        print('lr:', learning_rate)
        print('bs:', batch_size)

        if token_type == 'dejavu':
            self.rule_space_size = rca_parameters['hid_channels']
            self.metric_projector = nn.ModuleDict({})
            for k, v in rca_parameters['gru_size'].items():
                self.metric_projector[k] = GRUFeatureModule(v, self.rule_space_size).to(device)
        else:
            raise RuntimeError("No such token type: {}".format(token_type))

        self.catalog = catalog
        self.rule_dict = RuleDict(
            self.catalog, 
            self.rule_space_size, 
            self.get_node_vector,
            **rca_parameters
        ).to(device)

        self.gnn = nn.ModuleDict({})
        self.gnn['down'] = GNN(rca_parameters['hid_channels'], rca_parameters['hid_channels']).to(device)
        self.gnn['up'] = GNN(rca_parameters['hid_channels'], rca_parameters['hid_channels']).to(device)
        self.rca_parameters = rca_parameters

    def mini_deep_groot(self, data, require_res=False, require_eg=False):
        '''
            data = labels, key, EG
            D.nodes[n]['details'] in get_node_vector(self, node, detail):
            
        '''
        lss = {
            'A1': ['os_021', 'os_022'],
            'A2': ['os_021', 'os_022'],
            'B': ['OSB', 'Redis', 'Mysql']
        }

        EG = data[-1]
        key = str(data[1])

        debug = None
        if key == "1592070840":
            debug = key

        if require_eg:
            return EG
        
        if require_res:
            return deep_rca_rank(
                EG, 
                self, 
                limited_start_set = lss[self.catalog],
                max_iter = 4,
                debug = debug,
                **self.rca_parameters
            ) 
            #(result -> reason)
        return True

    def load_models(self, load_model_path):
        if load_model_path is not None and os.path.exists(load_model_path):
            rd = os.path.join(load_model_path, 'rule_dict.pt')
            gn = os.path.join(load_model_path, 'gnn.pt')
            mt = os.path.join(load_model_path, 'metric.pt')
            if os.path.exists(rd) and os.path.exists(gn) and os.path.exists(mt):
                self.rule_dict.load_state_dict(torch.load(rd))
                self.gnn.load_state_dict(torch.load(gn))
                self.metric_projector.load_state_dict(torch.load(mt))
            else:
                return False
        else:
            return False
        return True
    
    def save_models(self, save_model_path):
        if not os.path.exists(save_model_path):
            os.makedirs(save_model_path)
        rd = os.path.join(save_model_path, 'rule_dict.pt')
        gn = os.path.join(save_model_path, 'gnn.pt')
        mt = os.path.join(save_model_path, 'metric.pt')
        torch.save(self.rule_dict.state_dict(), rd)
        torch.save(self.gnn.state_dict(), gn)
        torch.save(self.metric_projector.state_dict(), mt)

    def set_model_train(self):
        self.rule_dict.train()
        self.gnn.train()
        self.metric_projector.train()
    
    def set_model_eval(self):
        self.rule_dict.eval()
        self.gnn.eval()
        self.metric_projector.eval()

    def train(self, training_data, load_model_path=None, save_model_path=None, testing_data=None, valid_data=None):
        '''
            training_data: graph_dataset
        '''
        
        if self.load_models(load_model_path):
            return

        optimizer = torch.optim.Adam(
            filter(
                lambda p: p.requires_grad, 
                list(self.rule_dict.parameters()) + list(self.gnn.parameters()) + list(self.metric_projector.parameters())
            ),
            lr=learning_rate,
            weight_decay=1e-3
        )
        loss = 0
        batch_num = 0
        save_loss = np.inf
        save_acc = (0.0, 0.0, 0.0) # acc3, avgrank, loss
        need_save = True

        for epoch in range(num_epochs):
            self.set_model_train()

            loss_list = []            
            random_index = np.arange(0, len(training_data))
            np.random.shuffle(random_index)
            for i in range(len(training_data)):
                data = training_data[random_index[i]]
                # data = labels, key, graph
                res = self.mini_deep_groot(data, require_res=True)
                if res is not None:
                    if isinstance(res[data[0][0]], torch.Tensor):
                        for lb in data[0]:
                            loss = loss - torch.log(res[lb] + 1e-8)
                            break
                        batch_num = batch_num + 1
                    if batch_num >= batch_size:
                        if isinstance(loss, torch.Tensor):
                            loss = loss / batch_num
                            optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(self.rule_dict.parameters()) + list(self.gnn.parameters()) + list(self.metric_projector.parameters()), 
                                clip_value
                            )
                            optimizer.step()
                            loss_list.append(loss.item())
                        loss = 0
                        batch_num = 0

            print("Epoch[{}/{}], Loss: {:.4f}, lr:{:.4f}".format(
                epoch + 1, num_epochs, 
                np.mean(loss_list),
                optimizer.state_dict()['param_groups'][0]['lr']
            ))

            # lr decay
            if (epoch + 1) % lr_epoch == 0:
                for p in optimizer.param_groups:
                    if p['lr'] >= 1e-4:
                        p['lr'] *= 0.5

            if ((epoch + 1) % 10 == 0) and ((testing_data is not None) or (valid_data is not None)):
                # output valid & test result
                self.set_model_eval()
                if valid_data is not None:
                    valid_loss_list = []
                    res_list = []
                    for i in range(len(valid_data)):
                        data = valid_data[i]
                        res = self.mini_deep_groot(data, require_res=True)
                        valid_loss_list.append(-torch.log(res[data[0][0]] + 1e-8).item())
                        tmp = []
                        for k, v in res.items():
                            tmp.append((k, v.item()))
                        tmp.sort(key=lambda x: -x[1])
                        rank = len(tmp)
                        for lb in data[0]:
                            for j in range(len(tmp)):
                                if lb == tmp[j][0]:
                                    rank = min(rank, j + 1)
                                    break
                        res_list.append(rank)
                    res_list = np.asarray(res_list)
                    valid_loss = np.mean(valid_loss_list)
                    valid_acc = (np.sum(res_list <= 3) / len(res_list), -np.mean(res_list), -valid_loss) # acc3, avgrank, loss
                    print(valid_loss_list)
                    print(res_list)
                    print('valid result: vloss: {:.4f}, vacc3: {:.4f} avgrank: {:.5f}'.format(valid_loss, valid_acc[0], np.mean(res_list)))
                    # if valid_loss < save_loss:
                    #     print('valid loss[{:.4f}] < save loss[{:.4f}], update model'.format(valid_loss, save_loss))
                    #     save_loss = valid_loss
                    if valid_acc > save_acc:
                        print(f'[acc3, -avgrank, -validloss] valid tuple[{valid_acc}] > save tuple[{save_acc}], update model')
                        save_acc = valid_acc
                        if save_model_path is not None:
                            self.save_models(save_model_path)
                            need_save = False
                    
                if testing_data is not None:
                    res_list = []
                    for i in range(len(testing_data)):
                        data = testing_data[i]
                        res = self.mini_deep_groot(data, require_res=True)
                        tmp = []
                        for k, v in res.items():
                            tmp.append((k, v.item()))
                        tmp.sort(key=lambda x: -x[1])
                        rank = len(tmp)
                        for lb in data[0]:
                            for j in range(len(tmp)):
                                if lb == tmp[j][0]:
                                    rank = min(rank, j + 1)
                                    break
                        res_list.append(rank)
                    res_list = np.asarray(res_list)
                    print("Testing result: avgrank: {:.4f}, acc1: {:.4f}, acc2: {:.4f}, acc3: {:.4f}, acc5: {:.4f}".format(
                        np.mean(res_list),
                        np.sum(res_list == 1) / len(res_list),
                        np.sum(res_list <= 2) / len(res_list),
                        np.sum(res_list <= 3) / len(res_list),
                        np.sum(res_list <= 5) / len(res_list)
                    ))
        
        # print('Training is over, best valid loss: {:.4f}'.format(save_loss))
        print(f'Training is over, best valid acc: {save_acc}')
        if (save_model_path is not None) and need_save:
            self.save_models(save_model_path)

    def transform(self, event):
        res = self.rule_dict.deep_transformer(event)
        return res

    def predict_likelihood(self, event0, event1, direction='d'):
        res = self.rule_dict(event0, event1, direction)
        # print('predict_likelihood', res)
        return res

    def predict_start_probability(self, event):
        res = self.rule_dict.get_start_probability(event)
        # print('predict_start_probability', res)
        return res

    def predict_root_cause(self, event):
        res = self.rule_dict.get_root_cause(event)
        # print('predict_root_cause', res)
        return res

    def get_node_vector(self, node, detail):
        '''
            node: str, node id
            {
               'metric': torch.tensor with shape (#metric * #timerange),
               'type': 'OSB'
            }
        '''
        detail_vector = detail['metric'] # (M * W)
        detail_type = detail['type']
        res_vector = self.metric_projector[detail_type](detail_vector)
        return res_vector
