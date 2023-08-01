import time
import os
import json
import random
import torch
import numpy as np

import pytz
from os import path
from datetime import datetime

from graph.deep_rule import DeepRule
from graph.rca_algo import deep_rca_rank
import logging
import argparse

from dejavu_dataloader.dataset import get_dejavu_dataset

logger = logging.getLogger('root_api')
dirname = './'


class DeepRuleEvaluation:
    def __init__(self, catalog, ratio=0.5):
        self.catalog = catalog
        self.dataset_path = path.join(dirname, self.catalog)
        self.model_path = path.join(dirname, 'models')
        self.report_folder = path.join(dirname, self.catalog, 'report_alertRCA')

        self.training_split = ratio

        self.catalog_train_files, self.catalog_valid_files, self.catalog_test_files = get_dejavu_dataset(
            self.dataset_path, 
            ratio=ratio
        )

        self.rca_parameters = {
            'attention_type': 'dagat',
            'hid_channels': 8,
            'attention_channels': 3,
            'heads': 4,
            'gru_size': {}
        }
        n_ts = sum(self.catalog_train_files.window_size)
        for k, v in self.catalog_train_files.cgraph.metric_size_dict.items():
            self.rca_parameters['gru_size'][k] = torch.Size([v, n_ts])
        
        self.dp = DeepRule(self.catalog, self.rca_parameters, token_type='dejavu')
        
        print(self.catalog)
        print('#self.catalog_train_files:', len(self.catalog_train_files))
        print('#self.catalog_valid_files:', len(self.catalog_valid_files))
        print('#self.catalog_test_files:', len(self.catalog_test_files))
        print('rca_parameters:')
        for k, v in self.rca_parameters.items():
            print(f'{k}: {v}')

    def train(self, train_file_list, save_name='deeprule', test_file_list=None, valid_file_list=None):
        # Training
        real_dir = path.join(self.model_path, save_name)
        self.dp.train(
            train_file_list, 
            load_model_path=real_dir, 
            save_model_path=real_dir, 
            testing_data=test_file_list,
            valid_data=valid_file_list) 
        

    def test(self, test_file_list, model_dir=None):
        # Testing
        if model_dir is not None:
            real_dir = path.join(self.model_path, model_dir)
            self.dp.load_models(real_dir)
        self.dp.set_model_eval()

        rca, report_obj = {}, {}

        n_test = len(test_file_list)
        for fi in range(n_test):
            labels, key_, data = test_file_list[fi]
            key = str(key_)
            report_obj[key], grades, length= {}, {}, 0
            
            res, _ = self.get_res(data, key)
            untied_res = {}
            for n in res:
                untied_res[n] = [res[n].detach().cpu().item(), 1]
            res = untied_res

            # sort res based on scores from high to low
            res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
            for n in res:
                res[n] = tuple(res[n])
            report_obj[key]['gt'] = labels
            report_obj[key]['fullResponse'] = res

            # default value, which means rca result failed to hit any rank in groot result
            report_obj[key]['rank'] = len(res)
            rca[key] = labels

            for k, v in res.items():
                if not v in grades:
                    grades[v] = length + 1
                    length += list(res.values()).count(v)
                    
                if k in rca[key]:
                    report_obj[key]['rank'], report_obj[key]['length'] = grades[v], length
                    # print(k, report_obj[key]['rank'], report_obj[key]['length'])
                    break
        
        # report file name & write report
        date_format = "%y-%m-%d_%H-%M-%S"
        timeString = datetime.now().astimezone(pytz.timezone('US/Pacific')).strftime(date_format)
        report_file = 'report_' + self.catalog + '_' + timeString + '_.json'
        report_path = path.join(self.report_folder, report_file)

        rank1_list = []
        rank2_list = []
        rank3_list = []
        rank5_list = []
        failure_list = []
        missing_list = []
        average_rank = []
        for key, data in report_obj.items():
            if 'length' in data:
                if data['rank'] == 1 and data['length'] <= 5:
                    rank1_list.append(key)
                elif data['rank'] <= 2 and data['length'] <= 5:
                    rank2_list.append(key)
                elif data['rank'] <= 3 and data['length'] <= 5:
                    rank3_list.append(key)
                elif data['length'] <= 5:
                    rank5_list.append(key)
                else:
                    failure_list.append(key)
            else:
                missing_list.append(key)
                print("RCA of {} is missing".format(key))
            average_rank.append(data['rank'])

        top1_accuracy = len(rank1_list) / len(report_obj)
        top2_accuracy = (len(rank1_list) + len(rank2_list)) / len(report_obj)
        top3_accuracy = (len(rank1_list) + len(rank2_list) + len(rank3_list)) / len(report_obj)
        top5_accuracy = (len(rank1_list) + len(rank2_list) + len(rank3_list) + len(rank5_list)) / len(report_obj)
        print(f'acc@1: {top1_accuracy}, acc@2: {top2_accuracy}, acc@3: {top3_accuracy}, acc@5:{top5_accuracy}')
        print(f'Average Rank{sum(average_rank)/len(average_rank)}')

        with open(report_path, 'w') as out_file:
            json.dump(
                {
                    'samples_report': report_obj,
                    'ACC_1': top1_accuracy,
                    'ACC_3': top3_accuracy
                },
                out_file, 
                indent=4
            )

    def get_res(self, data, key):
        lss = {
            'A1': ['os_021', 'os_022'],
            'A2': ['os_021', 'os_022'],
            'B': ['OSB', 'Redis', 'Mysql']
        }

        debug = None
        if key == "1592070840":
            debug = key

        return deep_rca_rank(
            data, 
            self.dp, 
            limited_start_set = lss[self.catalog],
            max_iter = 4,
            debug=debug,
            **self.rca_parameters
        ), 0

if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    parse = argparse.ArgumentParser()
    parse.add_argument('--ratio', default=0.5, type=float)
    parse.add_argument('--dataset', default='A1', type=str)
    parse.add_argument('--modeldir', default='deeprule', type=str)
    args = parse.parse_args()
    print('Configuration:', args)
    ratio = args.ratio
    dataset = args.dataset
    model_dir = args.modeldir
    
    f = DeepRuleEvaluation(dataset, ratio)

    f.train(f.catalog_train_files, save_name=model_dir, test_file_list=f.catalog_test_files, valid_file_list=f.catalog_valid_files)
    print(f'Above Train model {dataset} dataset')
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    
    f.test(f.catalog_test_files, model_dir=model_dir)
    print(f'Above Test model on {dataset} dataset')
    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    