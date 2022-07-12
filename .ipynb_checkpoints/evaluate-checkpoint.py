import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd

import torch

from utils import *
from model import *


def settings(param=[]):
    parser = argparse.ArgumentParser()
    # data params
    parser.add_argument('--path_in', type=str, default='./data/', help="input data path")
    parser.add_argument('--path_out', type=str, default='./results/', help="output data path")
    parser.add_argument('--data_name', type=str, default='NYC', help="data name")
    parser.add_argument('--cat_contained', action='store_false', default=True, help="whether contain category")
    parser.add_argument('--out_filename', type=str, default='', help="output data filename")
    # train params
    parser.add_argument('--gpu', type=str, default='0', help="GPU index to choose")
    parser.add_argument('--run_num', type=int, default=10, help="run number")
    parser.add_argument('--model_path', type=str, default='./results/pretrained/', help="model path")
    parser.add_argument('--model_name', type=str, default='model_NYC', help="model name to load")
    parser.add_argument('--batch_size', type=int, default=64, help="batch size")
    
    # model params
    # embedding
    parser.add_argument('--user_embed_dim', type=int, default=20, help="user embedding dimension")
    parser.add_argument('--loc_embed_dim', type=int, default=200, help="loc embedding dimension")
    parser.add_argument('--tim_h_embed_dim', type=int, default=20, help="time hour embedding dimension")
    parser.add_argument('--tim_w_embed_dim', type=int, default=10, help="time week embedding dimension")
    parser.add_argument('--cat_embed_dim', type=int, default=100, help="category embedding dimension")
    # rnn
    parser.add_argument('--rnn_type', type=str, default='gru', help="rnn type")
    parser.add_argument('--rnn_layer_num', type=int, default=1, help="rnn layer number")
    parser.add_argument('--rnn_t_hid_dim', type=int, default=600, help="rnn hidden dimension for t")
    parser.add_argument('--rnn_c_hid_dim', type=int, default=600, help="rnn hidden dimension for c")
    parser.add_argument('--rnn_l_hid_dim', type=int, default=600, help="rnn hidden dimension for l")
    parser.add_argument('--dropout', type=float, default=0.1, help="drop out for rnn")
    
    
    if __name__ == '__main__' and param == []:
        params =  parser.parse_args()
    else:
        params = parser.parse_args(param)
        
    if not os.path.exists(params.path_out):
        os.mkdir(params.path_out)
    
    return params


def evaluate(params, dataset):
    '''Evaluate model performance
    '''
    
    # dataset info
    params.uid_size = len(dataset['uid_list'])
    params.pid_size = len(dataset['pid_dict'])   
    params.cid_size = len(dataset['cid_dict']) if params.cat_contained else 0
    # generate input data
    data_test, test_id = dataset['test_data'], dataset['test_id']
    pid_lat_lon_radians = torch.tensor([[0, 0]] + list(dataset['pid_lat_lon_radians'].values())).to(params.device)
    
    # load model
    model = Model(params).to(params.device)
    model.load_state_dict(torch.load(params.model_path+params.model_name+'.pkl'))
    model.eval()
    
    
    l_acc_all = np.zeros(3)
    c_acc_all = np.zeros(3)
    t_mse_all = 0.
    valid_num_all = 0
    model.eval()
    # evaluate with batch
    for mask_batch, target_batch, data_batch in generate_batch_data(data_test, test_id, params.device, params.batch_size, params.cat_contained):
        # model forward
        th_pred, c_pred, l_pred, valid_num = model(data_batch, mask_batch)
        
        # calculate metrics
        l_acc = calculate_recall(target_batch[0], l_pred)
        l_acc_all += l_acc
        t_mse_all += torch.nn.functional.l1_loss(th_pred.squeeze(-1), target_batch[1].squeeze(-1), reduction='sum').item()
        valid_num_all += valid_num
        
        if params.cat_contained:
            c_acc = calculate_recall(target_batch[2], c_pred)
            c_acc_all += c_acc

    return  l_acc_all / valid_num_all, c_acc_all / valid_num_all, t_mse_all / valid_num_all


if __name__ == '__main__':
    
    print('='*20, ' Program Start')
    params = settings()
    params.device = torch.device(f"cuda:{params.gpu}")
    print('Parameter is\n', params.__dict__)
    
    # file name to store
    FILE_NAME = [params.path_out, f'{time.strftime("%Y%m%d")}_{params.data_name}_']
    FILE_NAME[1] += f'{params.out_filename}'
    
    # Load data
    print('='*20, ' Loading data')
    start_time = time.time()
    if params.cat_contained:
        dataset = pickle.load(open(f'{params.path_in}{params.data_name}_cat.pkl', 'rb'))
    else:    
        dataset = pickle.load(open(f'{params.path_in}{params.data_name}.pkl', 'rb'))
    print(f'Finished, time cost is {time.time()-start_time:.1f}')

    # metrics
    metrics = pd.DataFrame()
    
    # start running
    print('='*20, "Start Evaluating")
    for i in range(params.run_num):
        print('='*20, f'Run {i}')
        
        # To Revise
        results, results_c, results_t = evaluate(params, dataset)
        metric_dict = {'Rec-l@1': results[0], 'Rec-l@5': results[1], 'Rec-l@10': results[2],
                          'MAE': results_t, 'Rec-c@1': results_c[0], 'Rec-c@5': results_c[1], 'Rec-c@10': results_c[2]}
        metric_tmp = pd.DataFrame(metric_dict, index=[i])
        metrics = pd.concat([metrics, metric_tmp])
        
    
    print('='*20, "Finished")
    mean = pd.DataFrame(metrics.mean()).T
    mean.index = ['mean']
    std = pd.DataFrame(metrics.std()).T
    std.index = ['std']
    metrics = pd.concat([metrics, mean, std])
    print(metrics)
    
    # save
    metrics.to_csv(f'{FILE_NAME[0]}metrics_{FILE_NAME[1]}.csv')
    print('='*20, f'\nMetrics saved. File name is {FILE_NAME[0]}metrics_{FILE_NAME[1]}.csv')