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
    parser.add_argument('--epoch_num', type=int, default=30, help="epoch number")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-6, help="weight decay")
    parser.add_argument('--evaluate_step', type=int, default=2, help="evaluate step")
    parser.add_argument('--lam_t', type=float, default=1, help="loss lambda time")
    parser.add_argument('--lam_c', type=float, default=1, help="loss lambda category")
    parser.add_argument('--lam_s', type=float, default=1, help="loss lambda for geographcal consistency")
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


def train(params, dataset):
    
    # dataset info
    params.uid_size = len(dataset['uid_list'])
    params.pid_size = len(dataset['pid_dict'])   
    params.cid_size = len(dataset['cid_dict']) if params.cat_contained else 0
    # generate input data
    data_train, train_id = dataset['train_data'], dataset['train_id']
    data_test, test_id = dataset['test_data'], dataset['test_id']
    pid_lat_lon_radians = torch.tensor([[0, 0]] + list(dataset['pid_lat_lon_radians'].values())).to(params.device)
    
    # model and optimizer
    model = Model(params).to(params.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    print('==== Model is \n', model)
    get_model_params(model)
    print('==== Optimizer is \n', optimizer)
    
    # iterate epoch
    best_info_train = {'epoch':0, 'Recall@1':0}   # best metrics
    best_info_test = {'epoch':0, 'Recall@1':0}   # best metrics
    print('='*10, ' Training')
    for epoch in range(params.epoch_num):
        model.train()
        # variable
        loss_l_all = 0.
        loss_t_all = 0.
        loss_c_all = 0.
        loss_s_all = 0.
        loss_all = 0.
        valid_all = 0
        
        # train with batch
        time_start = time.time()
        print('==== Train', end=', ')
        for mask_batch, target_batch, data_batch in generate_batch_data(data_train, train_id, params.device, params.batch_size, params.cat_contained):
            # model forward
            th_pred, c_pred, l_pred, valid_num = model(data_batch, mask_batch)
            # calcuate loss
            loss_t, loss_c, loss_l, loss_s = model.calculate_loss(th_pred, c_pred, l_pred, target_batch, valid_num, pid_lat_lon_radians)
            loss = loss_l + params.lam_t * loss_t + params.lam_c * loss_c + params.lam_s * loss_s  
            valid_all += valid_num 
            loss_l_all += loss_l.item() * valid_num
            loss_t_all += loss_t.item() * valid_num
            loss_c_all += loss_c.item() * valid_num
            loss_s_all += loss_s.item() * valid_num
            loss_all += loss.item() * valid_num

            # backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        train_time = time.time() - time_start
        loss_all /= valid_all
        loss_l_all /= valid_all
        loss_t_all /= valid_all
        loss_c_all /= valid_all
        loss_s_all /= valid_all
        

        # evaluation 
        if epoch % params.evaluate_step == 0:
            # evaluate with train data
            print('==== Evaluate train data', end=', ')
            time_start = time.time()
            train_acc_l, train_acc_c, train_mse_t = evaluate(model, data_train, train_id, params)
            train_eval_time = time.time() - time_start
            # evaluate with test data
            print('==== Evaluate test data', end=', ')
            time_start = time.time()
            test_acc_l, test_acc_c, test_mse_t = evaluate(model, data_test, test_id, params)
            test_time = time.time() - time_start
            print(f'[Epoch={epoch+1}/{params.epoch_num}], loss={loss_all:.2f}, loss_l={loss_l_all:.2f},', end=' ')
            print(f'loss_t={loss_t_all:.2f}, loss_c={loss_c_all:.2f}, loss_s={loss_s_all:.2f};')
            print(f'Acc_loc: train_l={train_acc_l}, test_l={test_acc_l};')
            print(f'Acc_cat: train_c={train_acc_c}, test_c={test_acc_c};')
            print(f'MAE_time: train_t={train_mse_t:.2f}, test_t={test_mse_t:.2f};') 
            print(f'Eval time cost: train={train_eval_time:.1f}s, test={test_time:.1f}s\n')
            # store info
            if best_info_train['Recall@1'] < train_acc_l[0]:
                best_info_train['epoch'] = epoch
                best_info_train['Recall@1'] = train_acc_l[0]
                best_info_train['Recall@all'] = train_acc_l
                best_info_train['MAE'] = train_mse_t
                best_info_train['model_params'] = model.state_dict()
            if best_info_test['Recall@1'] < test_acc_l[0]:
                best_info_test['epoch'] = epoch
                best_info_test['Recall@1'] = test_acc_l[0]
                best_info_test['Recall@all'] = test_acc_l
                best_info_test['MAE'] = test_mse_t
                best_info_test['model_params'] = model.state_dict()

        else:
            print(f'[Epoch={epoch+1}/{params.epoch_num}], loss={loss_all:.2f}, loss_l={loss_l_all:.2f},', end=' ')
            print(f'loss_t={loss_t_all:.2f}, loss_c={loss_c_all:.2f}, loss_s={loss_s_all:.2f};')
           
    # evaluation
    print('='*10, ' Testing')
    results_l, results_c, results_t = evaluate(model, data_test, test_id, params)
    print(f'Test results: loc={results_l}, cat={results_c}, tim={results_t:.2f}')
    
    # best metrics info
    print('='*10,' Run finished')
    print(f'Best train results is {best_info_train["Recall@all"]} at Epoch={best_info_train["epoch"]}')
    print(f'Best test results is {best_info_test["Recall@all"]} at Epoch={best_info_test["epoch"]}')
    
    return results_l, results_c, results_t, best_info_test


def evaluate(model, data, data_id, params):
    '''Evaluate model performance
    '''
    l_acc_all = np.zeros(3)
    c_acc_all = np.zeros(3)
    t_mse_all = 0.
    valid_num_all = 0
    model.eval()
    # evaluate with batch
    for mask_batch, target_batch, data_batch in generate_batch_data(data, data_id, params.device, params.batch_size, params.cat_contained):
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
    best_info_all_run = {'epoch':0, 'Recall@1':0}
    
    # start running
    print('='*20, "Start Training")
    for i in range(params.run_num):
        print('='*20, f'Run {i}')
        
        # To Revise
        results, results_c, results_t, best_info_one_run = train(params, dataset)
        metric_dict = {'Rec-l@1': results[0], 'Rec-l@5': results[1], 'Rec-l@10': results[2],
                          'MAE': results_t, 'Rec-c@1': results_c[0], 'Rec-c@5': results_c[1], 'Rec-c@10': results_c[2]}
        metric_tmp = pd.DataFrame(metric_dict, index=[i])
        metrics = pd.concat([metrics, metric_tmp])
        
        if best_info_all_run['Recall@1'] < best_info_one_run['Recall@1']:
            best_info_all_run = best_info_one_run.copy()
            best_info_all_run['run'] = i

        
    
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
    torch.save(best_info_all_run["model_params"], f'{FILE_NAME[0]}model_{FILE_NAME[1]}.pkl')
    print(f'Model saved (Run={best_info_all_run["run"]}, Epoch={best_info_all_run["epoch"]})')