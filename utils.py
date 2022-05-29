import os
import time
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

def generate_batch_data(data_input, data_id, device, batch_size, cat_contained):
    '''generate batch data'''
    
    # generate (uid, sid) queue
    data_queue = list()
    uid_list = data_id.keys()
    for uid in uid_list:
        for sid in data_id[uid]:
            data_queue.append((uid, sid))
            
    # generate batch data
    data_len = len(data_queue)
    batch_num = int(data_len/batch_size)
    print(f'Number of batch is {batch_num}')
    # iterate batch number times
    for i in range(batch_num):
        # batch data
        uid_batch = []
        loc_cur_batch = []
        tim_w_cur_batch = []
        tim_h_cur_batch = []
        loc_his_batch = []
        tim_w_his_batch = []
        tim_h_his_batch = []
        target_l_batch = []
        target_c_batch = []
        target_th_batch = []
        target_len_batch = []
        history_len_batch = []
        current_len_batch = []
        if cat_contained:
            cat_cur_batch = []
            cat_his_batch = []
        
        if i % 100 == 0:
            print('====', f'[Batch={i}/{batch_num}]', end=', ')

        batch_idx_list = np.random.choice(data_len, batch_size, replace=False)
        # iterate batch index
        for batch_idx in batch_idx_list:
            uid, sid = data_queue[batch_idx]
            uid_batch.append([uid])
            # current
            loc_cur_batch.append(torch.LongTensor(data_input[uid][sid]['loc'][1]))  
            tim_cur_ts = torch.LongTensor(data_input[uid][sid]['tim'][1])
            tim_w_cur_batch.append(tim_cur_ts[:, 0])      
            tim_h_cur_batch.append(tim_cur_ts[:, 1]) 
            current_len_batch.append(tim_cur_ts.shape[0])
            # history
            loc_his_batch.append(torch.LongTensor(data_input[uid][sid]['loc'][0]))
            tim_his_ts = torch.LongTensor(data_input[uid][sid]['tim'][0])
            tim_w_his_batch.append(tim_his_ts[:, 0])      
            tim_h_his_batch.append(tim_his_ts[:, 1])   
            history_len_batch.append(tim_his_ts.shape[0])
            # target 
            target_l = torch.LongTensor(data_input[uid][sid]['target_l']) 
            target_l_batch.append(target_l) 
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor(data_input[uid][sid]['target_th']))   
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][0])) 
                cat_cur_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][1])) 
                target_c_batch.append(torch.LongTensor(data_input[uid][sid]['target_c'])) 
            
                
                  
        # padding
        uid_batch_tensor = torch.LongTensor(uid_batch).to(device)
        # current
        loc_cur_batch_pad = pad_sequence(loc_cur_batch, batch_first=True).to(device)
        tim_w_cur_batch_pad = pad_sequence(tim_w_cur_batch, batch_first=True).to(device)
        tim_h_cur_batch_pad = pad_sequence(tim_h_cur_batch, batch_first=True).to(device)
        # history
        loc_his_batch_pad = pad_sequence(loc_his_batch, batch_first=True).to(device)
        tim_w_his_batch_pad = pad_sequence(tim_w_his_batch, batch_first=True).to(device)
        tim_h_his_batch_pad = pad_sequence(tim_h_his_batch, batch_first=True).to(device)
        # target
        target_l_batch_pad = pad_sequence(target_l_batch, batch_first=True).to(device)   
        target_th_batch_pad = pad_sequence(target_th_batch, batch_first=True).to(device)   
            
        if cat_contained:    
            cat_his_batch_pad = pad_sequence(cat_his_batch, batch_first=True).to(device)
            cat_cur_batch_pad = pad_sequence(cat_cur_batch, batch_first=True).to(device)
            target_c_batch_pad = pad_sequence(target_c_batch, batch_first=True).to(device)   
            yield  (target_len_batch, history_len_batch, current_len_batch),\
                    (target_l_batch_pad, target_th_batch_pad, target_c_batch_pad),\
                    (uid_batch_tensor,\
                         loc_his_batch_pad, loc_cur_batch_pad,\
                         tim_w_his_batch_pad, tim_w_cur_batch_pad,\
                         tim_h_his_batch_pad, tim_h_cur_batch_pad,\
                         cat_his_batch_pad, cat_cur_batch_pad)
        else:
            yield  (target_len_batch, history_len_batch, current_len_batch),\
                    (target_l_batch_pad, target_th_batch_pad),\
                    (uid_batch_tensor,\
                         loc_his_batch_pad, loc_cur_batch_pad,\
                         tim_w_his_batch_pad, tim_w_cur_batch_pad,\
                         tim_h_his_batch_pad, tim_h_cur_batch_pad)
        
        
        
            
    print('Batch Finished')
             

def generate_mask(data_len):
    '''Generate mask
    Args:
        data_len : one dimension list, reflect sequence length
    '''
    mask = []
    for i_len in data_len:
        mask.append(torch.ones(i_len).bool())
    return ~pad_sequence(mask, batch_first=True)
        
        
        
def calculate_recall(target_pad, pred_pad):
    '''Calculate recall
    Args:
        target: (batch, max_seq_len), padded target
        pred: (batch, max_seq_len, pred_scores), padded
    '''
    # variable
    acc = np.zeros(3)    # 1, 5, 10
    
    # reshape and to numpy
    target_list = target_pad.data.reshape(-1).cpu().numpy()
    # topK
    pid_size = pred_pad.shape[-1]
    _, pred_list = pred_pad.data.reshape(-1, pid_size).topk(20)
    pred_list = pred_list.cpu().numpy()
    
    for idx, pred in enumerate(pred_list):
        target = target_list[idx]
        if target == 0:  # pad
            continue
        if target in pred[:1]:
            acc += 1
        elif target in pred[:5]:
            acc[1:] += 1
        elif target in pred[:10]:
            acc[2:] += 1
    
    return acc


def get_model_params(model):
    total_num = sum(param.numel() for param in model.parameters())
    trainable_num = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f'==== Parameter numbers:\n total={total_num}, trainable={trainable_num}')
    
    