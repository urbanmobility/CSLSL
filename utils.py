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
            for tar_idx in range(len(data_input[uid][sid]['target_l'])):
                data_queue.append((uid, sid, tar_idx))
            
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
            uid, sid, tar_idx = data_queue[batch_idx]
            uid_batch.append([uid])
            # current
            in_idx = tar_idx + 1
            loc_cur_batch.append(torch.LongTensor(data_input[uid][sid]['loc'][1][:in_idx]))  
            tim_cur_ts = torch.LongTensor(data_input[uid][sid]['tim'][1][:in_idx])
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
            target_l = torch.LongTensor([data_input[uid][sid]['target_l'][tar_idx]]) 
            target_l_batch.append(target_l) 
            target_len_batch.append(target_l.shape[0])
            target_th_batch.append(torch.LongTensor([data_input[uid][sid]['target_th'][tar_idx]]))   
            # catrgory
            if cat_contained:
                cat_his_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][0])) 
                cat_cur_batch.append(torch.LongTensor(data_input[uid][sid]['cat'][1][:in_idx])) 
                target_c_batch.append(torch.LongTensor([data_input[uid][sid]['target_c'][tar_idx]])) 
            
                
                  
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
    
    
    
class progress_supervisor(object):
    def __init__(self, all_num, path):
        self.cur_num = 1
        self.start_time = time.time()
        self.all_num = all_num
        self.path = path
        
        with open(self.path, 'w') as f:
            f.write('Start')
        
    def update(self):
        '''Usage:
            count_time = count_run_time(5 * 4 * 4)
            count_time.path = f'{args.out_dir}{args.model_name}_{args.data_name}.txt'
            main()
            count_time.current_count()
        '''

        past_time = time.time()-self.start_time
        avg_time = past_time / self.cur_num
        fut_time = avg_time * (self.all_num - self.cur_num)

        content = '=' * 10 + ' Progress observation'
        content += f'Current time is {time.strftime("%Y-%m-%d %H:%M:%S")}\n'
        content += f'Current Num: {self.cur_num} / {self.all_num}\n'
        content += f'Past time: {past_time:.2f}s ({past_time/3600:.2f}h)\n'
        content += f'Average time: {avg_time:.2f}s ({avg_time/3600:.2f}h)\n'
        content += f'Future time: {fut_time:.2f}s ({fut_time/3600:.2f}h)\n'

        with open(self.path, 'w') as f:
            f.write(content) 
            
        self.cur_num += 1
        return content
    
    def delete(self):
        if os.path.exists(self.path):
            os.remove(self.path)
            
        if not os.path.exists(self.path):
            print('Supervisor file delete success')