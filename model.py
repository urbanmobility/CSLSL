import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
from utils import generate_mask

NUM_TASK = 3
EARTHRADIUS = 6371.0

class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.__dict__.update(params.__dict__)
        self.th_size = 12 + 1         # +1 because of the padding value 0, 
        self.tw_size = 7 + 1
        self.pid_size += 1
        self.uid_size += 1
        self.cid_size += 1
        # Dim
        RNN_input_dim = self.user_embed_dim + self.loc_embed_dim + self.tim_w_embed_dim + self.tim_h_embed_dim
        if self.cat_contained:
            RNN_input_dim += self.cat_embed_dim
            
        # Embedding; (all id is start from 1)
        self.user_embedder = nn.Embedding(self.uid_size, self.user_embed_dim)   # without padding
        self.loc_embedder = nn.Embedding(self.pid_size, self.loc_embed_dim, padding_idx=0)
        self.tim_embedder_week = nn.Embedding(self.tw_size, self.tim_w_embed_dim, padding_idx=0)
        self.tim_embedder_hour = nn.Embedding(self.th_size, self.tim_h_embed_dim, padding_idx=0)
        self.cat_embedder = nn.Embedding(self.cid_size, self.cat_embed_dim, padding_idx=0)
            
        # Capturer
        # Version: Seperate
        self.capturer_t = SessionCapturer(RNN_input_dim, self.rnn_t_hid_dim, params)
        self.capturer_l = SessionCapturer(RNN_input_dim, self.rnn_l_hid_dim, params)
        
        if self.cat_contained:
            self.cat_embedder = nn.Embedding(self.cid_size, self.cat_embed_dim, padding_idx=0)
            self.capturer_c = SessionCapturer(RNN_input_dim, self.rnn_c_hid_dim, params)
        
        # CMTL
        self.fc_t = nn.Linear(self.rnn_t_hid_dim, 1)
        self.label_trans_t = nn.Linear(1, self.tim_h_embed_dim)
        if self.cat_contained:
            self.fc_c = nn.Linear(self.rnn_c_hid_dim + self.tim_h_embed_dim, self.cid_size)
            self.label_trans_c = nn.Linear(self.cid_size, self.cat_embed_dim)
            self.fc_l = nn.Linear(self.rnn_l_hid_dim + self.cat_embed_dim, self.pid_size)
        else:
            self.fc_l = nn.Linear(self.rnn_l_hid_dim + self.tim_h_embed_dim, self.pid_size)
        
        # Loss
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='sum', ignore_index=0)
        self.mse_loss = nn.L1Loss(reduction='sum')
               
    def forward(self, data, mask_batch):
        
        # Capture Short-term representation
        # 1) get data and generate mask
        if self.cat_contained:
            uid_tensor, loc_his_pad, loc_cur_pad, tim_w_his_pad, tim_w_cur_pad,\
                        tim_h_his_pad, tim_h_cur_pad, cat_his_pad, cat_cur_pad = data
        else:
            uid_tensor, loc_his_pad, loc_cur_pad, tim_w_his_pad, tim_w_cur_pad,\
                        tim_h_his_pad, tim_h_cur_pad = data
        target_mask = generate_mask(mask_batch[0]).unsqueeze(2).to(self.device)
        his_mask = generate_mask(mask_batch[1]).unsqueeze(2).to(self.device)
        cur_mask = generate_mask(mask_batch[2]).unsqueeze(2).to(self.device)
            
        # 2) embed
        uid_embed = self.user_embedder(uid_tensor)
        loc_his_embed, loc_cur_embed = self.loc_embedder(loc_his_pad), self.loc_embedder(loc_cur_pad)
        tim_week_his_embed, tim_week_cur_embed = self.tim_embedder_week(tim_w_his_pad), self.tim_embedder_week(tim_w_cur_pad)
        tim_hour_his_embed, tim_hour_cur_embed = self.tim_embedder_hour(tim_h_his_pad), self.tim_embedder_hour(tim_h_cur_pad)
        rnn_input_his_concat = torch.cat((uid_embed.expand(-1, loc_his_embed.shape[1], -1), loc_his_embed, tim_week_his_embed, tim_hour_his_embed), dim=-1)
        rnn_input_cur_concat = torch.cat((uid_embed.expand(-1, loc_cur_embed.shape[1], -1), loc_cur_embed, tim_week_cur_embed, tim_hour_cur_embed), dim=-1)
        if self.cat_contained:
            cat_his_embed, cat_cur_embed = self.cat_embedder(cat_his_pad), self.cat_embedder(cat_cur_pad)
            rnn_input_his_concat = torch.cat((rnn_input_his_concat, cat_his_embed), dim=-1)
            rnn_input_cur_concat = torch.cat((rnn_input_cur_concat, cat_cur_embed), dim=-1)
            
        # 3) rnn capturer
        # Version: seperate
        cur_t_rnn, hc_t = self.capturer_t(rnn_input_his_concat, rnn_input_cur_concat, his_mask, cur_mask, mask_batch[1:])
        if self.cat_contained:
            cur_c_rnn, hc_c = self.capturer_c(rnn_input_his_concat, rnn_input_cur_concat, his_mask, cur_mask, mask_batch[1:], hc_t) 
            cur_l_rnn, hc_l = self.capturer_l(rnn_input_his_concat, rnn_input_cur_concat, his_mask, cur_mask, mask_batch[1:], hc_c)   
            
            # 4) tower, t,c,l
            # CMTL
            hc_t, hc_c, hc_l = hc_t.squeeze(), hc_c.squeeze(), hc_l.squeeze()
            t_pred = self.fc_t(hc_t) 
            t_trans = self.label_trans_t(t_pred.clone())
            c_pred = self.fc_c(torch.cat((hc_c, t_trans), dim=-1)) 
            c_trans = self.label_trans_c(c_pred.clone())
            l_pred = self.fc_l(torch.cat((hc_l, c_trans), dim=-1)) 
        else:
            cur_l_rnn, _ = self.capturer_l(rnn_input_his_concat, rnn_input_cur_concat, his_mask, cur_mask, mask_batch[1:], hc_t)   
            # 4) tower, t,c,l
            # CMTL
            t_pred = self.fc_t(cur_t_rnn) 
            t_trans = self.label_trans_t(t_pred.clone())
            l_pred = self.fc_l(torch.cat((cur_l_rnn, t_trans), dim=-1)) 
       
        valid_num = (target_mask==0).sum().item()
        
        if self.cat_contained:
            return t_pred, c_pred, l_pred, valid_num
        else:
            return t_pred, 0, l_pred, valid_num
        
    def calculate_loss(self, th_pred_in, c_pred_in, l_pred_in, target_batch, valid_num, pid_lat_lon_radians):
        
        # location loss with cross entropy
        l_pred = l_pred_in.reshape(-1, self.pid_size)
        l_target = target_batch[0].reshape(-1)
        loss_l = self.cross_entropy_loss(l_pred, l_target) / valid_num
        # time loss with mse loss
        th_target = target_batch[1].reshape(-1)
        loss_t = self.mse_loss(th_pred_in.squeeze(-1), th_target.float()) / valid_num
        
        loss_geocons = self.geo_con_loss(l_pred, l_target, pid_lat_lon_radians) / valid_num
        
        if self.cat_contained:
            # category loss with cross entropy
            c_pred = c_pred_in.reshape(-1, self.cid_size)
            c_target = target_batch[2].reshape(-1)
            loss_c = self.cross_entropy_loss(c_pred, c_target) / valid_num

            return loss_t, loss_c, loss_l, loss_geocons
        else:
            return loss_t, torch.tensor(0), loss_l, loss_geocons
        
    def geo_con_loss(self, l_pred_in, l_target, pid_lat_lon_radians):
        
        log_softmax = nn.functional.log_softmax(l_pred_in, dim=-1)
        l_pred = torch.argmax(log_softmax, dim=-1)
        l_coor_pred = pid_lat_lon_radians[l_pred]
        l_coor_tar = pid_lat_lon_radians[l_target]

        dlat = l_coor_pred[:, 0] - l_coor_tar[:, 0]
        dlon = l_coor_pred[:, 1] - l_coor_tar[:, 1]
        a = torch.sin(dlat/2) **2 + torch.cos(l_coor_pred[:, 0]) * torch.cos(l_coor_tar[:, 0]) * (torch.sin(dlon/2))**2
        km = EARTHRADIUS * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))

        return km.sum()
    
class SessionCapturer(nn.Module):
    '''Expert module'''
    def __init__(self, RNN_input_dim, RNN_hid_dim, params):
        super().__init__()
        self.__dict__.update(params.__dict__)
        # RNN
        self.HistoryCapturer = RnnFactory(self.rnn_type).create(RNN_input_dim, RNN_hid_dim, self.rnn_layer_num, self.dropout) 
        self.CurrentCapturer = RnnFactory(self.rnn_type).create(RNN_input_dim, RNN_hid_dim, self.rnn_layer_num, self.dropout) 
        
    def forward(self, his_in, cur_in, his_mask, cur_mask, len_batch, hc=None):
        
        
        # 1) pack padded
        his_in_pack = pack_padded_sequence(his_in, len_batch[0], batch_first=True, enforce_sorted=False)
        cur_in_pack = pack_padded_sequence(cur_in, len_batch[1], batch_first=True, enforce_sorted=False)
        
        # 2) history capturer
        if hc == None:
            history_pack, history_hc = self.HistoryCapturer(his_in_pack)
        else:
            history_pack, history_hc = self.HistoryCapturer(his_in_pack, hc)
        # 3) current capturer
        current_pack, current_hc = self.CurrentCapturer(cur_in_pack, history_hc)
        
        # 4) unpack
        history_unpack, _ = pad_packed_sequence(history_pack, batch_first=True)
        current_unpack, _ = pad_packed_sequence(current_pack, batch_first=True)  # (B, S, BH)
        

        # Version: concat
#         return current_unpack, current_hc, history_unpack

        # Version: basic
        return current_unpack, current_hc



        

class RnnFactory():
    ''' Creates the desired RNN unit. '''
    
    def __init__(self, rnn_type):
        self.rnn_type = rnn_type
        
    def create(self, input_dim, hidden_dim, num_layer, dropout=0):
        if self.rnn_type == 'rnn':
            return nn.RNN(input_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout)
        if self.rnn_type == 'gru':
            return nn.GRU(input_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout)   
        if self.rnn_type == 'lstm':
            return nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True, dropout=dropout)
            