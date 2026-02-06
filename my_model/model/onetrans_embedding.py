# 1. 点击序列embedding
# 2. 曝光序列embedding
# 3. 时间戳embedding
# 4. 纵向拼接点击序列、时间戳
# 5. 纵向拼接曝光序列、时间戳
# 6. 横向拼接点击序列、<SEP>、曝光序列
# 7. uid embedding
# 8. itemid embedding

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init

import logging




class OneTransEmb(nn.Module):
    def __init__(
        self,
        max_itemid: int,
        max_uid: int,
        d_model: int,
        num_ratings: int,
        device='cpu'
    ):
        super().__init__()
        self.exposure_emb = nn.Embedding(max_itemid, d_model)
        self.click_emb = nn.Embedding(max_itemid, d_model)
        self.uid_emb = nn.Embedding(max_uid, d_model)
        self.timestamp_fc = nn.Linear(1, d_model)
        self.exposure_fc = nn.Linear(3*d_model, d_model)
        self.click_fc = nn.Linear(3*d_model, d_model)
        self.rating_emb = nn.Embedding(num_ratings+1, d_model)
        self.device=device

    def forward(self, row):
        high_items_pad = row[0].to(self.device)
        high_times_pad = row[1].to(self.device)
        high_len = row[2].to(self.device)
        user_id_tensor = row[3].to(self.device)
        item_id_tensor = row[4].to(self.device)[:, -1]
        item_rating_tensor = row[5].to(self.device)[:, -1]
        item_time_pad = row[6].to(self.device)[:, -1]
        seq_items_pad = row[4].to(self.device)[:, :-1]
        seq_ratings_pad = row[5].to(self.device)[:, :-1]
        seq_times_pad = row[6].to(self.device)[:, :-1]
        
        high_times_gap = item_time_pad.unsqueeze(1) - high_times_pad
        seq_times_gap = item_time_pad.unsqueeze(1) - seq_times_pad

        seq_len = row[7].to(self.device)
        logging.info(f'high_items_pad.shape: {high_items_pad.shape}')
        logging.info(f'high_times_pad.shape: {high_times_pad.shape}')
        logging.info(f'seq_items_pad.shape: {seq_items_pad.shape}')
        logging.info(f'seq_ratings_pad.shape: {seq_ratings_pad.shape}')
        logging.info(f'seq_times_pad.shape: {seq_times_pad.shape}')
        logging.info(f'high_times_gap: {high_times_gap}')
        logging.info(f'seq_times_gap: {seq_times_gap}')

        high_items_emb = self.click_emb(high_items_pad)
        high_times_emb = self.timestamp_fc(torch.log(high_times_gap.unsqueeze(2)+1.0))
        high_ratings_emb = self.rating_emb(torch.tensor(2, device=self.device)).view(1, 1, -1).expand(high_items_emb.size(0), high_items_emb.size(1), -1)
        click_emb = torch.cat([high_items_emb, high_times_emb, high_ratings_emb], dim=2)
        click_emb = self.click_fc(click_emb)

        sep_emb = self.exposure_emb(torch.tensor(0, device=self.device))
        logging.info(f'sep_emb: {sep_emb.shape}, {sep_emb}')
        seq_items_emb = self.exposure_emb(seq_items_pad)
        seq_times_emb = self.timestamp_fc(torch.log(seq_times_gap.unsqueeze(2)+1.0))
        seq_ratings_emb = self.rating_emb(seq_ratings_pad)
        exposure_emb = torch.cat([seq_items_emb, seq_times_emb, seq_ratings_emb], dim=2)
        exposure_emb = self.exposure_fc(exposure_emb)


        logging.info(f'high_items_emb.shape: {high_items_emb.shape}')
        logging.info(f'high_times_emb.shape: {high_times_emb.shape}')
        logging.info(f'high_ratings_emb.shape: {high_ratings_emb.shape}')
        logging.info(f'click_emb.shape: {click_emb.shape}')

        logging.info(f'seq_items_emb.shape: {seq_items_emb.shape}')
        logging.info(f'seq_times_emb.shape: {seq_times_emb.shape}')
        logging.info(f'seq_ratings_emb.shape: {seq_ratings_emb.shape}')
        logging.info(f'exposure_emb.shape: {exposure_emb.shape}')