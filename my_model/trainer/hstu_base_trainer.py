from absl import flags
import sys
import os
import logging
import json
from datetime import date
import time
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
import itertools

import torch
from torch import nn
import torch.optim as optim


# from model.HSTU import HSTU
from model.sequential.losses.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
)
from model.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from model.sequential.features import (
    movielens_seq_features_from_row,
)
from model.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)

from data.reco_dataset import get_reco_dataset
from data.data_loader import create_data_loader


class HSTUBaseTrainer:
    def __init__(self):
        self.define_flags()


    def define_flags(self):
        # setup params
        flags.DEFINE_string('model', 'HSTU', 'HSTU/HSTU_nsa')
        flags.DEFINE_string('mode', None, 'mode')
        flags.DEFINE_string('device', 'cpu', 'device')
        # shared params
        flags.DEFINE_string('model_args', '{}', 'model args in json')
        flags.DEFINE_string('dataset_name', 'ml-20m', 'dataset name')
        flags.DEFINE_bool('use_binary_ratings', False, 'multi/binary ratings')
        flags.DEFINE_integer('num_ratings', 5, 'num_ratings')
        flags.DEFINE_integer('max_seq_len', 200, 'max seq len')
        flags.DEFINE_float('positional_sampling_ratio', 1.0, 'only used for ml-1m')
        flags.DEFINE_integer('embedding_dim', 200, 'embedding_dim')
        # train params
        flags.DEFINE_string('train_data_dir', None, 'train_data_dir')
        flags.DEFINE_integer('num_epochs', -1, 'num_epochs')
        flags.DEFINE_integer('eval_interval', -1, 'eval_interval')
        flags.DEFINE_integer('train_batch_size', -1, 'train_batch_size')
        flags.DEFINE_float('learning_rate', 0, 'learning_rate')
        flags.DEFINE_string('loss_module', 'SampledSoftmaxLoss', 'loss: BCELoss/SampledSoftmaxLoss')
        flags.DEFINE_float('temperature', 0.05, 'temperature')
        flags.DEFINE_integer('num_negatives', 1, 'for SSoftmaxLoss')
        flags.DEFINE_bool('loss_activation_checkpoint', False, 'for SSoftmaxLoss')
        flags.DEFINE_string('sampling_strategy', 'in-batch', 'for SSoftmaxLoss: in-batch/local')
        flags.DEFINE_bool('item_l2_norm', False, 'for SSoftmaxLoss')
        flags.DEFINE_float('l2_norm_eps', 1e-6, 'for SSoftmaxLoss')
        flags.DEFINE_float('weight_decay', 1e-3, 'weight decay')
        flags.DEFINE_integer('accum_steps', 1, 'accum_steps')
        flags.DEFINE_integer('presort_steps', None, 'run kmeans every x step')
        # test params
        flags.DEFINE_string('test_data_dir', None, 'test_data_dir')
        flags.DEFINE_integer('eval_batch_size', -1, 'eval_batch_size')
        self.FLAGS = flags.FLAGS


    def run(self):
        logging.info(f'self.FLAGS.model: {self.FLAGS.model}')
        if self.FLAGS.model == 'HSTU':
            from model.HSTU import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_nsa':
            from model.HSTU_nsa import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_interleave':
            from model.HSTU_interleave import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_fuxian':
            from model.HSTU_fuxian import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_pretrain':
            from model.HSTU_pretrain import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_nsa_pretrain':
            from model.HSTU_nsa_pretrain import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_interleave_pretrain':
            from model.HSTU_interleave_pretrain import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_bsa_pretrain':
            from model.HSTU_bsa_pretrain import HSTU
            self.model_cls = HSTU
        elif self.FLAGS.model == 'HSTU_bsa_pretrain_interleave':
            from model.HSTU_bsa_pretrain_interleave import HSTU
            self.model_cls = HSTU
        else:
            pass
        if self.FLAGS.mode == 'train':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.train()
        elif self.FLAGS.mode == 'test':
            logging.info(f'mode: {self.FLAGS.mode}')
        elif self.FLAGS.mode == 'dev':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.dev()
        elif self.FLAGS.mode == 'test_dev':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.test_dev()
        elif self.FLAGS.mode == 'train_presort':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.train_presort()

    
    def get_model(self):
        self.model_args = json.loads(self.FLAGS.model_args)
        logging.info(f'self.model_cls: {self.model_cls}')
        self.model = self.model_cls(**self.model_args)
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Total Parameters: {total_params}")
        logging.info(f"Trainable Parameters: {trainable_params}")

        self.embedding_module = LocalEmbeddingModule(
            num_items=self.dataset.max_item_id,
            item_embedding_dim=self.FLAGS.embedding_dim,
        )
        self.embedding_module.to(self.device)


    def train(self):
        self.device = self.FLAGS.device
        self.get_dataset()
        self.get_model()
        self.get_loss()
        logging.info(f'model structure: {self.model}')
        self.accum_steps = self.FLAGS.accum_steps

        try:
            num_batches = len(self.train_data_loader)
        except:
            num_batches = float('inf')

        batch_id = 0
        epoch = 0
        
        # 1. 确保循环开始前梯度清零
        self.optimizer.zero_grad() 
        
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            
            for batch_id, row in enumerate(iter(self.train_data_loader)):
                # train
                logging.info(f'batch: {batch_id}')
                # logging.info(f'row: {row}')
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=0, 
                )

                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                jagged_out, out_offsets = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                pred_logits = jagged_out[::2, :].reshape(-1, 2)
                raw_targets = seq_features.past_payloads['ratings'].long()
                MaxLen = raw_targets.shape[1]
                col_indices = torch.arange(MaxLen, device=raw_targets.device).unsqueeze(0)
                valid_mask = col_indices <= (seq_features.past_lengths-1).unsqueeze(1)
                targets = raw_targets[valid_mask]
                
                loss = self.criterion(pred_logits, (targets-1).squeeze())
                
                loss_to_display = loss.item()
                loss = loss / self.accum_steps
                
                loss.backward()
                is_update_step = ((batch_id + 1) % self.accum_steps == 0) or ((batch_id + 1) == num_batches)
                
                if is_update_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
            # End of Epoch
            logging.info(f'start testing!')
            avg_loss, avg_acc, avg_binary_acc, global_auc = self.test()
            logging.info(f"[Eval] End of Epoch {epoch}: TrainLoss={loss_to_display:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, BinaryAcc={avg_binary_acc:.4f}, AUC={global_auc:.4f}")
            self.embedding_module.train()
            self.model.train()

        torch.save(
            {
                "epoch": epoch,
                "embedding_state_dict": self.embedding_module.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"./ckpts/test.pt"
        )
        return


    def get_loss(self):
        self.criterion = nn.CrossEntropyLoss()
        model_params = list(self.model.parameters()) + list(self.embedding_module.parameters())
        self.optimizer = optim.Adam(model_params, lr=self.FLAGS.learning_rate)


    def get_sampler(self):
        sampling_strategy = self.FLAGS.sampling_strategy
        item_l2_norm = self.FLAGS.item_l2_norm
        l2_norm_eps = self.FLAGS.l2_norm_eps
        if sampling_strategy == "in-batch":
            self.negatives_sampler = InBatchNegativesSampler(
                l2_norm=item_l2_norm,
                l2_norm_eps=l2_norm_eps,
                dedup_embeddings=True,
            )
        elif sampling_strategy == "local":
            self.negatives_sampler = LocalNegativesSampler(
                num_items=self.dataset.max_item_id,
                item_emb=self.model._embedding_module._item_emb,
                all_item_ids=self.dataset.all_item_ids,
                l2_norm=item_l2_norm,
                l2_norm_eps=l2_norm_eps,
            )
        else:
            raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")


    def get_dataset(self):
        self.dataset = get_reco_dataset(
            dataset_name=self.FLAGS.dataset_name,
            max_sequence_length=self.FLAGS.max_seq_len,
            chronological=True,
            positional_sampling_ratio=self.FLAGS.positional_sampling_ratio,
            use_binary_ratings=self.FLAGS.use_binary_ratings,
            num_ratings=self.FLAGS.num_ratings,
        )
        logging.info(f'dataset.max_item_id: {self.dataset.max_item_id}')
        self.train_data_sampler, self.train_data_loader = create_data_loader(
            self.dataset.train_dataset,
            batch_size=self.FLAGS.train_batch_size,
            world_size=1,
            rank=0,
            shuffle=True,
            drop_last=False,
        )
        self.eval_data_sampler, self.eval_data_loader = create_data_loader(
            self.dataset.eval_dataset,
            batch_size=self.FLAGS.eval_batch_size,
            world_size=1,
            rank=0,
            shuffle=True,  # needed for partial eval
            drop_last=False,
        )
        logging.info(f'train_dataloader_num: {len(self.train_data_loader)}')
        logging.info(f'eval_dataloader_num: {len(self.eval_data_loader)}')


    def get_loss_with_bce(self):
        # 根据 flag 选择 Loss 函数
        if self.FLAGS.use_binary_ratings:
            logging.info("Using BCEWithLogitsLoss for Binary Classification")
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            # 原逻辑：多分类使用 CrossEntropyLoss
            # 注意：如果你用SampledSoftmax，这里需要根据 flag.loss_module 再细分，这里默认回退到标准CE
            logging.info("Using CrossEntropyLoss for Multi-class Classification")
            self.criterion = nn.CrossEntropyLoss()
            
        model_params = list(self.model.parameters()) + list(self.embedding_module.parameters())
        self.optimizer = optim.Adam(model_params, lr=self.FLAGS.learning_rate)


    def dev(self):
        self.device = self.FLAGS.device
        self.get_dataset()
        self.get_model()
        self.get_loss()
        logging.info(f'model structure: {self.model}')
        self.accum_steps = self.FLAGS.accum_steps

        try:
            num_batches = len(self.train_data_loader)
        except:
            num_batches = float('inf')

        batch_id = 0
        epoch = 0
        
        # 1. 确保循环开始前梯度清零
        self.optimizer.zero_grad() 
        
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            
            for batch_id, row in enumerate(iter(self.train_data_loader)):
                # train
                logging.info(f'batch: {batch_id}')
                # logging.info(f'row: {row}')
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=0, 
                )

                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                jagged_out, out_offsets = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                pred_logits = jagged_out[::2, :].reshape(-1, 2)
                raw_targets = seq_features.past_payloads['ratings'].long()
                MaxLen = raw_targets.shape[1]
                col_indices = torch.arange(MaxLen, device=raw_targets.device).unsqueeze(0)
                valid_mask = col_indices <= (seq_features.past_lengths-1).unsqueeze(1)
                targets = raw_targets[valid_mask]
                
                loss = self.criterion(pred_logits, (targets-1).squeeze())
                
                loss_to_display = loss.item()
                loss = loss / self.accum_steps
                
                loss.backward()
                is_update_step = ((batch_id + 1) % self.accum_steps == 0) or ((batch_id + 1) == num_batches)
                
                if is_update_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
            # End of Epoch
            logging.info(f'start testing!')
            avg_loss, avg_acc, avg_binary_acc, global_auc = self.test()
            logging.info(f"[Eval] End of Epoch {epoch}: TrainLoss={loss_to_display:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, BinaryAcc={avg_binary_acc:.4f}, AUC={global_auc:.4f}")
            self.embedding_module.train()
            self.model.train()

        torch.save(
            {
                "epoch": epoch,
                "embedding_state_dict": self.embedding_module.state_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            f"./ckpts/test.pt"
        )
        return


    def test(self):
        self.embedding_module.eval()
        self.model.eval()
        
        batch_losses = []
        batch_accs = []         # Exact Match Accuracy
        batch_binary_accs = []  # Binary Accuracy
        
        all_pos_probs = []
        all_binary_targets = []
        
        with torch.no_grad():
            for row in iter(self.eval_data_loader):
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=0, 
                )
                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                jagged_out, x_offsets = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                outputs = jagged_out[x_offsets[1:]-2]
                
                # --- 通用 Label 处理 ---
                # 如果是 1-5分，变为 0-4
                # 如果是 1-2(Binary)，变为 0-1 (0:False, 1:True)
                targets = (target_ratings - 1).view(-1)
                
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                
                # --- 计算 Exact Match Accuracy ---
                # 在二分类模式下，这个值等于 Binary Acc
                pred_ids = torch.argmax(outputs, dim=1)
                acc = (pred_ids == targets).float().mean().item()
                batch_accs.append(acc)
                
                # --- 分支逻辑：处理二分类指标所需的概率和标签 ---
                probs = torch.softmax(outputs, dim=1)
                
                if self.FLAGS.use_binary_ratings:
                    # === 模式 A: Binary 输入 (Rating 1, 2) ===
                    # 此时 outputs 的 shape 应该是 [B, 2]
                    # Index 0 是负例(1分)，Index 1 是正例(2分)
                    
                    # 正类概率就是 Index 1 的概率
                    pos_probs_batch = probs[:, 1]
                    
                    # 真实标签已经是 0/1 了，无需转换
                    binary_targets_batch = targets.float()
                    
                else:
                    # === 模式 B: 5-Star 输入 (Rating 1-5) ===
                    # 此时 outputs 的 shape 应该是 [B, 5]
                    
                    # 正类概率是 4星(index 3) 和 5星(index 4) 之和
                    if probs.shape[1] >= 5:
                        pos_probs_batch = probs[:, 3:].sum(dim=1)
                    else:
                        pos_probs_batch = probs[:, -1] # 兜底逻辑
                    
                    # 真实标签需要手动截断 (>=3 为正例)
                    binary_targets_batch = (targets >= 3).float()
                
                # --- 统一计算 Binary Accuracy 和 收集 AUC 数据 ---
                # 此时 pos_probs_batch 和 binary_targets_batch 已经对齐了
                
                # 阈值取 0.5 判断
                binary_preds = (pos_probs_batch >= 0.5).float()
                binary_acc = (binary_preds == binary_targets_batch).float().mean().item()
                batch_binary_accs.append(binary_acc)
                
                all_pos_probs.extend(pos_probs_batch.cpu().numpy().tolist())
                all_binary_targets.extend(binary_targets_batch.cpu().numpy().tolist())
        
        avg_loss = np.mean(batch_losses)
        avg_acc = np.mean(batch_accs)
        avg_binary_acc = np.mean(batch_binary_accs)
        
        try:
            global_auc = roc_auc_score(all_binary_targets, all_pos_probs)
        except ValueError:
            logging.warning("AUC calc failed: valid set needs both pos and neg samples.")
            global_auc = 0.5

        print(f"Debug: 预测为正例的比例: {binary_preds.mean().item():.2f}")
        return avg_loss, avg_acc, avg_binary_acc, global_auc


    def test_dev(self):
        return


    def test_with_binary_acc(self):
        self.embedding_module.eval()
        self.model.eval()
        
        batch_losses = []
        batch_accs = []         # Exact Match Accuracy
        batch_binary_accs = []  # Binary Accuracy
        
        all_pos_probs = []
        all_binary_targets = []
        
        with torch.no_grad():
            for row in iter(self.eval_data_loader):
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=0, 
                )
                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                outputs = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                
                # --- 通用 Label 处理 ---
                # 如果是 1-5分，变为 0-4
                # 如果是 1-2(Binary)，变为 0-1 (0:False, 1:True)
                targets = (target_ratings - 1).view(-1)
                
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                
                # --- 计算 Exact Match Accuracy ---
                # 在二分类模式下，这个值等于 Binary Acc
                pred_ids = torch.argmax(outputs, dim=1)
                acc = (pred_ids == targets).float().mean().item()
                batch_accs.append(acc)
                
                # --- 分支逻辑：处理二分类指标所需的概率和标签 ---
                probs = torch.softmax(outputs, dim=1)
                
                if self.FLAGS.use_binary_ratings:
                    # === 模式 A: Binary 输入 (Rating 1, 2) ===
                    # 此时 outputs 的 shape 应该是 [B, 2]
                    # Index 0 是负例(1分)，Index 1 是正例(2分)
                    
                    # 正类概率就是 Index 1 的概率
                    pos_probs_batch = probs[:, 1]
                    
                    # 真实标签已经是 0/1 了，无需转换
                    binary_targets_batch = targets.float()
                    
                else:
                    # === 模式 B: 5-Star 输入 (Rating 1-5) ===
                    # 此时 outputs 的 shape 应该是 [B, 5]
                    
                    # 正类概率是 4星(index 3) 和 5星(index 4) 之和
                    if probs.shape[1] >= 5:
                        pos_probs_batch = probs[:, 3:].sum(dim=1)
                    else:
                        pos_probs_batch = probs[:, -1] # 兜底逻辑
                    
                    # 真实标签需要手动截断 (>=3 为正例)
                    binary_targets_batch = (targets >= 3).float()
                
                # --- 统一计算 Binary Accuracy 和 收集 AUC 数据 ---
                # 此时 pos_probs_batch 和 binary_targets_batch 已经对齐了
                
                # 阈值取 0.5 判断
                binary_preds = (pos_probs_batch >= 0.5).float()
                binary_acc = (binary_preds == binary_targets_batch).float().mean().item()
                batch_binary_accs.append(binary_acc)
                
                all_pos_probs.extend(pos_probs_batch.cpu().numpy().tolist())
                all_binary_targets.extend(binary_targets_batch.cpu().numpy().tolist())
        
        avg_loss = np.mean(batch_losses)
        avg_acc = np.mean(batch_accs)
        avg_binary_acc = np.mean(batch_binary_accs)
        
        try:
            global_auc = roc_auc_score(all_binary_targets, all_pos_probs)
        except ValueError:
            logging.warning("AUC calc failed: valid set needs both pos and neg samples.")
            global_auc = 0.5

        print(f"Debug: 预测为正例的比例: {binary_preds.mean().item():.2f}")
        return avg_loss, avg_acc, avg_binary_acc, global_auc


    def train_presort(self):
        self.device = self.FLAGS.device
        self.get_dataset_presort()

        self.get_model()
        self.get_loss()
        logging.info(f'model structure: {self.model}')
        self.accum_steps = self.FLAGS.accum_steps
        self.presort_steps = self.FLAGS.presort_steps

        try:
            num_batches = len(self.train_data_loader)
        except:
            num_batches = float('inf')

        batch_id = 0
        epoch = 0
        self.optimizer.zero_grad() 
        
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            if (epoch > 0) and (epoch % self.presort_steps==0):
                logging.info(f'epoch {epoch}: starting presort data!')
                self.get_dataset_presort(block_size=16, emb_matrix=self.embedding_module)
                logging.info(f'epoch {epoch}: finishing presort data!')
                return
            

    def get_dataset_presort(self, block_size=None, emb_matrix=None):
        from data.reco_dataset_v1 import get_reco_dataset
        self.dataset = get_reco_dataset(
            dataset_name=self.FLAGS.dataset_name,
            max_sequence_length=self.FLAGS.max_seq_len,
            chronological=True,
            positional_sampling_ratio=self.FLAGS.positional_sampling_ratio,
            use_binary_ratings=self.FLAGS.use_binary_ratings,
            num_ratings=self.FLAGS.num_ratings,
        )
        if (block_size is not None) and (emb_matrix is not None):
            self.device = self.FLAGS.device
            self.dataset.presort(block_size, emb_matrix, self.device)
        else:
            logging.info(f'dataset.max_item_id: {self.dataset.max_item_id}')
        self.train_data_sampler, self.train_data_loader = create_data_loader(
            self.dataset.train_dataset,
            batch_size=self.FLAGS.train_batch_size,
            world_size=1,
            rank=0,
            shuffle=True,
            drop_last=False,
        )
        self.eval_data_sampler, self.eval_data_loader = create_data_loader(
            self.dataset.eval_dataset,
            batch_size=self.FLAGS.eval_batch_size,
            world_size=1,
            rank=0,
            shuffle=True,  # needed for partial eval
            drop_last=False,
        )
        logging.info(f'train_dataloader_num: {len(self.train_data_loader)}')
        logging.info(f'eval_dataloader_num: {len(self.eval_data_loader)}')

                