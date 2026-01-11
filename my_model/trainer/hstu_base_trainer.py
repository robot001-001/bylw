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
        else:
            pass
        if self.FLAGS.mode == 'train':
            logging.info(f'mode: {self.FLAGS.mode}')
        elif self.FLAGS.mode == 'test':
            logging.info(f'mode: {self.FLAGS.mode}')
        elif self.FLAGS.mode == 'dev':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.dev()
        elif self.FLAGS.mode == 'test_dev':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.test_dev()

    
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

        batch_id = 0
        epoch = 0
        self.optimizer.zero_grad()
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            for batch_id, row in enumerate(iter(self.train_data_loader)):
                # train
                # logging.info(f'row info: {row}')
                logging.info(f'batch: {batch_id}')
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=0, # 精排架构不需要补充
                )
                # logging.info(f'seq_features: {seq_features}')
                # logging.info(f'target_ids: {target_ids}')
                # logging.info(f'target_ratings: {target_ratings}')
                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                outputs = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                loss = self.criterion(outputs, (target_ratings-1).squeeze())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                # eval
                if (batch_id % self.FLAGS.eval_interval) == 0 and batch_id != 0:
                    logging.info(f'start testing!')
                    avg_loss, avg_acc, global_auc = self.test()
                    logging.info(f"[Eval] Step {batch_id}: TrainLoss={loss:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUC={global_auc:.4f}")
                    self.embedding_module.train()
                    self.model.train()
                # return
            avg_loss, avg_acc, global_auc = self.test()
            logging.info(f"[End of Epoch {epoch}] TrainLoss={loss:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUC={global_auc:.4f}")
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
                
                loss = self.criterion(outputs, (target_ratings-1).squeeze())
                
                loss_to_display = loss.item()
                loss = loss / self.accum_steps
                
                loss.backward()
                is_update_step = ((batch_id + 1) % self.accum_steps == 0) or ((batch_id + 1) == num_batches)
                
                if is_update_step:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # eval
                if (batch_id % (self.FLAGS.eval_interval*self.accum_steps)) == 0 and batch_id != 0:
                    logging.info(f'start testing!')
                    avg_loss, avg_acc, global_auc = self.test()
                    logging.info(f"[Eval] Step {batch_id}: TrainLoss={loss_to_display:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUC={global_auc:.4f}")
                    self.embedding_module.train()
                    self.model.train()

            # End of Epoch
            avg_loss, avg_acc, global_auc = self.test()
            logging.info(f"[End of Epoch {epoch}] TrainLoss={loss_to_display:4g}, EvalLoss={avg_loss:.4f}, Acc={avg_acc:.4f}, AUC={global_auc:.4f}")
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


    def test_dev(self):
        return
            

    def test(self):
        self.embedding_module.eval()
        self.model.eval()
        batch_losses = []
        batch_accs = []
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
                targets = (target_ratings - 1).view(-1)
                loss = self.criterion(outputs, targets)
                batch_losses.append(loss.item())
                pred_ids = torch.argmax(outputs, dim=1)
                acc = (pred_ids == targets).float().mean().item()
                batch_accs.append(acc)
                probs = torch.softmax(outputs, dim=1)
                if probs.shape[1] >= 5:
                    pos_probs_batch = probs[:, 3:].sum(dim=1)
                else:
                    pos_probs_batch = probs[:, -1]
                binary_targets_batch = (targets >= 3).float()
                all_pos_probs.extend(pos_probs_batch.cpu().numpy().tolist())
                all_binary_targets.extend(binary_targets_batch.cpu().numpy().tolist())
        avg_loss = np.mean(batch_losses)
        avg_acc = np.mean(batch_accs)
        try:
            global_auc = roc_auc_score(all_binary_targets, all_pos_probs)
        except ValueError:
            logging.warning("AUC calc failed: valid set needs both pos and neg samples.")
            global_auc = 0.5

        return avg_loss, avg_acc, global_auc



