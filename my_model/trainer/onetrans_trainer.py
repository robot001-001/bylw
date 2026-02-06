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
from data.onetrans_dataloader_v3 import MovieLensFullDataset
from data.data_loader import create_data_loader
from model.onetrans_embedding import OneTransEmb


class ONETRANSTrainer:
    def __init__(self):
        self.define_flags()


    def define_flags(self):
        # setup params
        flags.DEFINE_string('model', 'ONETRANS', 'model file')
        flags.DEFINE_string('mode', None, 'mode')
        flags.DEFINE_string('device', 'cpu', 'device')
        # shared params
        flags.DEFINE_string('model_args', '{}', 'model args in json')
        flags.DEFINE_string('dataset_name', 'ml-20m', 'dataset name')
        flags.DEFINE_bool('use_binary_ratings', False, 'multi/binary ratings')
        flags.DEFINE_integer('num_ratings', 5, 'num_ratings')
        flags.DEFINE_integer('max_seq_len', 200, 'max seq len')
        flags.DEFINE_integer('embedding_dim', 200, 'embedding_dim')
        # train params
        flags.DEFINE_string('train_data_dir', None, 'train_data_dir')
        flags.DEFINE_integer('num_epochs', -1, 'num_epochs')
        flags.DEFINE_integer('eval_interval', -1, 'eval_interval')
        flags.DEFINE_integer('train_batch_size', -1, 'train_batch_size')
        flags.DEFINE_float('learning_rate', 0, 'learning_rate')
        flags.DEFINE_integer('accum_steps', 1, 'accum_steps')
        # test params
        flags.DEFINE_string('test_data_dir', None, 'test_data_dir')
        flags.DEFINE_integer('eval_batch_size', -1, 'eval_batch_size')
        self.FLAGS = flags.FLAGS


    def run(self):
        logging.info(f'self.FLAGS.model: {self.FLAGS.model}')
        if self.FLAGS.model == 'ONETRANS':
            from model.ONETRANS import ONETRANS
            self.model_cls = ONETRANS
        if self.FLAGS.mode == 'train':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.train()
        elif self.FLAGS.mode == 'test':
            logging.info(f'mode: {self.FLAGS.mode}')
        elif self.FLAGS.mode == 'dev':
            logging.info(f'mode: {self.FLAGS.mode}')
            self.dev()

    
    def get_model(self):
        self.model_args = json.loads(self.FLAGS.model_args)
        logging.info(f'self.model_cls: {self.model_cls}')
        self.model = self.model_cls(**self.model_args)
        self.model.to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logging.info(f"Total Parameters: {total_params}")
        logging.info(f"Trainable Parameters: {trainable_params}")

        self.embedding_module = OneTransEmb(
            self.max_item_id, self.max_user_id, self.FLAGS.embedding_dim, self.device
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
                logging.info(f'row: {row}')
                input_embedding = self.embedding_module(row)
                return


    def get_loss(self):
        self.criterion = nn.CrossEntropyLoss()
        model_params = list(self.model.parameters()) + list(self.embedding_module.parameters())
        self.optimizer = optim.Adam(model_params, lr=self.FLAGS.learning_rate)


    def get_dataset(self):
        if self.FLAGS.dataset_name == 'ml-1m':
            self.dataset = MovieLensFullDataset(
                f'tmp/ml-1m/sasrec_format_binary_augment_onetrans.csv',
                max_len=self.FLAGS.max_seq_len
            )
            logging.info(f'dataset.max_item_id: {self.dataset.max_item_id}')
            self.train_data_sampler, self.train_data_loader = create_data_loader(
                self.dataset,
                batch_size=self.FLAGS.train_batch_size,
                world_size=1,
                rank=0,
                shuffle=True,
                drop_last=False,
            )
            self.test_dataset = MovieLensFullDataset(
                f'tmp/ml-1m/sasrec_format_binary_onetrans_testset.csv',
                max_len=self.FLAGS.max_seq_len
            )
            self.max_item_id = self.test_dataset.max_item_id
            self.max_user_id = self.test_dataset.max_user_id
            self.eval_data_sampler, self.eval_data_loader = create_data_loader(
                self.test_dataset,
                batch_size=self.FLAGS.eval_batch_size,
                world_size=1,
                rank=0,
                shuffle=True,  # needed for partial eval
                drop_last=False,
            )
            logging.info(f'train_dataloader_num: {len(self.train_data_loader)}')
            logging.info(f'eval_dataloader_num: {len(self.eval_data_loader)}')


    def dev(self):
        pass


    def test(self):
        pass


    def test_dev(self):
        return


    