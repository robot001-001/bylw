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


class ONETRANSTrainer:
    def __init__(self):
        self.define_flags()


    def define_flags(self):
        # setup params
        flags.DEFINE_string('model', 'HSTU', 'HSTU/HSTU_nsa')
        self.FLAGS = flags.FLAGS


    def run(self):
        logging.info(f'self.FLAGS.model: {self.FLAGS.model}')
        if self.FLAGS.model == 'HSTU':
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
        pass


    def train(self):
        pass


    def get_loss(self):
        pass


    def get_dataset(self):
        pass


    def dev(self):
        pass


    def test(self):
        pass


    def test_dev(self):
        return


    