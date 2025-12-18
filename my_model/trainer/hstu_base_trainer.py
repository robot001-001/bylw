from absl import flags
import sys
import os
import logging
import json
from datetime import date
import time

import torch
from torch import nn


from model.HSTU import HSTU
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
        # test params
        flags.DEFINE_string('test_data_dir', None, 'test_data_dir')
        flags.DEFINE_integer('eval_batch_size', -1, 'eval_batch_size')
        self.FLAGS = flags.FLAGS


    def run(self):
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
        self.model = HSTU(**self.model_args)
        self.model.to(self.device)

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
        self.model.to(self.device)
        self.ar_loss.to(self.device)
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.FLAGS.learning_rate,
            betas=(0.9, 0.98),
            weight_decay=self.FLAGS.weight_decay,
        )
        date_str = date.today().strftime("%Y-%m-%d")
        model_subfolder = f"{self.FLAGS.dataset_name}-l{self.FLAGS.max_sequence_length}"
        os.makedirs(f"./exps/{model_subfolder}", exist_ok=True)
        os.makedirs(f"./ckpts/{model_subfolder}", exist_ok=True)
        
        last_training_time = time.time()
        torch.autograd.set_detect_anomaly(True)

        batch_id = 0
        epoch = 0
        for epoch in range(self.FLAGS.num_epochs):
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            if self.eval_data_sampler is not None:
                self.eval_data_sampler.set_epoch(epoch)
            self.model.train()
            for row in iter(self.train_data_loader):
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=11,
                )


    def get_loss(self):
        loss_module = self.FLAGS.loss_module
        temperature = self.FLAGS.temperature
        num_negatives = self.FLAGS.num_negatives
        loss_activation_checkpoint = self.FLAGS.loss_activation_checkpoint
        if loss_module == "BCELoss":
            assert temperature == 1.0
            ar_loss = BCELoss(temperature=temperature, model=self.model)
        elif loss_module == "SampledSoftmaxLoss":
            ar_loss = SampledSoftmaxLoss(
                num_to_sample=num_negatives,
                softmax_temperature=temperature,
                model=self.model,
                activation_checkpoint=loss_activation_checkpoint,
            )
        else:
            raise ValueError(f"Unrecognized loss module {loss_module}.")
        self.ar_loss = ar_loss


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


    def dev(self):
        self.device = self.FLAGS.device
        self.get_dataset()
        self.get_model()
        logging.info(f'model structure: {self.model}')

        batch_id = 0
        epoch = 0
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            for row in iter(self.train_data_loader):
                logging.info(f'row info: {row}')
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=11,
                )
                logging.info(f'seq_features: {seq_features}')
                logging.info(f'target_ids: {target_ids}')
                logging.info(f'target_ratings: {target_ratings}')
                seq_features.past_ids.scatter_(
                    dim=1,
                    index=seq_features.past_lengths.view(-1, 1),
                    src=target_ids.view(-1, 1),
                )
                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                ret = self.model(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                logging.info(f'ret: {ret}')
                break
        return


    def test_dev(self):
        self.device = self.FLAGS.device
        self.get_dataset()
        self.get_model()
        from model.sequential.input_features_preprocessors import CombinedItemAndRatingInputFeaturesPreprocessorV1
        _input_features_preproc = CombinedItemAndRatingInputFeaturesPreprocessorV1(
            max_sequence_len = 211, # 200历史序列+1tgt
            item_embedding_dim = 50,
            dropout_rate = 0.2,
            num_ratings=5
        ).to(self.device)
        batch_id = 0
        epoch = 0
        for epoch in range(self.FLAGS.num_epochs):
            logging.info(f'num_epochs: {self.FLAGS.num_epochs}, current: {epoch}')
            if self.train_data_sampler is not None:
                self.train_data_sampler.set_epoch(epoch)
            for row in iter(self.train_data_loader):
                logging.info(f'row info: {row}')
                seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                    row,
                    device=self.device,
                    max_output_length=10,
                )
                logging.info(f'seq_features: {seq_features}')
                logging.info(f'seq_features: {seq_features.past_ids.shape}')
                logging.info(f'target_ids: {target_ids}')
                logging.info(f'target_ratings: {target_ratings}')
                seq_features.past_ids.scatter_(
                    dim=1,
                    index=seq_features.past_lengths.view(-1, 1),
                    src=target_ids.view(-1, 1),
                )
                input_embeddings = self.embedding_module.get_item_embeddings(seq_features.past_ids)
                # modify
                lengths, user_embeddings, valid_mask = _input_features_preproc(
                    past_lengths=seq_features.past_lengths,
                    past_ids=seq_features.past_ids,
                    past_embeddings=input_embeddings,
                    past_payloads=seq_features.past_payloads,
                )
                logging.info(f'lengths: {lengths}')
                logging.info(f'user_embeddings: {user_embeddings}')
                logging.info(f'valid_mask: {valid_mask}')
                return
            
        

