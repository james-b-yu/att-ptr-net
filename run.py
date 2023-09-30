# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
import torch.nn.utils.clip_grad as utils
from german_parser.util import get_progress_bar, get_filename
from german_parser.util.epoch import one_epoch
from german_parser.util.const import CONSTS
from german_parser.util.c_and_d import ConstituentTree, DependencyTree

import random
random.seed()

import re

import subprocess

from math import ceil, floor

from torch.utils.tensorboard import SummaryWriter

DEVICE_NAME = "cuda"

(train_dataloader, train_new_words), (dev_dataloader, dev_new_words), _, character_set, character_flag_generators, inverse_word_dict, inverse_sym_dict, inverse_pos_dict, inverse_morph_dicts = pickle.load(open("required_vars.pkl", "rb"))

MAX_ITER = float("inf")

from time import time, strftime, gmtime
from datetime import timedelta

summary_writer = SummaryWriter()
from transformers.models.bert import BertLMHeadModel

bert_model: BertLMHeadModel = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-german-cased')

model = TigerModel(
    bert_model=bert_model,
    bert_embedding_dim=768,

    word_embedding_params=TigerModel.WordEmbeddingParams(
        char_set=character_set,
        char_flag_generators=character_flag_generators,
        char_internal_embedding_dim=100,
        char_part_embedding_dim=200, 
        word_part_embedding_dim=200, 
        char_internal_window_size=3,
        word_dict=inverse_word_dict,
        unk_rate=0.5),
    enc_lstm_params=TigerModel.LSTMParams(
        hidden_size=512,
        bidirectional=True,
        num_layers=5),
    dec_lstm_params=TigerModel.LSTMParams(
        hidden_size=512,
        bidirectional=False,
        num_layers=1
        ),

    enc_attention_mlp_dim=512,
    dec_attention_mlp_dim=512,
    
    enc_label_mlp_dim=128,
    dec_label_mlp_dim=128,

    enc_pos_mlp_dim=128,
    dec_pos_mlp_dim=128,

    enc_morph_mlp_dim=128,
    dec_morph_mlp_dim=128,

    morph_pos_interaction_dim=128,
    morph_prop_classes=[len(inverse_morph_dicts[prop]) for prop in CONSTS["morph_props"]],

    num_biaffine_attention_classes=2,

    num_constituent_labels=len(inverse_sym_dict),
    num_terminal_poses=len(inverse_pos_dict),

    enc_attachment_mlp_dim=128,
    dec_attachment_mlp_dim=64,
    max_attachment_order=max(train_dataloader.dataset.attachment_orders.max(), dev_dataloader.dataset.attachment_orders.max()) + 1
    )
model = model.to(device=DEVICE_NAME, dtype=torch.half) # type: ignore

print(f"Model has {sum([p.numel() for p in model.parameters()]):_} parameters".replace("_", " "))

optim = torch.optim.SGD(model.parameters(), lr=5e-2, weight_decay=1e-4, momentum=0.9, nesterov=True) #, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9, last_epoch=-1)

num_epochs = 15

train_total_sentences = len(train_dataloader.dataset)
dev_total_sentences = len(dev_dataloader.dataset)

total_iteration_train = 0
total_iteration_dev = 0

for i in range(num_epochs):
    for training in (True, None, False):
        if training is None:
            # save the model
            filename = f"{CONSTS['model_dir']}/{get_filename(i)}.pickle"
            pickle.dump(model, open(filename, "wb"))
            continue

        one_epoch(
            model=model,
            optim=optim,
            device=DEVICE_NAME,
            dataloader=train_dataloader if training else dev_dataloader,
            training=training,
            new_words_dict=train_new_words if training else dev_new_words,
            inverse_word_dict=inverse_word_dict,
            inverse_sym_dict=inverse_sym_dict,
            inverse_pos_dict=inverse_pos_dict,
            inverse_morph_dicts=inverse_morph_dicts,
            discodop_config_file=CONSTS["discodop_config_file"],
            epoch_num=i,
            eval_dir=CONSTS["eval_dir"],
            gradient_clipping=1,
            morph_props=CONSTS["morph_props"],
            pos_replacements=CONSTS["pos_replacements"],
            scheduler=scheduler,
            summary_writer=summary_writer,
            tree_gen_rate=CONSTS["train_tree_rate"] if training else CONSTS["dev_tree_rate"]
        )
            
    # update the learning rate
    scheduler.step()

print("\n", flush=True)