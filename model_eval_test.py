import sys
import os
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
import torch.nn.utils.clip_grad as utils
from german_parser.util import get_filename, print_module_parameters
from german_parser.util.epoch import one_epoch
from german_parser.util.const import CONSTS
from german_parser.util.c_and_d import ConstituentTree, DependencyTree

import random
random.seed()

import re

import subprocess

DEVICE_NAME = "cuda"
DTYPE = torch.half

from math import ceil, floor

from torch.utils.tensorboard import SummaryWriter

for model_num in range(5, 27):
    print(f"loading model {model_num}")
    model_dir = f"/mnt/d/ML/att_ptr_net/models/{model_num}"

    model_list = [os.path.join(model_dir, f) for f in os.listdir(model_dir)]
    model_list.sort(key=lambda x: os.path.getmtime(x))

    print(f"loading model {model_num} file name {model_list[-1]}")

    try:
        model = pickle.load(open(model_list[-1], "rb"))
    except:
        continue

    print(f"loading data {model_num}")
    _, _, (test_dataloader, test_new_words), _, _, inverse_word_dict, inverse_sym_dict, inverse_pos_dict, inverse_morph_dicts = pickle.load(open(f"/mnt/d/ML/att_ptr_net/data/required_vars_{model_num}.pkl", "rb"))

    optim = torch.optim.SGD(model.parameters(), lr=5e-2, weight_decay=1e-4, momentum=0.9, nesterov=True) #, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.8, last_epoch=-1)

    summary_writer = SummaryWriter()

    saved_metrics = one_epoch(model=model,
              optim=optim,
              device=DEVICE_NAME,
              dataloader=test_dataloader,
              training=False,
              new_words_dict=test_new_words,
              inverse_word_dict=inverse_word_dict,
              inverse_sym_dict=inverse_sym_dict,
              inverse_pos_dict=inverse_pos_dict,
              inverse_morph_dicts=inverse_morph_dicts,
              discodop_config_file=CONSTS["discodop_config_file"],
              epoch_num=0,
              eval_dir=CONSTS["eval_dir"],
              gradient_clipping=1,
              pos_replacements=CONSTS["pos_replacements"],
              scheduler=scheduler,
              summary_writer=summary_writer,
              tree_gen_rate=1)
    
    summary_writer.close()

    pickle.dump(saved_metrics, open(f"/mnt/d/ML/att_ptr_net/test_metrics/saved_metrics_{model_num}", "wb"))