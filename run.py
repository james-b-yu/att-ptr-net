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
from german_parser.util import get_progress_bar
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

model = TigerModel(
    word_embedding_params=TigerModel.WordEmbeddingParams(char_set=character_set, char_flag_generators=character_flag_generators, char_internal_embedding_dim=100,
                                   char_part_embedding_dim=150, 
                                   word_part_embedding_dim=200, 
                                   char_internal_window_size=3,
                                   word_dict=inverse_word_dict,
                                   unk_rate=0.2),
    enc_lstm_params=TigerModel.LSTMParams(
        hidden_size=512,
        bidirectional=True,
        num_layers=3),
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

    morph_pos_interaction_dim=64,
    morph_prop_classes=[len(inverse_morph_dicts[prop]) for prop in CONSTS["morph_props"]],

    num_biaffine_attention_classes=2,

    num_constituent_labels=len(inverse_sym_dict),
    num_terminal_poses=len(inverse_pos_dict),

    enc_attachment_mlp_dim=128,
    dec_attachment_mlp_dim=64,
    max_attachment_order=max(train_dataloader.dataset.attachment_orders.max(), dev_dataloader.dataset.attachment_orders.max()) + 1
    )
model = model.to(device=DEVICE_NAME, dtype=torch.half) # type: ignore

print(f"Model has {sum([p.numel() for p in model.parameters()])} parameters")

optim = torch.optim.SGD(model.parameters(), lr=5e-2, weight_decay=1e-4, momentum=0.9, nesterov=True) #, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=4, gamma=0.9, last_epoch=-1)

num_epochs = 100

train_total_sentences = len(train_dataloader.dataset)
dev_total_sentences = len(dev_dataloader.dataset)

total_iteration_train = 0
total_iteration_dev = 0

filename_prefix = f"tiger_model_{strftime('%Y_%m_%d-%I_%M_%S_%p')}"

def get_filename(epoch: int):
    """get filename of model (without extension)

    Args:
        epoch (int): 0-indexed epoch number
    """

    return f"{filename_prefix}_epoch_{epoch + 1}"

for i in range(num_epochs):
    for training in (True, None, False):
        if training is None:
            # save the model
            filename = f"{CONSTS['model_dir']}/{get_filename(i)}.pickle"
            pickle.dump(model, open(filename, "wb"))
            continue

        if training == "f1":
            # calculate f1 scores
            pass
        
        assert training in (True, False)

        sum_sentences = 0
        epoch_length = len(train_dataloader if training else dev_dataloader)
        epoch_start_time = time()

        epoch_attention_loss = 0
        epoch_label_loss = 0
        epoch_pos_loss = 0
        epoch_order_loss = 0
        epoch_morph_loss = 0
        epoch_total_loss = 0

        brackets = []
        brackets_structure = []
        t_brackets = []
        t_brackets_structure = []

        new_words_dict = train_new_words if training else dev_new_words

        input: tuple[torch.Tensor, ...]

        optim.zero_grad(set_to_none=True)
        for j, input in (enumerate(train_dataloader) if training else enumerate(dev_dataloader)):
            if j > MAX_ITER:
                continue

            if training:
                model.train()
            else:
                model.eval()

            sentence_lengths, words, target_heads, target_syms, target_poses, target_attachment_orders, *target_morphs = input
            batch_size = words.shape[0]
            sum_sentences += batch_size

            words = words.to(device=DEVICE_NAME)
            target_heads = target_heads.to(device=DEVICE_NAME)
            target_syms = target_syms.to(device=DEVICE_NAME)
            target_poses = target_poses.to(device=DEVICE_NAME)
            target_attachment_orders = target_attachment_orders.to(device=DEVICE_NAME)

            for m, target_morph in enumerate(target_morphs):
                target_morphs[m] = target_morph.to(device=DEVICE_NAME)

            self_attention, labels, poses, attachment_orders, *morphs, indices = model((words, sentence_lengths), new_words_dict)

            loss_attention = F.cross_entropy(self_attention[indices], target_heads[indices])
            loss_labels    = F.cross_entropy(labels[indices, target_heads[indices]], target_syms[indices])
            loss_poses     = F.cross_entropy(poses[indices, target_heads[indices]], target_poses[indices])
            loss_orders    = F.cross_entropy(attachment_orders[indices, target_heads[indices]], target_attachment_orders[indices])

            loss_morph = torch.zeros(1, device=DEVICE_NAME, requires_grad=True)

            for m, morph_porp in enumerate(CONSTS["morph_props"]):
                morph_pred_flattened = morphs[m][indices, target_heads[indices], target_poses[indices]]
                morph_target_flattened = target_morphs[m][indices]

                loss_morph = loss_morph + F.cross_entropy(morph_pred_flattened, morph_target_flattened)


            loss = (loss_attention + loss_labels + loss_poses + loss_orders + loss_morph)

            # perform backpropagation
            if training:
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()
                optim.zero_grad(set_to_none=True)

                total_iteration_train += 1

                # detach and empty cache
                loss_attention.detach_()
                loss_labels.detach_()
                loss_poses.detach_()
                loss_orders.detach_()
                loss_morph.detach_()
                loss.detach_()
            else:
                total_iteration_dev += 1

            torch.cuda.empty_cache()
            # save metrics
            with torch.no_grad():
                epoch_attention_loss += loss_attention.item()
                epoch_label_loss += loss_labels.item()
                epoch_pos_loss += loss_poses.item()
                epoch_order_loss += loss_orders.item()
                epoch_morph_loss += loss_morph.item()
                epoch_total_loss += loss.item()

                progress = sum_sentences / (train_total_sentences if training else dev_total_sentences)
                eta_seconds = round((time() - epoch_start_time) * (1 - progress) / progress)
                eta_time = strftime("%H:%M", gmtime(time() + eta_seconds))
                eta_str = timedelta(seconds=eta_seconds)
                speed = round(sum_sentences / (time() - epoch_start_time))

                print(f"EPOCH {i + 1:5d} {'TRN' if training else 'DEV'} {get_progress_bar(progress, 20)} ({100 * progress:6.2f}%) ATTENTION {loss_attention.item():6.4e} LABEL {loss_labels.item():6.4e} POS {loss_poses.item():6.4e} ORDERS {loss_orders.item():6.4e} MORPH {loss_morph.item():6.4e} TOTAL {loss.item():6.4e} LR {scheduler.get_last_lr()[0]:4.2e} ETA {eta_str} @ {eta_time} ({speed:4d} ex s⁻¹)", end="\r", flush=True)


                for name, iteration in [(f"epoch_{i + 1}_{'trn' if training else 'dev'}", j), (f"all_epochs_{'trn' if training else 'dev'}", total_iteration_train if training else total_iteration_dev)]:
                    summary_writer.add_scalars(name, {
                        "loss_attention": loss_attention,
                        "loss_labels": loss_labels,
                        "loss_poses": loss_poses,
                        "loss_orders": loss_orders,
                        "loss_morph": loss_morph,
                        "loss_total": loss
                    }, iteration)

                # now perform f1 evaluation
                rate = CONSTS["train_tree_rate"] if training else CONSTS["dev_tree_rate"]
                if random.random() <= rate: # only generate trees for this batch (100 * rate) % of the time
                    best_edges, labels_best_edges, attachment_orders_best_edges, (edges, joint_logits) = model._find_tree(sentence_lengths, self_attention, labels, attachment_orders, indices)

                    for s_num in range(batch_size):
                        try:
                            tree_words = words[s_num, :sentence_lengths[s_num]].to("cpu")

                            tree_heads = best_edges[s_num, :sentence_lengths[s_num]].to("cpu")
                            tree_syms = labels_best_edges[s_num, :sentence_lengths[s_num]].to("cpu")
                            tree_attachment_orders = attachment_orders_best_edges[s_num, :sentence_lengths[s_num]].to("cpu")

                            t_tree_heads = target_heads[s_num, :sentence_lengths[s_num]].to("cpu")
                            t_tree_syms = target_syms[s_num, :sentence_lengths[s_num]].to("cpu")
                            t_tree_attachment_orders = target_attachment_orders[s_num, :sentence_lengths[s_num]].to("cpu")

                            the_sentence = [inverse_word_dict[w.item()] if w > 0 else new_words_dict[-w.item()] for w in tree_words]

                            c_tree = ConstituentTree.from_collection(heads=tree_heads, syms=[inverse_sym_dict[l.item()] for l in tree_syms], orders=tree_attachment_orders, words=the_sentence)
                            t_c_tree = ConstituentTree.from_collection(heads=t_tree_heads, syms=[inverse_sym_dict[l.item()] for l in t_tree_syms], orders=t_tree_attachment_orders, words=the_sentence)

                            brackets.append(c_tree.get_bracket(zero_indexed=True, ignore_words=True))
                            brackets_structure.append(c_tree.get_bracket(ignore_all_syms=True, zero_indexed=True, ignore_words=True))

                            t_brackets.append(t_c_tree.get_bracket(zero_indexed=True, ignore_words=True))
                            t_brackets_structure.append(t_c_tree.get_bracket(ignore_all_syms=True, zero_indexed=True, ignore_words=True))
                            
                        except Exception as e:
                            pass

        # calculate epoch-level metrics
        epoch_attention_loss /= epoch_length
        epoch_label_loss     /= epoch_length
        epoch_pos_loss       /= epoch_length
        epoch_order_loss     /= epoch_length
        epoch_morph_loss     /= epoch_length
        epoch_total_loss     /= epoch_length

        summary_writer.add_scalars(f"epoch_{'trn' if training else 'dev'}_losses", {
            "attention": epoch_attention_loss,
            "label": epoch_label_loss,
            "pos": epoch_pos_loss,
            "order": epoch_order_loss,
            "morph": epoch_morph_loss,
            "total": epoch_total_loss
        }, i + 1)

        brackets_filename = f"{CONSTS['eval_dir']}/{get_filename(i)}.txt"
        brackets_gold_filename = f"{CONSTS['eval_dir']}/{get_filename(i)}_gold.txt"
        brackets_structure_filename = f"{CONSTS['eval_dir']}/{get_filename(i)}_structure.txt"
        brackets_structure_gold_filename = f"{CONSTS['eval_dir']}/{get_filename(i)}_gold_structure.txt"

        open(brackets_filename, "w").write("\n".join(brackets))
        open(brackets_gold_filename, "w").write("\n".join(t_brackets))

        open(brackets_structure_filename, "w").write("\n".join(brackets_structure))
        open(brackets_structure_gold_filename, "w").write("\n".join(t_brackets_structure))

        if brackets:
            brackets_res = subprocess.run(["discodop",
                                           "eval",
                                           brackets_gold_filename,
                                           brackets_filename,
                                           "--fmt=discbracket",
                                           CONSTS["discodop_config_file"]
                                           ],
                                        capture_output=True,
                                        text=True).stdout
            brackets_structure_res = subprocess.run(["discodop",
                                                     "eval",
                                                     brackets_structure_gold_filename,
                                                     brackets_structure_filename,
                                                     "--fmt=discbracket",
                                                     CONSTS["discodop_config_file"]
                                                     ],
                                                capture_output=True,
                                                text=True).stdout
            disc_brackets_res = subprocess.run(["discodop",
                                           "eval",
                                           brackets_gold_filename,
                                           brackets_filename,
                                           "--fmt=discbracket",
                                           "--disconly",
                                           CONSTS["discodop_config_file"]
                                           ],
                                        capture_output=True,
                                        text=True).stdout
            disc_brackets_structure_res = subprocess.run(["discodop",
                                                     "eval",
                                                     brackets_structure_gold_filename,
                                                     brackets_structure_filename,
                                                     "--fmt=discbracket",
                                                     "--disconly",
                                                     CONSTS["discodop_config_file"]
                                                     ],
                                                capture_output=True,
                                                text=True).stdout

            summary_writer.add_scalars(f"epoch_{'trn' if training else 'dev'}_f1", {
                label: float(re.search(r".*labeled f-measure.*?([\d.]|nan)+.*?([\d.]+|nan)", res).groups()[1])
                for (label, res) in [
                    ("full", brackets_res),
                    ("structure", brackets_structure_res),
                    ("full_disc", disc_brackets_res),
                    ("structure_disc", disc_brackets_structure_res)
                ]}, i + 1)
            
    # update the learning rate
    scheduler.step()

print("\n", flush=True)