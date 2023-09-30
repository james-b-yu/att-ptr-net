from torch.utils.data import DataLoader
from time import time
import torch
import torch.nn.functional as F


from . import TigerDataset
from ..model import TigerModel
from . import get_progress_bar, get_filename, ConstituentTree

import torch.nn.utils.clip_grad as nn_utils

import random
random.seed()
from time import time, strftime, gmtime
from datetime import timedelta
import subprocess

import re

from torch.utils.tensorboard import SummaryWriter

def one_epoch(model: TigerModel, optim: torch.optim.Optimizer, device: torch.device | str, epoch_num: int, dataloader: DataLoader[TigerDataset], new_words_dict: dict[int, str], inverse_word_dict: dict[int, str], inverse_sym_dict: dict[int, str], inverse_pos_dict: dict[int, str], inverse_morph_dicts: dict[str, dict[int, str]], tree_gen_rate: float, discodop_config_file: str, eval_dir, gradient_clipping=1, scheduler: torch.optim.lr_scheduler.LRScheduler|None=None, summary_writer: SummaryWriter|None=None, pos_replacements: dict[str, str]={}, training=False):
    poses_one_shot_helper = torch.eye(model.num_terminal_poses, device=device)

    total_sentences = len(dataloader.dataset) #type:ignore

    sum_sentences = 0
    epoch_length = len(dataloader)
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

    input: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]

    optim.zero_grad(set_to_none=True)
    for j, input in (enumerate(dataloader)):
        if training:
            model.train()
        else:
            model.eval()

        sentence_lengths, words, tokens, token_transformations, token_lengths, target_heads, target_syms, target_poses, target_attachment_orders, target_morphs = input
        batch_size = words.shape[0]
        sum_sentences += batch_size

        words = words.to(device=device)
        target_heads = target_heads.to(device=device)
        target_syms = target_syms.to(device=device)
        target_poses = target_poses.to(device=device)
        target_attachment_orders = target_attachment_orders.to(device=device)
        tokens = tokens.to(device=device)
        token_transformations = token_transformations.to(device=device)

        for m, target_morph in target_morphs.items():
            target_morphs[m] = target_morph.to(device=device)

        self_attention, labels, poses, attachment_orders, morphs, indices = model((words, tokens, sentence_lengths, token_transformations), new_words_dict)

        # calculate loss metrics
        target_poses_indexed = target_poses[indices]
        target_syms_indexed = target_syms[indices]
        target_heads_indexed = target_heads[indices]
        target_attachment_orders_indexed = target_attachment_orders[indices]

        poses_logits = poses[indices, target_heads_indexed] # NOTE: we are using teacher-forcing here
        labels_logits = labels[indices, target_heads_indexed]
        attachment_orders_logits = attachment_orders[indices, target_heads_indexed]
        self_attention_logits = self_attention[indices]

        loss_attention = F.cross_entropy(self_attention_logits, target_heads_indexed)
        loss_labels    = F.cross_entropy(labels_logits, target_syms_indexed)
        loss_poses     = F.cross_entropy(poses_logits, target_poses_indexed)
        loss_orders    = F.cross_entropy(attachment_orders_logits, target_attachment_orders_indexed)

        loss_morph = torch.zeros(1, device=device, requires_grad=True)

        for m in target_morphs.keys():
            morph_pred_flattened = morphs[m][indices, target_heads[indices], target_poses[indices]]
            morph_target_flattened = target_morphs[m][indices]

            loss_morph = loss_morph + F.cross_entropy(morph_pred_flattened, morph_target_flattened)


        loss = (loss_attention + loss_labels + loss_poses + loss_orders + loss_morph)

        # calculate f1 metrics
        pred_poses = poses_logits.argmax(dim=-1)
        pred_poses_one_shot = poses_one_shot_helper[pred_poses]
        target_poses_one_shot = poses_one_shot_helper[target_poses_indexed]
        pred_poses_confusion_matrix = torch.einsum("bi,bj->ij", target_poses_one_shot, pred_poses_one_shot) # m[i, j] = number of examples for which the true pos is i and was classed as j

        # perform backpropagation
        if training and not loss.isnan():
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)
            optim.step()
            optim.zero_grad(set_to_none=True)

            # detach and empty cache
            loss_attention.detach_()
            loss_labels.detach_()
            loss_poses.detach_()
            loss_orders.detach_()
            loss_morph.detach_()
            loss.detach_()
        
        if loss.isnan():
            print("WARNING: NaN Loss")

        torch.cuda.empty_cache()
        # save metrics
        with torch.no_grad():
            epoch_attention_loss += loss_attention.item()
            epoch_label_loss += loss_labels.item()
            epoch_pos_loss += loss_poses.item()
            epoch_order_loss += loss_orders.item()
            epoch_morph_loss += loss_morph.item()
            epoch_total_loss += loss.item()

            progress = sum_sentences / total_sentences
            eta_seconds = round((time() - epoch_start_time) * (1 - progress) / progress)
            eta_time = strftime("%H:%M", gmtime(time() + eta_seconds))
            eta_str = timedelta(seconds=eta_seconds)
            speed = round(sum_sentences / (time() - epoch_start_time))

            print(f"EPOCH {epoch_num + 1:5d} {'TRN' if training else 'DEV'} {get_progress_bar(progress, 20)} ({100 * progress:6.2f}%) ATTENTION {loss_attention.item():6.4e} LABEL {loss_labels.item():6.4e} POS {loss_poses.item():6.4e} ORDERS {loss_orders.item():6.4e} MORPH {loss_morph.item():6.4e} TOTAL {loss.item():6.4e} ", end="")

            if scheduler is not None:
                print(f"LR {scheduler.get_last_lr()[0]:4.2e} ", end="")
            
            print(f"ETA {eta_str} @ {eta_time} ({speed:4d} ex s⁻¹)", end="\r", flush=True)

            # now perform f1 evaluation
            if random.random() <= tree_gen_rate: # only generate trees for this batch (100 * rate) % of the time
                best_edges, labels_best_edges, poses_best_edges, attachment_orders_best_edges, morphs_best_edges, (edges, joint_logits) = model._find_tree(sentence_lengths, self_attention, labels, poses, attachment_orders, morphs, indices=indices)

                for s_num in range(batch_size):
                    try:
                        tree_words = words[s_num, :sentence_lengths[s_num]].cpu()

                        tree_heads = best_edges[s_num, :sentence_lengths[s_num]].cpu()
                        tree_syms = labels_best_edges[s_num, :sentence_lengths[s_num]].cpu()
                        tree_poses = poses_best_edges[s_num, :sentence_lengths[s_num]].cpu()
                        tree_attachment_orders = attachment_orders_best_edges[s_num, :sentence_lengths[s_num]].cpu()
                        tree_morphs = {key: value[s_num, :sentence_lengths[s_num]].cpu() for key, value in morphs_best_edges.items()}

                        t_tree_heads = target_heads[s_num, :sentence_lengths[s_num]].cpu()
                        t_tree_syms = target_syms[s_num, :sentence_lengths[s_num]].cpu()
                        t_tree_poses = target_poses[s_num, :sentence_lengths[s_num]].cpu()
                        t_tree_attachment_orders = target_attachment_orders[s_num, :sentence_lengths[s_num]].cpu()
                        t_tree_morphs = {key: value[s_num, :sentence_lengths[s_num]].cpu() for key, value in target_morphs.items()}

                        the_sentence = [inverse_word_dict[int(w.item())] if w > 0 else new_words_dict[-int(w.item())] for w in tree_words]

                        # TODO: keep morphs as dictionary instead of tuple
                        c_tree = ConstituentTree.from_collection(heads=tree_heads, syms=[inverse_sym_dict[int(l.item())] for l in tree_syms], poses=[inverse_pos_dict[int(p.item())] for p in tree_poses], orders=tree_attachment_orders, words=the_sentence, morphs={
                            prop: [inverse_morph_dicts[prop][int(v.item())] for v in value]
                            for prop, value in tree_morphs.items()
                        })
                        t_c_tree = ConstituentTree.from_collection(heads=t_tree_heads, syms=[inverse_sym_dict[int(l.item())] for l in t_tree_syms], poses=[inverse_pos_dict[int(p.item())] for p in t_tree_poses], orders=t_tree_attachment_orders, words=the_sentence, morphs={
                            prop: [inverse_morph_dicts[prop][int(v.item())] for v in value]
                            for prop, value in t_tree_morphs.items()
                        })

                        brackets.append(c_tree.get_bracket(zero_indexed=True, ignore_words=True, pos_replacements=pos_replacements))
                        brackets_structure.append(c_tree.get_bracket(ignore_all_syms=True, zero_indexed=True, ignore_words=True))

                        t_brackets.append(t_c_tree.get_bracket(zero_indexed=True, ignore_words=True, pos_replacements=pos_replacements))
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

    if summary_writer is not None:
        summary_writer.add_scalars(f"epoch_{'trn' if training else 'dev'}_losses", {
            "attention": epoch_attention_loss,
            "label": epoch_label_loss,
            "pos": epoch_pos_loss,
            "order": epoch_order_loss,
            "morph": epoch_morph_loss,
            "total": epoch_total_loss
        }, epoch_num + 1)

    brackets_filename = f"{eval_dir}/{get_filename(epoch_num)}.txt"
    brackets_gold_filename = f"{eval_dir}/{get_filename(epoch_num)}_gold.txt"
    brackets_structure_filename = f"{eval_dir}/{get_filename(epoch_num)}_structure.txt"
    brackets_structure_gold_filename = f"{eval_dir}/{get_filename(epoch_num)}_gold_structure.txt"

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
                                        discodop_config_file
                                        ],
                                    capture_output=True,
                                    text=True).stdout
        brackets_structure_res = subprocess.run(["discodop",
                                                    "eval",
                                                    brackets_structure_gold_filename,
                                                    brackets_structure_filename,
                                                    "--fmt=discbracket",
                                                    discodop_config_file
                                                    ],
                                            capture_output=True,
                                            text=True).stdout
        disc_brackets_res = subprocess.run(["discodop",
                                        "eval",
                                        brackets_gold_filename,
                                        brackets_filename,
                                        "--fmt=discbracket",
                                        "--disconly",
                                        discodop_config_file
                                        ],
                                    capture_output=True,
                                    text=True).stdout
        disc_brackets_structure_res = subprocess.run(["discodop",
                                                    "eval",
                                                    brackets_structure_gold_filename,
                                                    brackets_structure_filename,
                                                    "--fmt=discbracket",
                                                    "--disconly",
                                                    discodop_config_file
                                                    ],
                                            capture_output=True,
                                            text=True).stdout

        if summary_writer is not None:
            summary_writer.add_scalars(f"epoch_{'trn' if training else 'dev'}_f1", {
                label: float(re.search(r".*labeled f-measure.*?([\d.]|nan)+.*?([\d.]+|nan)", res).groups()[1])
                for (label, res) in [
                    ("full", brackets_res),
                    ("structure", brackets_structure_res),
                    ("full_disc", disc_brackets_res),
                    ("structure_disc", disc_brackets_structure_res)
                ]}, epoch_num + 1)