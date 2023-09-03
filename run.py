import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
from string import punctuation
import torch.nn.utils.clip_grad as utils
from german_parser.util import get_progress_bar

from math import ceil, floor

from torch.utils.tensorboard import SummaryWriter

(train_dataloader, train_new_words), (dev_dataloader, dev_new_words), _, character_set, character_flag_generators, inverse_word_dict, inverse_sym_dict = pickle.load(open("required_vars.pkl", "rb"))

from time import time, strftime, gmtime
from datetime import timedelta

summary_writer = SummaryWriter()

model = TigerModel(
    word_embedding_params=TigerModel.WordEmbeddingParams(char_set=character_set, char_flag_generators=character_flag_generators, char_internal_embedding_dim=100,
                                   char_part_embedding_dim=100, 
                                   word_part_embedding_dim=100, 
                                   char_internal_window_size=3,
                                   word_dict=inverse_word_dict),
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
        num_biaffine_attention_classes=2,
        num_constituent_labels=len(inverse_sym_dict),
        enc_attachment_mlp_dim=128,
        dec_attachment_mlp_dim=64,
        max_attachment_order=train_dataloader.dataset.attachment_orders.max() + 1
    )
model.cuda()

print(f"Model as {sum([p.numel() for p in model.parameters()])} parameters")

optim = torch.optim.SGD(model.parameters(), lr=1e-1) #, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm

num_epochs = 10

train_total_sentences = len(train_dataloader.dataset)

total_iteration_train = 0
total_iteration_dev = 0

for i in range(num_epochs):
    for training in (True, False):
        sum_sentences = 0
        epoch_length = len(train_dataloader if training else dev_dataloader)
        epoch_start_time = time()

        epoch_attention_loss = 0
        epoch_label_loss = 0
        epoch_order_loss = 0
        epoch_total_loss = 0

        for j, input in (enumerate(train_dataloader) if training else enumerate(dev_dataloader)):
            if training:
                model.train()
                optim.zero_grad()
            else:
                model.eval()

            words, sentence_lengths, target_heads, target_syms, target_attachment_orders = input
            batch_size = words.shape[0]
            sum_sentences += batch_size

            words = words.cuda()
            target_heads = target_heads.cuda()
            target_syms = target_syms.cuda()
            target_attachment_orders = target_attachment_orders.cuda()

            self_attention, labels, attachment_orders, indices = model((words, sentence_lengths), train_new_words if training else dev_new_words)

            loss_attention = F.cross_entropy(self_attention[indices], target_heads[indices])
            loss_labels    = F.cross_entropy(labels[indices, target_heads[indices]], target_syms[indices])
            loss_orders    = F.cross_entropy(attachment_orders[indices, target_heads[indices]], target_attachment_orders[indices])

            loss = (loss_attention + loss_labels + loss_orders)

            
            epoch_attention_loss += loss_attention.item()
            epoch_label_loss += loss_labels.item()
            epoch_order_loss += loss_orders.item()
            epoch_total_loss += loss.item()


            progress = sum_sentences / train_total_sentences
            eta_seconds = round((time() - epoch_start_time) * (1 - progress) / progress)
            eta_time = strftime("%H:%M", gmtime(time() + eta_seconds))
            eta_str = timedelta(seconds=eta_seconds)
            speed = round(sum_sentences / (time() - epoch_start_time))

            print(f"EPOCH {i + 1} {'TRN' if training else 'DEV'} {get_progress_bar(progress, 20)} ({100 * progress: .2f}%) ATTENTION {loss_attention.item():.6f} LABEL {loss_labels.item():.6f} ORDERS {loss_orders.item():.6f} TOTAL {loss.item():.6f} ETA {eta_str} @ {eta_time} ({speed} ex s⁻¹)")


            for name, iteration in [(f"epoch_{i + 1}_{'trn' if training else 'dev'}", j), (f"all_epochs_{'trn' if training else 'dev'}", total_iteration_train if training else total_iteration_dev)]:
                summary_writer.add_scalars(name, {
                    "loss_total": loss,
                    "loss_attention": loss_attention,
                    "loss_orders": loss_orders,
                    "loss_labels": loss_labels
                }, iteration)


            if training:
                loss.backward()
                utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optim.step()

                total_iteration_train += 1
            else:
                total_iteration_dev += 1

            torch.cuda.empty_cache()


        epoch_attention_loss /= epoch_length
        epoch_label_loss     /= epoch_length
        epoch_order_loss     /= epoch_length
        epoch_total_loss     /= epoch_length

        summary_writer.add_scalars(f"epoch_{'trn' if training else 'dev'}_losses", {
            "attention": epoch_attention_loss,
            "label": epoch_label_loss,
            "order": epoch_order_loss,
            "total": epoch_total_loss
        }, i + 1)