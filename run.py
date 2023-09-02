import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
from string import punctuation
import torch.nn.utils.clip_grad as utils

from math import ceil, floor

train_dataloader, train_new_words, character_set, character_flag_generators, inverse_word_dict, inverse_sym_dict = pickle.load(open("required_vars.pkl", "rb"))

from time import time, strftime, gmtime
from datetime import timedelta

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
        num_constituent_labels=len(inverse_sym_dict)
    )
model.cuda()

print(f"Model as {sum([p.numel() for p in model.parameters()])} parameters")

optim = torch.optim.SGD(model.parameters(), lr=1e-1) #, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm

num_epochs = 10

train_total_sentences = len(train_dataloader.dataset)

for i in range(num_epochs):
    model.train()
    optim.zero_grad()

    sum_sentences = 0
    epoch_length = len(train_dataloader)
    epoch_start_time = time()

    for j, input in enumerate(train_dataloader):
        words, sentence_lengths, target_heads, target_syms, target_attachment_orders = input
        batch_size = words.shape[0]
        sum_sentences += batch_size

        words = words.cuda()
        target_heads = target_heads.cuda()
        target_syms = target_syms.cuda()

        self_attention, labels, indices = model((words, sentence_lengths), train_new_words)

        loss_attention = F.cross_entropy(self_attention[indices], target_heads[indices])
        loss_labels    = F.cross_entropy(labels[indices, target_heads[indices]], target_syms[indices])

        loss = (loss_attention + loss_labels)

        progress = sum_sentences / train_total_sentences
        eta_seconds = round((time() - epoch_start_time) * (1 - progress) / progress)
        eta_time = strftime("%H:%M", gmtime(time() + eta_seconds))
        eta_str = timedelta(seconds=eta_seconds)
        speed = round(sum_sentences / (time() - epoch_start_time))

        print(f"EPOCH {i + 1} {'█'*ceil(10 * progress) + '░'* floor(10 - 10 * progress)} ({100 * progress: .2f}%) ATTENTION {loss_attention.item():.6f} LABEL {loss_labels.item():.6f} TOTAL {loss.item():.6f} ETA {eta_str} @ {eta_time} ({speed} ex s⁻¹)")

        loss.backward()

        utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optim.step()
        torch.cuda.empty_cache()
