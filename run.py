import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
from string import punctuation
import torch.nn.utils.clip_grad as utils

train_dataloader, train_new_words, character_set, character_flag_generators, inverse_word_dict = pickle.load(open("required_vars.pkl", "rb"))


model = TigerModel(TigerModel.WordEmbeddingParams(char_set=character_set, char_flag_generators=character_flag_generators, char_internal_embedding_dim=100, char_part_embedding_dim=100, word_part_embedding_dim=100, char_internal_window_size=3, word_dict=inverse_word_dict), TigerModel.LSTMParams(hidden_size=512, bidirectional=True, num_layers=3), TigerModel.LSTMParams(hidden_size=512, bidirectional=False, num_layers=1))
model.cuda()

print(f"Model as {sum([p.numel() for p in model.parameters()])} parameters")

optim = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.9)) # Dozat and Manning (2017) suggest that beta2 of 0.999 means model does not sufficiently adapt to new changes in moving average of gradient norm

i = 0

train_total_sentences = len(train_dataloader.dataset)
sum_sentences = 0

while True:
    i += 1
    model.train()
    optim.zero_grad()
    input = next(iter(train_dataloader))

    words, sentence_lengths, target_heads = input
    batch_size = words.shape[0]
    sum_sentences += batch_size

    words = words.cuda()
    target_heads = target_heads.cuda()

    self_attention, labels, indices = model((words, sentence_lengths), train_new_words)
    loss = F.cross_entropy(self_attention[indices], target_heads[indices])

    print(f"{sum_sentences}/{train_total_sentences} ({100 * sum_sentences / train_total_sentences: .2f}%) iteration {i} loss {loss.item():.6f}")

    loss.backward()

    optim.step()
    