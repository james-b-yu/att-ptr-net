import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
import torch
import torch.nn.functional as F

import dill as pickle

from german_parser.model import TigerModel
from string import punctuation

train_dataloader, train_new_words, character_set, character_flag_generators, inverse_word_dict = pickle.load(open("required_vars.pkl", "rb"))


model = TigerModel(TigerModel.WordEmbeddingParams(char_set=character_set, char_flag_generators=character_flag_generators, char_internal_embedding_dim=10, char_part_embedding_dim=100, word_part_embedding_dim=100, char_internal_window_size=3, word_dict=inverse_word_dict), TigerModel.LSTMParams(hidden_size=10, bidirectional=True), TigerModel.LSTMParams(hidden_size=10, bidirectional=False))
model.cuda()

print(f"Model as {sum([p.numel() for p in model.parameters()])} parameters")

optim = torch.optim.Adam(model.parameters(), lr=1e-4)

i = 0

while True:
    i += 1
    model.train()
    optim.zero_grad()
    input = next(iter(train_dataloader))

    words, sentence_lengths, target_heads = input
    
    words = words.cuda()
    target_heads = target_heads.cuda()

    self_attention, labels, indices = model((words, sentence_lengths), train_new_words)
    loss = F.cross_entropy(self_attention[indices], target_heads[indices])

    print(f"iteration {i} loss {loss.item():.6f}")

    loss.backward()

    optim.step()
    