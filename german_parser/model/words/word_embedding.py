from typing import Callable, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from .word_cnn import WordCNN

from ...util.logger import model_logger

class WordEmbedding(nn.Module):
    def __init__(self, char_set: dict[str, int], char_flag_generators: list[Callable[[str], Literal[1, 0]]], char_internal_embedding_dim: int, char_part_embedding_dim: int,  word_part_embedding_dim: int, char_internal_window_size: int, word_dict: dict[int, str], unk_rate: float):
        """initialize a WordEmbedding module. Produces embeddings from word codes, which are concatenations of word- and character-level embeddings

        Args:
            char_set (dict[str, int]): set of characters to be recognised, excluding <UNK> and <PAD>. Codes must start from 2 (0 and 1 are reserved for <UNK> and <PAD> respectively). When calling .forward, characters not in char_set will be treated as <UNK>
            char_flag_generators (list[Callable[[str], Literal[1, 0]]): set of flag generators to pass to WordCNN
            char_internal_embedding_dim (int): size of internal character embeddings in WordCNN
            char_part_embedding_dim (int): size of character-level embedding representation in output
            word_part_embedding_dim (int): size of word-level embedding representation in output
            char_internal_window_size (int): window size when performing convolution in WordCNN
            word_dict (dict[int, str]): dictionary of recognised words. The keys should be a range of integers from 2 to the length of the dictionary + 1, inclusive. The values should be the corresponding words. The value corresponding to key 0 is treated as <UNK> and key 1 is <PAD> (optional). When calling .forward, words not in word_dict will be treated as <UNK>, but can be added to the dictionary by passing a new_words_dict to .forward, to help generate a character-level embedding nonetheless.
        """

        super().__init__()

        word_dict = word_dict.copy()

        self.word_cnn = WordCNN(char_set, char_flag_generators, char_internal_embedding_dim, char_part_embedding_dim, char_internal_window_size, word_dict)

        if 0 in word_dict:
            word_dict.pop(0)
            model_logger.warning("Treating word code 0 as <UNK>.")

        if 1 in word_dict:
            word_dict.pop(1)
            model_logger.warning("Treating word code 1 as <PAD>.")

        self.num_words = len(word_dict) # number of known words excluding <UNK> and <PAD>

        assert set(word_dict.keys()) == set(range(2, self.num_words + 2)), "(excluding 0 for <UNK> and 1 for <PAD>), word dictionary keys must be a range of integers from 1 to the length of the dictionary + 1, inclusive"

        word_dict[0] = "<UNK>"
        word_dict[1] = "<PAD>"

        self.word_dict = word_dict
        self.word_part_embedding_dim = word_part_embedding_dim

        self.embeddings = nn.Embedding(self.num_words + 2, self.word_part_embedding_dim, padding_idx=1)
        with torch.no_grad():
            self.embeddings.weight[1] = 1 # padding embedding is ones within word-level embedding

        self.unk_rate = unk_rate

        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False) # dummy parameter to find device

    def forward(self, x: torch.Tensor, new_words_dict: dict[int, str] | None = None):
        """take a tensor of word codes and return a tensor of word embeddings
           NOTE: during training, we assume new_words_dict is empty, and x contains no negative indices. instead, <UNK> will be randomly set 

        Args:
            x (torch.Tensor): tensor of shape S containing word codes. These word codes correspond to those in the word_dict passed to the constructor. If new words are used, these should be indicated by negative indices. The absolute value of these indices should correspond to the key in new_words_dict. If no new words are used, this can be left unspecified.
            new_words_dict (dict[int, str] | None, optional): If new words (indicated by negative indices) are used, provide a dictionary with keys equal to the POSITIVE component of these negative indices and values equal to the corresponding string. This is fed into WordCNN to generate character-level embeddings. If no new words are present or you would not like to (indicated by all indices being >= 0), this can be left unspecified. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (*S, self.char_part_embedding_dim + self.word_part_embedding_dim)
        """

        x = torch.as_tensor(x, dtype=torch.long)

        char_part: torch.Tensor = self.word_cnn(x, new_words_dict) # dimensions (*S, self.char_part_embedding_dim)

        if self.training:
            x = torch.empty_like(x).bernoulli_(1 - self.unk_rate) * x
        else:
            x = x.maximum(torch.zeros(1, device=self.dummy_param.device, dtype=torch.long)) # replace negative indices with 0 (corresponding to <UNK>) when calculating word-level embeddings
        word_part: torch.Tensor = self.embeddings(x) # dimensions (*S, self.word_part_embedding_dim)

        res = torch.cat((char_part, word_part), dim=-1) # dimensions (*S, self.char_part_embedding_dim + self.word_part_embedding_dim)
        
        return res