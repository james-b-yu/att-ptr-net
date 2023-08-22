import torch
import torch.nn as nn
from .word_cnn import WordCNN

class WordEmbedding(nn.Module):
    def __init__(self, char_set: list[str] | set[str], char_internal_embedding_dim: int, char_part_embedding_dim: int,  word_part_embedding_dim: int, char_internal_window_size: int, word_dict: dict[int, str]):
        """initialize a WordEmbedding module. Produces embeddings from word codes, which are concatenations of word- and character-level embeddings

        Args:
            char_set (list[str] | set[str]): set of recognised characters to be fed into WordCNN, excluding <UNK> and <PAD>. When calling .forward, characters not in char_set will be treated as <UNK>
            char_internal_embedding_dim (int): size of internal character embeddings in WordCNN
            char_part_embedding_dim (int): size of character-level embedding representation in output
            word_part_embedding_dim (int): size of word-level embedding representation in output
            char_internal_window_size (int): window size when performing convolution in WordCNN
            word_dict (dict[int, str]): dictionary of recognised words. The keys should be a range of integers from 1 to the length of the dictionary. The values should be the corresponding words. The value corresponding to key 0 is treated as <UNK> (optional). When calling .forward, words not in word_dict will be treated as <UNK>, but can be added to the dictionary by passing a new_words_dict to .forward, to help generate a character-level embedding nonetheless.
        """

        super().__init__()

        word_dict = word_dict.copy()

        self.word_cnn = WordCNN(char_set, char_internal_embedding_dim, char_part_embedding_dim, char_internal_window_size, word_dict)

        if 0 in word_dict:
            word_dict.pop(0)
            print("Warning: treating word code 0 as <UNK>")

        self.num_words = len(word_dict) # number of known words excluding <UNK>

        assert set(word_dict.keys()) == set(range(1, self.num_words + 1)), "Word dictionary (excluding 0 for <UNK>) keys must be a range of integers from 1 to the length of the dictionary"

        word_dict[0] = "<UNK>"

        self.word_dict = word_dict
        self.word_part_embedding_dim = word_part_embedding_dim

        self.embeddings = nn.Embedding(self.num_words + 1, self.word_part_embedding_dim)

    def forward(self, x: torch.Tensor, new_words_dict: dict[int, str] | None = None):
        """take a tensor of word codes and return a tensor of word embeddings

        Args:
            x (torch.Tensor): tensor of shape S containing word codes. These word codes correspond to those in the word_dict passed to the constructor. If new words are used, these should be indicated by negative indices. The absolute value of these indices should correspond to the key in new_words_dict. If no new words are used, this can be left unspecified.
            new_words_dict (dict[int, str] | None, optional): If new words (indicated by negative indices) are used, provide a dictionary with keys equal to the POSITIVE component of these negative indices and values equal to the corresponding string. This is fed into WordCNN to generate character-level embeddings. If no new words are present or you would not like to (indicated by all indices being >= 0), this can be left unspecified. Defaults to None.

        Returns:
            torch.Tensor: output tensor of shape (*S, self.char_part_embedding_dim + self.word_part_embedding_dim)
        """

        unknown_words = x <= 0

        char_part: torch.Tensor = self.word_cnn(x, new_words_dict) # dimensions (*S, self.char_part_embedding_dim)

        x[unknown_words] = 0 # set unknown words to <UNK> code when generating word-part embeddings
        word_part: torch.Tensor = self.embeddings(x) # dimensions (*S, self.word_part_embedding_dim)

        return torch.cat((char_part, word_part), dim=-1) # dimensions (*S, self.char_part_embedding_dim + self.word_part_embedding_dim)