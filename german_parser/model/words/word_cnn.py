import torch
import torch.nn as nn
from collections.abc import Callable
from typing import Literal

from ...util.logger import model_logger

class WordCNN(nn.Module):
    def __init__(self, char_set: dict[str, int], flag_generators: list[Callable[[str], Literal[1, 0]]], char_embedding_dim: int, out_embedding_dim: int, window_size: int, word_dict: dict[int, str]):
        """_summary_

        Args:
            char_set (dict[str, int]): set of characters to be recognised, excluding <UNK> and <PAD>. Codes must start from 2 (0 and 1 are reserved for <UNK> and <PAD> respectively)
            flag_generators (list[Callable[[str], Literal[1, 0]]): set of callable functions that determine whether a character contains a certain property (e.g. is_capital)
            char_embedding_dim (int): dimension of internal embedding dimension when looking at chars
            out_embedding_dim (int): dimension of final embedding per word
            window_size (int): _description_
            word_dict (dict[int, str]): excluding <UNK> word. <UNK>
        """
        super().__init__()

        char_set = char_set.copy()
        word_dict = word_dict.copy()

        # save the character set
        for j in char_set:
            assert len(j) == 1, "Character set must be a list of single characters"
        if (0 in char_set.values()) or (1 in char_set.values()):
            model_logger.warning("Overriding character code 0 as <UNK> and character code 1 as <PAD>.")

        char_set["<UNK>"] = 0
        char_set["<PAD>"] = 1
        self.character_set: dict[str, int] = char_set

        # define embeddings
        self.num_embeddings = len(self.character_set) # number of characters to deal with, including <PAD> and <UNK>
        self.embedding_dim = char_embedding_dim
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim, padding_idx=1)
        with torch.no_grad():
            self.embeddings.weight[1] = 0 # padding embedding is zeros 

        # define flags
        self.num_flags = 1 + len(flag_generators) # note: only unknown flag will be set for unknown characters (even if e.g. it is capital)
        self.flags = nn.Parameter(torch.zeros(self.num_embeddings, self.num_flags, dtype=torch.long), requires_grad=False) # self.flags[n] is an array of size self.num_flags
        self.flags[0,0] = 1 # the first flag always corresponds to unknown
        for c, b in self.character_set.items():
            if (b == 0) or (b == 1):
                continue # skip <PAD> and <UNK>
        
            for j, f in enumerate(flag_generators):
                self.flags[b, j + 1] = f(c) # + 1 because 0th index corresponds to flag for <UNK>


        # define conv
        self.num_out_channels = out_embedding_dim
        self.window_size = window_size

        self.conv = nn.Conv1d(in_channels=self.embedding_dim + self.num_flags, out_channels=self.num_out_channels, padding_mode="zeros", kernel_size=self.window_size)

        # prepare tensor of character codes
        if 0 in word_dict:
            word_dict.pop(0)
            model_logger.warning("Treating word code 0 as <UNK>")
        self.num_words = len(word_dict) # number of known words excluding <UNK>

        assert set(word_dict.keys()) == set(range(1, self.num_words + 1)), "Word dictionary (excluding 0 for <UNK>) keys must be a range of integers from 1 to the length of the dictionary"
        self.max_word_length = max([len(v) for v in word_dict.values()])

        self.word_dict = nn.Parameter(torch.ones(self.num_words + 1, self.max_word_length, dtype=torch.long), requires_grad=False) # word code -> list of character codes; index 0 corresponds to <UNK> but will be unused; fill with ones to indicate padding
        self.word_lengths = nn.Parameter(torch.zeros(self.num_words + 1, dtype=torch.long), requires_grad=False)

        for i, word in word_dict.items(): # iterate word_dict keys (excluding 0)
            self.word_lengths[i] = len(word)


            for j, c in enumerate(word):
                self.word_dict[i, j] = self.character_set.get(c, 0)

        self.word_lengths_masks = nn.Parameter(torch.full((self.max_word_length + 1, self.max_word_length), True).triu()[self.word_lengths], requires_grad=False)


    def forward(self, x: torch.Tensor, new_words_dict: dict[int, str] | None = None) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): an input tensor of shape S, containing keys corresponding to words in self.word_dict. Unknown words are marked with a key of NEGATIVE values where the negative values correspond to (positive) values in a new_words_dict. If you would not like to provide this dict, you can code unknown words using 0. Note: words longer than self.max_word_length will be truncated from the right

        Returns:
            torch.Tensor: output tensor of shape (*S, self.out_embedding_size)
        """
        x = torch.as_tensor(x)

        if x.dtype != torch.long:
            model_logger.warning("Warning: converting into long tensor")
            x = x.to(dtype=torch.long)

        unknown_words = x < 0
        known_words = x > 0
        x_words = torch.ones(*x.shape, self.max_word_length, dtype=torch.long, device=x.device)
        x_words[known_words] = self.word_dict[x[known_words]] # dimensions (*S, self.max_word_length)

        unknown_words_locs = unknown_words.nonzero(as_tuple=False)

        for loc in unknown_words_locs:
            if loc.numel() == 0:
                continue

            assert type(new_words_dict) == dict, "If we have new words, must provide an accompanying dict for the new words"
            new_word_idx = -x[tuple(loc)]

            assert new_word_idx.item() in new_words_dict
            new_word = new_words_dict[int(new_word_idx.item())]

            for j, c in enumerate(new_word):
                if j < self.max_word_length:
                    x_words[*tuple(loc), j] = self.character_set.get(c, 0)

        x_characters_embeddings: torch.Tensor = self.embeddings(x_words) # dimensions (*S, self.max_word_length, self.embedding_dim)
        x_characters_flags: torch.Tensor = self.flags[x_words] # dimensions (*S, self.max_word_length, self.num_flags)

        x_characters = torch.cat((x_characters_embeddings, x_characters_flags), dim=-1) # dimensions (*S, self.max_word_length, self.embedding_dim + self.num_flags)

        x_characters = x_characters.transpose(-1, -2) # dimensions (*S, self.embedding_dim + self.num_flags, self.max_word_length)

        conv_out: torch.Tensor = self.conv(x_characters.flatten(start_dim=0, end_dim=-3)) # dimensions (N, self.out_embedding_size, self.max_word_length)

        # TODO: see whether applying mask will be useful (so that long words do not compromise "accuracy" of taking max for shorter words). probably won't make a difference
       
        max_out: torch.Tensor = conv_out.max(dim=-1).values # dimensions (N, self.out_embedding_size)

        return max_out.reshape(*x.shape, -1) # dimensions (*S, self.out_embedding_size)