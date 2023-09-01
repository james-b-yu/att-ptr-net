
import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils
import torch.nn.functional as F

from pydantic import BaseModel, Field
from collections.abc import Callable
from typing import Literal

from ..nn import LSTM, LSTMSkip
from ..nn import BiAffine

from .words import WordEmbedding

class TigerModel(nn.Module):
    class WordEmbeddingParams(BaseModel):
        char_set: dict[str, int]
        char_flag_generators: list[Callable[[str], Literal[1, 0]]]
        char_internal_embedding_dim: int
        char_part_embedding_dim: int
        word_part_embedding_dim: int
        char_internal_window_size: int
        word_dict: dict[int, str]

    class LSTMParams(BaseModel):
        hidden_size: int
        bidirectional: bool = Field(default=False)
        num_layers: int = Field(default=1)
        dropout: float = Field(default=0.2)

    def __init__(self, word_embedding_params: WordEmbeddingParams, enc_lstm_params: LSTMParams, dec_lstm_params: LSTMParams, enc_attention_mlp_dim: int, dec_attention_mlp_dim: int, enc_label_mlp_dim: int, dec_label_mlp_dim: int, num_biaffine_attention_classes=2, num_constituent_labels=10):
        super().__init__()
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False) # to get self device
        
        # create word embeddor
        self.word_embedding_params = word_embedding_params
        self.word_embedding = WordEmbedding(
            char_set=self.word_embedding_params.char_set,
            char_flag_generators=self.word_embedding_params.char_flag_generators,
            char_internal_embedding_dim=self.word_embedding_params.char_internal_embedding_dim,
            char_part_embedding_dim=self.word_embedding_params.char_part_embedding_dim,
            word_part_embedding_dim=self.word_embedding_params.word_part_embedding_dim,
            char_internal_window_size=self.word_embedding_params.char_internal_window_size,
            word_dict=self.word_embedding_params.word_dict
        )

        # define encoder
        self.enc_lstm_params = enc_lstm_params
        assert enc_lstm_params.bidirectional == True, "Encoder must be bidirectional"
        self.enc_lstm = LSTMSkip(
            input_size=word_embedding_params.char_part_embedding_dim + word_embedding_params.word_part_embedding_dim,
            hidden_size=self.enc_lstm_params.hidden_size,
            num_layers=self.enc_lstm_params.num_layers,
            bidirectional=self.enc_lstm_params.bidirectional,
            dropout_rate=self.enc_lstm_params.dropout
        )

        # define decoder
        self.dec_lstm_params = dec_lstm_params
        assert self.dec_lstm_params.bidirectional == False, "Decoder must not be bidirectional"
        self.dec_lstm = LSTMSkip(
            input_size=2 * self.enc_lstm_params.hidden_size,
            hidden_size=self.dec_lstm_params.hidden_size,
            num_layers=self.dec_lstm_params.num_layers,
            bidirectional=self.dec_lstm_params.bidirectional,
            dropout_rate=self.dec_lstm_params.dropout
        )

        # define initial encoder state for first layer
        self.enc_init_state = nn.Parameter(
            torch.zeros(2, 1, self.enc_lstm_params.hidden_size),
            requires_grad=True
        )

        # define dense layer to convert encoder final cell state into decoder initial cell state
        self.enc_final_cell_to_dec_init_cell = nn.Linear(
            2 * self.enc_lstm_params.hidden_size,
            self.dec_lstm_params.hidden_size
        )

        # define dense layers to convert between encoder/decoder output and input to biaffine layers
        self.enc_attention_mlp_dim = enc_attention_mlp_dim
        self.dec_attention_mlp_dim = dec_attention_mlp_dim

        self.enc_label_mlp_dim = enc_label_mlp_dim
        self.dec_label_mlp_dim = dec_label_mlp_dim

        self.enc_attention_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_attention_mlp_dim)
        self.dec_attention_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_attention_mlp_dim)

        self.enc_label_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_label_mlp_dim)
        self.dec_label_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_label_mlp_dim)

        # define biaffine layer for attention
        self.biaffine_attention = BiAffine(
            num_classes=num_biaffine_attention_classes,
            enc_input_size=self.enc_attention_mlp_dim,
            dec_input_size=self.dec_attention_mlp_dim,
            include_attention=True
        )

        # define biaffine layer for classification of constituent labels
        self.biaffine_constituent_classifier = BiAffine(
            num_classes=num_constituent_labels,
            enc_input_size=self.enc_label_mlp_dim,
            dec_input_size=self.dec_label_mlp_dim,
            include_attention=False
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.enc_init_state)

    def _get_final_concatenated_enc_hidden_state(self, c: torch.Tensor):
        """takes final two layers of final cell state of encoder and returns a tensor of size (B, 1, 2 * enc_hidden_size) to initialise the decoder (or encoder)

        Args:
            c (torch.Tensor): final cell state or hidden state, of size (enc_num_layers * 2, B, enc_hidden_size)

        Returns:
            torch.Tensor: last two layers of the state concatenated together. size (B, 1, 2 * enc_hidden_size)
        """
        _, B, _ = c.shape
        res = c[-2:] # take the last two layers (2, B, enc_hidden_size)
        res = res.transpose(0, 1).contiguous() # (B, 2, enc_hidden_size)
        res = res.view(B, 1, 2 * self.enc_lstm_params.hidden_size) # (B, 1, 2 * enc_hidden_size)
        return res

    def _get_decoder_init_state(self, encoder_final_hc: tuple[torch.Tensor, torch.Tensor]):
        """convert final encoder hidden state into a value to initialise hidden state for the decoder

        Args:
            encoder_final_hc (tuple[torch.Tensor, torch.Tensor]): tuple of tensors, each with size (enc_num_layers * 2, B, enc_hidden_size)

        returns tuple[torch.Tensor, torch.Tensor]: tuple of initial decoder state tensors, each with size (dec_num_layers, B, dec_hidden_size). First is initial hidden state, second is inital cell state for the decoder
        """
        h, c = encoder_final_hc
        _, B, _ = h.shape

        c = self._get_final_concatenated_enc_hidden_state(c)
        c = c.transpose(0, 1) # (1, B, 2 * enc_hidden_size)
        
        c_dec: torch.Tensor = self.enc_final_cell_to_dec_init_cell(c) # (1, B, dec_hidden_size)
        
        h_dec = c_dec.tanh()

        return (h_dec, c_dec)
        

    def forward(self, input: tuple[torch.Tensor, torch.Tensor], new_words_dict: dict[int, str] | None):
        """forward

        Args:
            input (tuple[torch.Tensor, torch.Tensor]): tuple of (data, sentence_lengths), where data is a tensor of size (B, T) and sentence_lengths is a tensor of size (B,). B is batch size, T is max(sentence_length) across all batches. The input must be sorted in descending order of sentence length
            new_words_dict (dict[int, str] | None): dictionary of new words. positive indices in new_words_dict correspond to negative indices in input[0] (data). If None, then all unknown words must be coded as 0

        Returns:
            _type_: _description_
        """

        # transfer to current device. avoid making a copy if possible
        x, lengths = input     
        x = torch.as_tensor(x, device=self.dummy_param.device)

        B = len(lengths)

        # create packed embedding sequences
        x_embedded = self.word_embedding(x, new_words_dict) # (B, T, E) where B is batch_size, T is max(sentence_length), E is embedding dimension (char_part_embedding_dim + word_part_embedding_dim)
        x_embedded_packed = rnn_utils.pack_padded_sequence(x_embedded, lengths, batch_first=True, enforce_sorted=True)

        # henceforth, T refers to max(sentence_length) within the batch, rather than across all batches

        # define initial encoder state
        c_init = self.enc_init_state.repeat(1, B, 1) # (enc_num_layers * 2, B, enc_hidden_size)
        h_init = c_init.tanh()

        # feed through encoder
        enc_out, enc_final_state = self.enc_lstm(x_embedded_packed, (h_init, c_init)) # enc_out has size (B, T, 2 * enc_hidden_size)  
        enc_out_pad, _ = rnn_utils.pad_packed_sequence(enc_out, batch_first=True)

        # feed through decoder
        dec_init_state = self._get_decoder_init_state(enc_final_state) # tuple of tensors, each with size (dec_num_layers, B, dec_hidden_size)
        dec_out, _ = self.dec_lstm(enc_out, (dec_init_state[0], dec_init_state[1])) # (B, T, hidden_size)

        # TODO: apply dropout to enc_out

        # unpad encoder output (B, T + 1, 2 * enc_hidden_size)
        # concatenate final layer of initial encoder state with the output of the encoder
        h_init_res = self._get_final_concatenated_enc_hidden_state(h_init) # (B, 1, 2 * enc_hidden_size)
        enc_out_pad = torch.cat((h_init_res, enc_out_pad), dim=1) # (B, T + 1, 2 * enc_hidden_size)

        # unpad decoder output (B, T, hidden_size)
        dec_out_pad, _ = rnn_utils.pad_packed_sequence(dec_out, batch_first=True)

        # henceforth, indices are 0-indexed in the comments. Effectively, head indices are 1-indexed (0 indicates root), and dependency indices are 0-indexed
        # TASK 1: predict HEAD words
        # for batch b and word index j, argmax(self_attention[b, j]) gives a pointer i to HEAD of word j

        enc_out_attention = F.elu(self.enc_attention_mlp(enc_out_pad))
        dec_out_attention = F.elu(self.dec_attention_mlp(dec_out_pad))
        self_attention = self.biaffine_attention(enc_out_attention, dec_out_attention) # size (B, T, T + 1). index by (batch_num, decoder_index, encoder_index + 1), which represents (batch_num, dependency_index, head_index + 1)

        # TASK 2: predict ATTACHMENT labels
        # for batch b and dependency index j and head index i, constituent_lables[b, j, i] gives logits to classify the label of the dependency from word j to HEAD word i
        enc_out_label = F.elu(self.enc_label_mlp(enc_out_pad))
        dec_out_label = F.elu(self.dec_label_mlp(dec_out_pad))
        constituent_labels = self.biaffine_constituent_classifier(enc_out_label, dec_out_label) # size (B, T, T + 1, num_constituent_labels). index by (batch_num, decoder_index, encoder_index + 1, label_index), which represents (batch_num, dependency_index, head_index + 1, label_index)

        # TASK 3: predict attachment ORDER

        self._mask_out(self_attention, lengths)
        self._mask_out(constituent_labels, lengths)

        # TODO: TASK 4: predict DEPENDENCY labels (according to GM 2022, this will improve overall performance in a multitask setting)

        indices = self._get_batch_indices(lengths)

        return self_attention, constituent_labels, indices

    def _mask_out(self, out: torch.Tensor, lengths: torch.Tensor):
        """mask out unneeded output elements IN PLACE, given sentence lengths

        Args:
            out (torch.Tensor): size (B, T, T + 1)
            lengths (torch.Tensor): sorted tensor in descending order where len(lengths) = B and lengths[0] = T
        """
        B, T, *_ = out.shape
        dependency_index_mask = torch.triu(torch.full((T + 1, T), True))[lengths].unsqueeze(-1).repeat(1, 1, T + 1) # (B, T, T + 1)
        out[dependency_index_mask] = -torch.inf

        head_index_mask = torch.triu(torch.full((T + 2, T + 1), True))[lengths + 1].unsqueeze(-2).repeat(1, T, 1) # (B, T, T + 1)
        out[head_index_mask] = -torch.inf

    def _get_batch_indices(self, lengths: torch.Tensor):
        """Get indices for a given set of sentence lengths.
           Suppose x is a tensor of size (B, T, *), which has been masked out
           Some elements of x are not needed, as they correspond to padding. This function returns the indices of the elements that are needed
           x[indices] will give you a tensor of size (N, *)

        Args:
            lengths (torch.Tensor): sorted tensor in descending order where len(lengths) = B and lengths[0] = T, the longest sentence _within the batch_

        Returns:
            torch.Tensor: indices of size (N, *) where each row is non-masked
        """

        T = lengths[0].item() # max sentence length within batch
        indices = ~torch.triu(torch.full((T + 1, T), True))[lengths] # type: ignore
        return indices
        