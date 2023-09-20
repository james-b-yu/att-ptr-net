
import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils
import torch.nn.functional as F

from pydantic import BaseModel, Field
from collections.abc import Callable
from typing import Literal, Sequence

from ..nn import LSTM, LSTMSkip
from ..nn import BiAffine, MAffine

from .words import WordEmbedding

from ..util import BatchUnionFind

class TigerModel(nn.Module):
    class WordEmbeddingParams(BaseModel):
        char_set: dict[str, int]
        char_flag_generators: list[Callable[[str], Literal[1, 0]]]
        char_internal_embedding_dim: int
        char_part_embedding_dim: int
        word_part_embedding_dim: int
        char_internal_window_size: int
        word_dict: dict[int, str]
        unk_rate: float

    class LSTMParams(BaseModel):
        hidden_size: int
        bidirectional: bool = Field(default=False)
        num_layers: int = Field(default=1)
        dropout: float = Field(default=0.2)

    def __init__(self, word_embedding_params: WordEmbeddingParams, enc_lstm_params: LSTMParams, dec_lstm_params: LSTMParams, enc_attention_mlp_dim: int, dec_attention_mlp_dim: int, enc_label_mlp_dim: int, dec_label_mlp_dim: int, enc_attachment_mlp_dim: int, dec_attachment_mlp_dim: int, enc_pos_mlp_dim: int, dec_pos_mlp_dim: int, enc_morph_mlp_dim: int, dec_morph_mlp_dim: int, max_attachment_order: int, num_constituent_labels: int, num_terminal_poses: int, morph_prop_classes: Sequence[int], morph_pos_interaction_dim: int, num_biaffine_attention_classes=2, beam_size=10):
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
            word_dict=self.word_embedding_params.word_dict,
            unk_rate=self.word_embedding_params.unk_rate
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

        self.enc_attachment_mlp_dim = enc_attachment_mlp_dim
        self.dec_attachment_mlp_dim = dec_attachment_mlp_dim

        self.enc_pos_mlp_dim = enc_pos_mlp_dim
        self.dec_pos_mlp_dim = dec_pos_mlp_dim

        self.enc_morph_mlp_dim = enc_morph_mlp_dim
        self.dec_morph_mlp_dim = dec_morph_mlp_dim

        self.enc_attention_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_attention_mlp_dim)
        self.dec_attention_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_attention_mlp_dim)

        self.enc_label_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_label_mlp_dim)
        self.dec_label_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_label_mlp_dim)

        self.enc_pos_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_pos_mlp_dim)
        self.dec_pos_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_pos_mlp_dim)


        self.enc_attachment_mlp = nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_attachment_mlp_dim)
        self.dec_attachment_mlp = nn.Linear(self.dec_lstm_params.hidden_size, self.dec_attachment_mlp_dim)

        # define morphology mlps
        self.morph_prop_classes = morph_prop_classes
        self.enc_morph_mlps = nn.ModuleList([nn.Linear(2 * self.enc_lstm_params.hidden_size, self.enc_morph_mlp_dim) for _ in self.morph_prop_classes])
        self.dec_morph_mlps = nn.ModuleList([nn.Linear(self.dec_lstm_params.hidden_size, self.dec_morph_mlp_dim) for _ in self.morph_prop_classes])

        # define morphology part-of-speech interaction parameter
        self.morph_pos_interaction_dim = morph_pos_interaction_dim
        self.morph_pos_interactor = nn.Parameter(torch.zeros(1, num_terminal_poses, self.morph_pos_interaction_dim))

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

        # define biaffine layer for classification of attachment orders
        self.biaffine_attachment_classifier = BiAffine(
            num_classes=max_attachment_order,
            enc_input_size=self.enc_attachment_mlp_dim,
            dec_input_size=self.dec_attachment_mlp_dim,
            include_attention=False
        )

        # define biaffine layer for classification of terminal parts-of-speech
        self.biaffine_terminal_classifier = BiAffine(
            num_classes=num_terminal_poses,
            enc_input_size=self.enc_pos_mlp_dim,
            dec_input_size=self.dec_pos_mlp_dim,
            include_attention=False
        )

        # TODO: experiment with a 4-affine layer
        # define 3-affine layer for morphologies
        self.affine_morph_classifiers = nn.ModuleList([
            MAffine(n, self.dec_morph_mlp_dim, self.enc_morph_mlp_dim, self.morph_pos_interaction_dim)
            for n in self.morph_prop_classes
        ])

        self.beam_size = beam_size

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
            tuple[torch.Tensor, ...]: self_attention (B, T, T + 1), constituent_labels (B, T, num_constituent_labels), attachment_orders (B, T, max_attachment_order), indices (used to get a tensor of shape (N, *) by disgarding all unneeded elements in second-dimension of the previous tensors)
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
        self_attention: torch.Tensor = self.biaffine_attention(enc_out_attention, dec_out_attention) # size (B, T, T + 1). index by (batch_num, decoder_index, encoder_index + 1), which represents (batch_num, dependency_index, head_index + 1)

        # TASK 2: predict ATTACHMENT labels
        # for batch b and dependency index j and head index i, constituent_lables[b, j, i] gives logits to classify the label of the dependency from word j to HEAD word i
        enc_out_label = F.elu(self.enc_label_mlp(enc_out_pad))
        dec_out_label = F.elu(self.dec_label_mlp(dec_out_pad))
        constituent_labels: torch.Tensor = self.biaffine_constituent_classifier(enc_out_label, dec_out_label) # size (B, T, T + 1, num_constituent_labels). index by (batch_num, decoder_index, encoder_index + 1, label_index), which represents (batch_num, dependency_index, head_index + 1, label_index)

        # TASK 3: predict attachment ORDER
        enc_out_attachment = F.elu(self.enc_attachment_mlp(enc_out_pad))
        dec_out_attachment = F.elu(self.dec_attachment_mlp(dec_out_pad))
        attachment_orders: torch.Tensor = self.biaffine_attachment_classifier(enc_out_attachment, dec_out_attachment) # size (B, T, T + 1, max_attachment_order)

        # TASK 4: predict POS
        enc_out_pos = F.elu(self.enc_pos_mlp(enc_out_pad))
        dec_out_pos = F.elu(self.dec_pos_mlp(dec_out_pad))
        poses: torch.Tensor = self.biaffine_terminal_classifier(enc_out_pos, dec_out_pos)

        # TASK 5: predict MORPHs
        morphs: list[torch.Tensor] = []
        for i in range(len(self.morph_prop_classes)):
            enc_out_morph = F.elu(self.enc_morph_mlps[i](enc_out_pad))
            dec_out_morph = F.elu(self.dec_morph_mlps[i](dec_out_pad))
            morph = self.affine_morph_classifiers[i](dec_out_morph, enc_out_morph, self.morph_pos_interactor.expand(B, -1, -1), num_batch_dims=1)
            self._mask_out_(morph, lengths)
            morphs.append(morph)
            pass

        self._mask_out_(self_attention, lengths)
        self._mask_out_(constituent_labels, lengths)
        self._mask_out_(attachment_orders, lengths)
        self._mask_out_(poses, lengths)

        # TODO: TASK -1: predict DEPENDENCY labels (according to GM 2022, this will improve overall performance in a multitask setting)

        indices = self._get_batch_indices(lengths)

        return self_attention, constituent_labels, poses, attachment_orders, *morphs, indices

    def _mask_out_(self, out: torch.Tensor, lengths: torch.Tensor):
        """mask out unneeded output elements IN PLACE, given sentence lengths

        Args:
            out (torch.Tensor): size (B, T, T + 1)
            lengths (torch.Tensor): sorted tensor in descending order where len(lengths) = B and lengths[0] = T
        """
        B, T, *_ = out.shape
        dependency_index_mask = torch.triu(torch.full((T + 1, T), True))[lengths].unsqueeze(-1).expand(-1, -1, T + 1) # (B, T, T + 1)
        head_index_mask = torch.triu(torch.full((T + 2, T + 1), True))[lengths + 1].unsqueeze(-2).expand(-1, T, -1) # (B, T, T + 1)

        out[dependency_index_mask | head_index_mask] = -torch.inf

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
        

    def find_tree(self, input: tuple[torch.Tensor, torch.Tensor], new_words_dict: dict[int, str] | None):
        """input (tuple[torch.Tensor, torch.Tensor]): tuple of (data, sentence_lengths), where data is a tensor of size (B, T) and sentence_lengths is a tensor of size (B,). B is batch size, T is max(sentence_length) across all batches. The input must be sorted in descending order of sentence length
            new_words_dict (dict[int, str] | None): dictionary of new words. positive indices in new_words_dict correspond to negative indices in input[0] (data). If None, then all unknown words must be coded as 0

        Args:
            input (tuple[torch.Tensor, torch.Tensor]): words, lengths
            new_words_dict (dict[int, str] | None): _description_
        """    


        _, lengths = input
        self_attention, constituent_labels, poses, attachment_orders, *morphs, indices = self.forward(input, new_words_dict)

        return self._find_tree(lengths, self_attention, constituent_labels, attachment_orders, indices)
    
    def _find_tree(self, lengths: torch.Tensor, self_attention: torch.Tensor, constituent_labels: torch.Tensor, attachment_orders: torch.Tensor, indices: torch.Tensor):
        # self_attention has size (B, T, T + 1)
        # NOTE: self_attention indices are 1-indexed. index 0 corresponds to virtual root of D-tree (which is different from virtual root of C-tree)

        B, T, *_ = self_attention.shape
        uf = BatchUnionFind(B, self.beam_size, N=T + 1, device=self.dummy_param.device)

        # initialise beams by finding top-K most probable root nodes
        best_roots = self_attention[:, :, 0].topk(k=self.beam_size, dim=-1)

        current_root_indices = best_roots.indices # (B, K)
        joint_logits = best_roots.values # (B, K)

        edges = torch.zeros(B, self.beam_size, T, dtype=torch.long, device=self.dummy_param.device) # (B, K, T); m[b, k, t - 1] is the 1-indexed parent of 1-indexed node t, in batch b, beam k

        same_as_beams = torch.arange(self.beam_size, dtype=torch.long, device=self.dummy_param.device).repeat(B, 1) # m[b, k] == 1 if beam k in batch b is equal to beam 1 in batch b. allows us to keep track of duplicates. duplicates are when m[b, k] != k. Beams begin unique because the top-k best roots as initialised above will be unique
        beam_check = torch.arange(self.beam_size, dtype=torch.long, device=self.dummy_param.device) # used multiple times
        beam_check_repeated = beam_check.repeat(B, 1)
        parents = torch.arange(0, T + 1, 1, device=self.dummy_param.device, dtype=torch.long).repeat(B, self.beam_size, 1) # (B, K, T + 1), where m[b, k, t] = t to represent the index of each parent # these same indices are used multiple times

        def beams_are_unique():
            return same_as_beams == beam_check_repeated # return where m[b, k] == same_as_beams[b, k] != k


        for t in range(T):
            relevant_batches = t < lengths # used for updating the final arcs. otherwise, we get nans in batch b if t >= sentence_length[b]

            arc_probs = self_attention[:, t, :].log_softmax(dim=-1) # (B, T + 1)

            candidate_joint_logits = joint_logits[:, :, None] + arc_probs[:, None, :] # (B, K, T + 1); the heuristic we would like to maximise

            children = torch.tensor(t + 1, device=self.dummy_param.device) # all batches and beams share the same child index (1-indexed)

            # prevent cycles
            disable_mask = uf.is_same_set(children.expand_as(parents), parents) # (B, K, T + 1); m[b, k, s + 1] is true if in batch b, beam k, joining child (t + 1) and parent (s + 1) would lead to a cycle
            # avoid setting words as head that are beyond the end of the sentence
            disable_mask[:, :, 1:] |= ~indices.unsqueeze(1).to(device=self.dummy_param.device) 
            # force beams to be unique
            disable_mask |= ~beams_are_unique()[:, :, None]
            # force root indices to be enabled
            disable_mask[:, :, 1:][current_root_indices == t] = True # for the batches and beams where t would be a root node, t's parent must be 0 (cannot be 1:)
            # prevent other indices from becoming root
            disable_mask[:, :, 0][current_root_indices != t] = True
            candidate_joint_logits[disable_mask] = -torch.inf # these indices can never be a maximiser

            flattened_top_candidate_idx = candidate_joint_logits.flatten(-2, -1).topk(k=self.beam_size, dim=-1).indices # (B, K) in range [0, (K * T + 1)); for each batch, find top 10 best performing parent-beam combinations
            
            top_parents = parents.flatten(-2, -1).gather(index=flattened_top_candidate_idx, dim=-1) # (B, K); for each batch and beam, get the 1-indexed id of the parent we want to attach

            batch_names = beam_check.view(1, -1, 1).expand(B, -1, T + 1).flatten(-2, -1) # (B, K, T + 1); m[b, k, s] = k for all b, s, k
            used_batches = batch_names.gather(index=flattened_top_candidate_idx, dim=-1) # (B, K) where each element is in the range [0, K). m[b, k] tells you what the kth new beam should be

            same_as_beams = used_batches # we have copied over these beams, so for now, these beams must be equal

            new_data = uf.data.gather(index=used_batches.unsqueeze(-1).expand_as(uf.data), dim=1)
            new_rank = uf.rank.gather(index=used_batches.unsqueeze(-1).expand_as(uf.rank), dim=1)
            new_edges = edges.gather(index=used_batches.unsqueeze(-1).expand_as(edges), dim=1)
            new_joint_logits = candidate_joint_logits.flatten(-2, -1).gather(index=flattened_top_candidate_idx, dim=-1)
            new_roots = current_root_indices.gather(index=used_batches, dim=-1)
            

            uf.data[relevant_batches] = new_data[relevant_batches]
            uf.rank[relevant_batches] = new_rank[relevant_batches]
            edges[relevant_batches] = new_edges[relevant_batches]
            joint_logits[relevant_batches] = new_joint_logits[relevant_batches]
            current_root_indices[relevant_batches] = new_roots[relevant_batches]

            uf.union(children.expand_as(top_parents), top_parents)

            edges[:, :, t] = top_parents
            new_unique_beams = top_parents.gather(index=same_as_beams, dim=-1) != top_parents  # indicates the beams that have BECOME unqiue: given any batch b, suppose beam j was a copy of beam k. suppose the new parent for beam j is different to the new parent of beam k. then m[b, j] = True
            same_as_beams[new_unique_beams] = beam_check_repeated[new_unique_beams]

            pass

        num_labels = constituent_labels.shape[-1]
        num_attachment_orders = attachment_orders.shape[-1]

        best_edges = edges[torch.arange(edges.shape[0]), joint_logits.argmax(dim=-1)] # (B, T) containing elements in range [0, T + 1), where m[b, t - 1] denotes the 1-indexed parent of 1-indexed node t

        label_logits_best_edges = constituent_labels.gather(index=best_edges[:, :, None, None].expand(-1, -1, -1, num_labels), dim=2).squeeze(2)
        attachment_logits_best_edges = attachment_orders.gather(index=best_edges[:, :, None, None].expand(-1, -1, -1, num_attachment_orders), dim=2).squeeze(2)

        labels_best_edges = label_logits_best_edges.argmax(-1)
        attachment_orders_best_edges = attachment_logits_best_edges.argmax(-1)

        return best_edges, labels_best_edges, attachment_orders_best_edges, (edges, joint_logits)