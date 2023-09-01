import torch
import torch.nn as nn
from torch.nn.utils import rnn as rnn_utils

class LSTM(nn.Module): # batch_first = True
    def __init__(self, input_size, hidden_size, bidirectional=True, num_layers=1, dropout_rate=0.2):
        """_summary_

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            bidirectional (bool, optional): _description_. Defaults to True.
            num_layers (int, optional): _description_. Defaults to 1.
            dropout (float, optional): _description_. Defaults to 0.2. Only applies to non-final layers (so no effect if num_layers = 1)
        """
        super().__init__()

        self.input_size = input_size   # E_in
        self.hidden_size = hidden_size # H_in
        self.num_layers = num_layers   # H_cell
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate if num_layers > 1 else 0        # dropout rate

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout_rate,
            bidirectional=self.bidirectional,
            batch_first=True)
        
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, *args):
        """compute BiLSTM

        Args:
            x (*Any): input directly to pytorch LSTM

        Returns:
            _type_: _description_
        """
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        out, (h, c) = self.lstm(*args) # output of size (B, T, D * hidden_size), where D = 1 if not bidirectional else 2; h and c are of size (num_layers * D, B, hidden_size)

        return out, (h, c)
    


class LSTMSkip(nn.Module): # batch_first = True
    def __init__(self, input_size, hidden_size, bidirectional=True, num_layers=1, dropout_rate=0.2):
        """_summary_

        Args:
            input_size (_type_): _description_
            hidden_size (_type_): _description_
            bidirectional (bool, optional): _description_. Defaults to True.
            num_layers (int, optional): _description_. Defaults to 1.
            dropout (float, optional): _description_. Defaults to 0.2. Only applies to non-final layers (so no effect if num_layers = 1)
        """
        super().__init__()

        num_layers = int(num_layers)
        assert num_layers >= 1

        self.input_size = input_size   # E_in
        self.hidden_size = hidden_size # H_in
        self.num_layers = num_layers   # H_cell
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate if num_layers > 1 else 0        # dropout rate

        self.chain = nn.ModuleList(
            [nn.LSTM(input_size=self.input_size if i == 0 else ((2 if self.bidirectional else 1) * self.hidden_size), 
            hidden_size=self.hidden_size, 
            num_layers=1, 
            dropout=0.0,
            bidirectional=self.bidirectional,
            batch_first=True) for i in range(self.num_layers)]
        )

        self.dropout = nn.Dropout(p=self.dropout_rate, inplace=False)
        
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def _dropout_packed_sequence(self, packed_sequence: rnn_utils.PackedSequence | torch.Tensor, batch_first: bool=True, enforce_sorted=True):

        if isinstance(packed_sequence, torch.Tensor):
            res: torch.Tensor = self.dropout(packed_sequence)
            return res
        
        assert isinstance(packed_sequence, rnn_utils.PackedSequence)

        unpacked, lengths = rnn_utils.pad_packed_sequence(sequence=packed_sequence, batch_first=batch_first)
        unpacked = self.dropout(unpacked)

        return rnn_utils.pack_padded_sequence(unpacked, lengths, batch_first, enforce_sorted)
    
    def _add_packed_sequences(self, packed_sequence_a: rnn_utils.PackedSequence | torch.Tensor, packed_sequence_b: rnn_utils.PackedSequence | torch.Tensor, batch_first: bool=True, enforce_sorted=True):

        if isinstance(packed_sequence_a, torch.Tensor) and isinstance(packed_sequence_b, torch.Tensor):
            return packed_sequence_a + packed_sequence_b
        
        assert isinstance(packed_sequence_a, rnn_utils.PackedSequence) and isinstance(packed_sequence_b, rnn_utils.PackedSequence)

        unpacked_a, lengths = rnn_utils.pad_packed_sequence(sequence=packed_sequence_a, batch_first=batch_first)
        unpacked_b, lengths = rnn_utils.pad_packed_sequence(sequence=packed_sequence_b, batch_first=batch_first)
        
        res = unpacked_a + unpacked_b

        return rnn_utils.pack_padded_sequence(res, lengths, batch_first, enforce_sorted)

    def forward(self, *args):
        """compute BiLSTM. Args gets directly fed into first layer (so you can define initial hidden state of first layer). Initial hidden state of all subsequent layers will be zeros.

        Args:
            x (*Any): input directly to PyTorch FIRST layer of the LSTM. Must be batched (of dimension 3) and batch-first. 

        Returns:
            _type_: _description_
        """
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        res: tuple[torch.Tensor | rnn_utils.PackedSequence, tuple[torch.Tensor, torch.Tensor]] | None = None

        layer_hiddens: list[torch.Tensor | rnn_utils.PackedSequence] = []
        for i in range(self.num_layers):
            args_in: tuple | None = None

            if i == 0:
                args_in = args

            else:
                x_in = layer_hiddens[-1]
                
                # add skip connections when processing final layer
                if i + 1 == self.num_layers:
                    for j in range(i - 1):
                        x_in = self._add_packed_sequences(x_in, layer_hiddens[j])

                args_in = (x_in,)

            assert args_in is not None

            res = self.chain[i](*args_in)
            assert res is not None
            hidden_out = res[0]

            layer_hiddens.append(self._dropout_packed_sequence(hidden_out))

        assert res is not None

        return res # output of size (B, T, D * hidden_size), where D = 1 if not bidirectional else 2; h and c are of size (num_layers * D, B, hidden_size)