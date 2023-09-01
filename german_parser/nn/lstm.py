import torch
import torch.nn as nn

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

        self.dropout = nn.Dropout(p=self.dropout_rate, inplace=True)
        
        self.dummy_param = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, *args):
        """compute BiLSTM. Args gets directly fed into first layer (so you can define initial hidden state of first layer). Initial hidden state of all subsequent layers will be zeros.

        Args:
            x (*Any): input directly to PyTorch FIRST layer of the LSTM. Must be batch-first. If is not in a batch, will batch into a batch of one sample

        Returns:
            _type_: _description_
        """
        # h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        # c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)

        res: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] | None = None

        layer_hiddens: list[torch.Tensor] = []
        for i in range(self.num_layers):
            args_in: tuple | None = None

            if i == 0:
                args_in = args
                assert isinstance(args_in[0], torch.Tensor), "First argument of LSTM input must be a tensor"
                if args_in[0].dim() != 3:
                    args_in[0].unsqueeze_(0)
            else:
                x_in = layer_hiddens[-1]
                
                # add skip connections
                for j in range(i - 1):
                    x_in += layer_hiddens[j]

                args_in = (x_in,)

            assert args_in is not None

            res = self.chain[i](*args_in)
            assert res is not None
            hidden_out = res[1][1].permute(1, 0, 2)

            if i + 1 != self.num_layers:
                self.dropout(hidden_out) # dropout in-place

            layer_hiddens.append(hidden_out)

        assert res is not None

        return res # output of size (B, T, D * hidden_size), where D = 1 if not bidirectional else 2; h and c are of size (num_layers * D, B, hidden_size)