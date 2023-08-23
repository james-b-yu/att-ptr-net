import torch
import torch.nn as nn

class LSTM(nn.Module): # batch_first = True
    def __init__(self, input_size, hidden_size, bidirectional=True, num_layers=1, dropout=0.2):
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
        self.dropout = dropout if num_layers > 1 else 0        # dropout rate

        self.lstm = nn.LSTM(
            input_size=self.input_size, 
            hidden_size=self.hidden_size, 
            num_layers=self.num_layers, 
            dropout=self.dropout,
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