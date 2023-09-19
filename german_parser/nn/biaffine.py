import torch
import torch.nn as nn

class BiAffine(nn.Module):
    def __init__(self, num_classes: int, enc_input_size: int, dec_input_size: int, include_attention=False, check_accuracy: bool=False):
        super().__init__()

        self.num_classes = num_classes
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size

        self.Z = nn.Parameter(torch.zeros(self.num_classes, self.dec_input_size, self.enc_input_size), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)
        self.U_enc = nn.Parameter(torch.zeros(self.num_classes, self.enc_input_size), requires_grad=True)
        self.U_dec = nn.Parameter(torch.zeros(self.num_classes, self.dec_input_size), requires_grad=True)

        self.include_attention = include_attention
        self.w = None
        if self.include_attention:
            self.w = nn.Parameter(torch.zeros(self.num_classes), requires_grad=True)

        self.check_accuracy = check_accuracy
        self._reset_parameters()

    def forward(self, enc: torch.Tensor, dec: torch.Tensor):
        """calculate biaffine score between enc and dec

        Args:
            enc (torch.Tensor): tensor of size (B, T + 1, enc_input_size)
            dec (torch.Tensor): tensor of size (B, T, dec_input_size)
        """

        # BEGIN EINSUM METHOD
        interaction_score = torch.einsum("nij,bsj,bti->btsn", self.Z, enc, dec) # (B, T, T + 1, num_classes)
        enc_score         = torch.einsum("nj,bsj->bsn", self.U_enc, enc)        # (B, T + 1, num_classes)
        dec_score         = torch.einsum("ni,bti->btn", self.U_dec, dec)        # (B, T, num_classes)

        enc_score = enc_score.unsqueeze(1) # (B, 1, T + 1, num_classes)
        dec_score = dec_score.unsqueeze(2) # (B, T, 1,     num_classes)

        bias = self.b[None, None, None, :] # (1, 1, 1,     num_classes)
        # END EINSUM METHOD

        res = interaction_score + enc_score + dec_score + bias # (B, T, T + 1, num_classes)

        if self.include_attention:
            res = self.w @ res.tanh().transpose(-1, -2) # (B, T, T + 1)

        return res


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.U_enc)
        nn.init.xavier_uniform_(self.U_dec)
        nn.init.xavier_uniform_(self.Z)