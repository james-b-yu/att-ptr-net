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

        # BEGIN NAIVE METHOD
        # dec_brd = dec[:, :, None, None, :, None]         # (B, T, 1,     1,           dec_input_size, 1)
        # Z = self.Z[None, None, None, :, :, :]            # (1, 1, 1,     num_classes, dec_input_size, enc_input_size)
        # enc_brd = enc[:, None, :, None, :, None]         # (B, 1, T + 1, 1,           enc_input_size, 1)

        # # (B, T, T + 1, num_classes) index via [batch_number, DECoder_index, ENCoder_index]
        # interaction_score = (dec_brd.transpose(-1, -2) @ Z @ enc_brd).squeeze(-1, -2)

        # dec_brd = dec_brd.squeeze(3)
        # enc_brd = enc_brd.squeeze(3)

        # enc_score = (self.U_enc @ enc_brd).squeeze(-1) # (B, 1, T + 1, num_classes)
        # dec_score = (self.U_dec @ dec_brd).squeeze(-1) # (B, T, 1, num_classes)

        # bias = self.b[None, None, None, :]             # (1, 1, 1, num_classes)
        # END NAIVE METHOD

        # BEGIN EINSUM METHOD
        interaction_score = torch.einsum("nij,bsj,bti->btsn", self.Z, enc, dec) # (B, T, T + 1, num_classes)
        enc_score         = torch.einsum("nj,bsj->bsn", self.U_enc, enc)        # (B, T + 1, num_classes)
        dec_score         = torch.einsum("ni,bti->btn", self.U_dec, dec)        # (B, T, num_classes)

        enc_score = enc_score.unsqueeze(1) # (B, 1, T + 1, num_classes)
        dec_score = dec_score.unsqueeze(2) # (B, T, 1,     num_classes)

        bias = self.b[None, None, None, :] # (1, 1, 1,     num_classes)
        # END EINSUM METHOD

        res = interaction_score + enc_score + dec_score + bias # (B, T, T + 1, num_classes)

        # # check correctness
        # if self.check_accuracy:
        #     n_batches = enc.shape[0]
        #     seq_length = dec.shape[1]
        #     assert enc.shape[1] == seq_length + 1, "Encoder output must have one more item than decoder, as the first item denotes ROOT"

        #     for batch_num in range(n_batches):
        #         for c in range(self.num_classes):
        #             for i in range(seq_length + 1): # encoder index
        #                 for j in range(seq_length): # decoder index
        #                     res_val = res[batch_num, j, i, c]

        #                     true_val = dec[batch_num, j] @ self.Z[c] @ enc[batch_num, i] + self.U_enc[c] @ enc[batch_num, i] + self.U_dec[c] @ dec[batch_num, j] + self.b[c]
                        
        #                     print((res_val.item() - true_val.item()) / true_val.item())

        if self.include_attention:
            res = self.w @ res.tanh().transpose(-1, -2) # (B, T, T + 1)

        return res


    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.U_enc)
        nn.init.xavier_uniform_(self.U_dec)
        nn.init.xavier_uniform_(self.Z)

        # return
        # with torch.no_grad():
        #     Zb_var = 2 * (((self.enc_input_size ** 0.5) * (self.dec_input_size ** 0.5)) ** -1)
        #     self.Z.normal_(0, Zb_var)
        #     self.b.normal_(0, Zb_var)

        #     U_enc_var = 2 * (self.enc_input_size ** -1)
        #     self.U_enc.normal_(0, U_enc_var)

        #     U_dec_var = 2 * (self.dec_input_size ** -1)
        #     self.U_dec.normal_(0, U_dec_var)

        #     if self.include_attention:
        #         w_var = 2 * (self.num_classes ** -1)
        #         self.w.normal_(0, w_var)