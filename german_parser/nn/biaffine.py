import torch
import torch.nn as nn

from functools import reduce

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


class MLinear(nn.Module):
    letters = "abcdefghijklmopqrstuvwxyz" # all letters except n

    def _index_generator(self, num_indices_to_skip=0):
        for c in (self.available_indexing_names[num_indices_to_skip:]):
            yield c

    def __init__(self, N: int, *S: int):
        """class for computing a linear transformation between $M$ vectors, where vector i lives in S[i]-dimensional space
        Args:
            N (int): the dimension of the output
            *S (int): the dimensions of the inputs. Should be of length $M$
        """
        super().__init__()

        self.S = S
        self.M = len(S)
        self.N = N

        self.Z = nn.Parameter(torch.zeros(N, *S)) # (N, S_{1}, ..., S_{M})

        self.summing_names = self.letters[-self.M:]
        self.available_indexing_names = self.letters[:-self.M]

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.Z)

    def forward(self, *X: torch.Tensor, num_batch_dims=0):
        """compute linear transformation given the M tensors in N
        Args:
            *X (torch.Tensor): sequence of M tensors, where X[i].shape[-1] == S[i]
        Returns:
            torch.Tensor: _description_
        """
        total_indexing_dim = sum(map(lambda x: x.dim() - 1 - num_batch_dims, X))
        i_gen = self._index_generator(num_batch_dims)

        batch_indexing_names = self.available_indexing_names[:num_batch_dims]

        z_indices = "n" + self.summing_names
        x_indices = ",".join([batch_indexing_names + "".join([next(i_gen) for j in range(m.dim() - 1 - num_batch_dims)]) + self.summing_names[i] for i, m in enumerate(X)])
        r_indices = batch_indexing_names + self.available_indexing_names[num_batch_dims:num_batch_dims + total_indexing_dim] + "n"

        return torch.einsum(f"{z_indices},{x_indices}->{r_indices}", self.Z, *X)

class MAffine(nn.Module):
    @classmethod
    def _combinations(cls, n, c):
        # Initialize the first combination (lexicographically smallest)
        combination = torch.arange(c)

        while combination[0] < n - c + 1:
            yield combination

            # Find the rightmost element that can be incremented
            j = c - 1
            while j >= 0 and combination[j] == n - c + j:
                j -= 1

            # Increment the rightmost element that can be incremented
            combination[j] += 1

            # Adjust the elements to the right
            for k in range(j + 1, c):
                combination[k] = combination[k - 1] + 1

    def __init__(self, N: int, *S: int):
        """class for computing an affine transformation between $M$ vectors, where vector i lives in S[i]-dimensional space
        Args:
            N (int): the dimension of the output
            *S (int): the dimensions of the inputs. Should be of length $M$
        """

        super().__init__()

        self.S = torch.tensor(S, dtype=torch.long)
        self.M = len(S)
        self.N = N
        
        self.b = nn.Parameter(torch.zeros(self.N)) # the bias term

        self.linears = nn.ModuleList()

        for k in range(1, self.M + 1):
            for comb in self._combinations(self.M, k):
                S_subset = self.S.gather(index=comb, dim=-1)
                self.linears.append(MLinear(self.N, *S_subset.numpy()))

    def forward(self, *X: torch.Tensor, num_batch_dims=0):
        assert len(X) == self.M

        indexing_dims = torch.tensor([num_batch_dims, *map(lambda x: len(x.shape) - 1 - num_batch_dims, X)], dtype=torch.long)
        indexing_dims_cum = indexing_dims.cumsum(dim=0)

        total_indexing_dims = int(indexing_dims_cum[-1])

        tensor_indices = torch.zeros(self.M, total_indexing_dims + 1, dtype=torch.long) # M[i, j] != 0 if in the result, the jth index corresponds to one of the indices we used to index tensor i. M[i, j] is the size of the jth dimension in the final result
        
        helper = torch.full((total_indexing_dims + 1, total_indexing_dims + 1), True).tril(diagonal=-1)

        tensor_indices_sizes = torch.tensor(reduce(lambda a, b: a + b.shape[num_batch_dims:-1], X, X[0].shape[:num_batch_dims]) + (self.N,), dtype=torch.long)

        tensor_indices[helper[indexing_dims_cum[1:]]] = 1
        tensor_indices[helper[indexing_dims_cum[:-1]]] = 0
        tensor_indices[:, -1] = 1 # final position of output tensor always corresponds to the self.N classes
        tensor_indices[:, :num_batch_dims] = 1 # initial positions of output tensor always correspond to the num_batch_indices

        lin_iter = iter(self.linears)

        all_pre_broadcast: list[torch.Tensor] = []

        for k in range(1, self.M + 1):
            for comb in self._combinations(self.M, k):
                S_subset = self.S.gather(index=comb, dim=-1)
                inputs = [X[int(i)] for i in comb]

                model: MLinear = next(lin_iter) #type:ignore

                res: torch.Tensor = model(*inputs, num_batch_dims=num_batch_dims)
                res_positions = tensor_indices[comb].any(dim=0) # v[i] = 1 if res's indices should correspond to the ith index of final output
                res_layout = tensor_indices_sizes.clone()

                res_layout[~res_positions] = 1

                res_pre_broadcast = res.view(*res_layout.numpy())
                all_pre_broadcast.append(res_pre_broadcast)
                pass

        return sum(all_pre_broadcast) + self.b.view([1] * total_indexing_dims + [self.N])