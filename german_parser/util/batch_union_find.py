import torch
from typing import Iterable

class BatchUnionFind:
    """class to perform batched union-find
    """
    def __init__(self, *B: int, N: int, device=torch.device("cpu")):
        """initialises prd(*B) independent union-find data-structures with N distinct elements

        Args:
            B: tuple indicating how we will be indexing into the prd(*B) union-finds
            N (int): the number of distinct elements
            device (torch.device, optional): the device on which to keep tensors. Defaults to torch.device("cpu").
        """
        self.data = torch.arange(N, device=device).repeat(*B, 1)
        self._rank = torch.zeros((*B, N), device=device)

        self.B = B
        self.N = N

    def _to_tensor(self, t: torch.Tensor | Iterable):
        t = torch.as_tensor(t)
        assert t.shape == (*self.B, 1) or t.shape == (*self.B, )
        if t.shape == (*self.B, 1):
            t = t.squeeze(-1)
        
        return t

    def find(self, f: torch.Tensor | Iterable):
        """find the principle element corresponding to each element in f

        Args:
            f (torch.Tensor | Iterable): tensor or iterable of shape (*B) or (*B, 1), whose elements are in the range [0, self.N)

        Returns:
            torch.Tensor: tensor of shape (*B) whose elements are in the range [0, self.N), containing values to the principle corresponding to each element of f
        """
        f = self._to_tensor(f)

        things_to_compress: list[torch.Tensor] = []

        next_f = self.data.gather(dim=-1, index=f.unsqueeze(-1)).squeeze(-1)

        while not (next_f == self.data.gather(dim=-1, index=next_f.unsqueeze(-1)).squeeze(-1)).all():
            things_to_compress.append(next_f)
            next_f = self.data.gather(dim=-1, index=next_f.unsqueeze(-1)).squeeze(-1)

        for c in things_to_compress:
            self.data.scatter_(dim=-1, index=c.unsqueeze(-1), src=next_f.unsqueeze(-1)).squeeze_(-1)
        
        return next_f
    
    def union(self, left: torch.Tensor | Iterable, right: torch.Tensor | Iterable):
        """perform the union (by rank) between the sets containing each element of left, and the sets containing each element of right

        Args:
            left (torch.Tensor | Iterable): tensor or iterable of shape (*B) or (*B, 1), whose elements are in the range [0, self.N)
            right (torch.Tensor | Iterable): tensor or iterable of shape (*B) or (*B, 1), whose elements are in the range [0, self.N)
        """
        left = self.find(left)
        right = self.find(right)

        left_ranks = self._rank.gather(dim=-1, index=left.unsqueeze(-1)).squeeze(-1)
        right_ranks = self._rank.gather(dim=-1, index=right.unsqueeze(-1)).squeeze(-1)

        comp = left_ranks < right_ranks

        parent = (right * comp + left * ~comp).long()
        child  = (left * comp  + right * ~comp).long()

        parent_ranks = (right_ranks * comp + left_ranks * ~comp).long()
        child_ranks  = (left_ranks * comp  + right_ranks * ~comp).long()

        self.data.scatter_(dim=-1, index=child.unsqueeze(-1), src=parent.unsqueeze(-1)).squeeze_(-1)

        self._rank[parent_ranks == child_ranks, parent[parent_ranks == child_ranks]] += 1

    def is_same_set(self, left: torch.Tensor | Iterable, right: torch.Tensor | Iterable):
        """check whether elements of left are in the same set as the elements of right

        Args:
            left (torch.Tensor | Iterable): tensor or iterable of shape (*B) or (*B, 1), whose elements are in the range [0, self.N)
            right (torch.Tensor | Iterable): tensor or iterable of shape (*B) or (*B, 1), whose elements are in the range [0, self.N)

        Returns:
            torch.Tensor[torch.bool]: boolean tensor of shape (*B), indicating whether for each prd(*B) union finds, the elements of left and right are in the same set
        """

        left = self.find(left)
        right = self.find(right)

        return self.data.gather(dim=-1, index=left.unsqueeze(-1)).squeeze(-1) == self.data.gather(dim=-1, index=right.unsqueeze(-1)).squeeze(-1)
