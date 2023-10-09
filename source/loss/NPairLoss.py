import torch
from torch import nn, Tensor
from typing import Iterable, Dict


class NPairLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss.
    """

    def __init__(self, params):
        super(NPairLoss, self).__init__()
        self.params = params
        self.criterion = nn.CrossEntropyLoss()


    def forward(self, desc_repr, code_repr):
        scores = torch.einsum("ab,cb->ac", desc_repr, code_repr) * self.params.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long,
                              device=scores.device)  # Example a[i] should match with b[i]
        return self.criterion(scores, labels)
