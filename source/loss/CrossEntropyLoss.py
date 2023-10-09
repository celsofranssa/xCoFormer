import torch
from torch import nn


class CrossEntropyLoss(nn.Module):
    """
    The N-Pairs Loss.
    It measures the loss given predicted tensors x1, x2 both with shape [batch_size, hidden_size],
    and target tensor y which is the identity matrix with shape  [batch_size, batch_size].
    """

    def __init__(self, params):
        super(CrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.params = params

    def forward(self, desc_repr, code_repr):
        scores = torch.einsum("ab,cb->ac", desc_repr, code_repr)
        return self.criterion(
            scores * 20,
            torch.arange(desc_repr.size(0),
            device=scores.device)
        )
