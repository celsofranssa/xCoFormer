import torch
from torch import nn, Tensor
from typing import Iterable, Dict

class MNRankingLoss(nn.Module):
    """
    Multiple Negatives Ranking Loss.
    """

    def __init__(self, params):
        super(MNRankingLoss, self).__init__()
        self.params=params
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def cos_sim(self, r1, r2):
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        r1_norm = torch.nn.functional.normalize(r1, p=2, dim=1)
        r2_norm = torch.nn.functional.normalize(r2, p=2, dim=1)
        return torch.mm(r1_norm, r2_norm.transpose(0, 1))


    def forward(self, r1, r2):
        scores = self.cos_sim(r1, r2) * self.params.scale
        labels = torch.tensor(range(len(scores)), dtype=torch.long, device=scores.device)  # Example a[i] should match with b[i]
        return self.cross_entropy_loss(scores, labels)
