import torch
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("mrrs", default=[])

    def similarities(self, x1, x2):
        """
        Calculates the cosine similarity matrix for every pair (i, j),
        where i is an embedding from x1 and j is another embedding from x2.

        :param x1: a tensors with shape [batch_size, hidden_size].
        :param x2: a tensors with shape [batch_size, hidden_size].
        :return: the cosine similarity matrix with shape [batch_size, batch_size].
        """
        x1 = x1 / torch.norm(x1, dim=1, p=2, keepdim=True)
        x2 = x2 / torch.norm(x2, dim=1, p=2, keepdim=True)
        return torch.matmul(x1, x2.t())

    def update(self, r1, r2):
        distances = 1 - self.similarities(r1, r2)
        correct_elements = torch.unsqueeze(torch.diag(distances), dim=-1)
        batch_ranks = torch.sum(distances < correct_elements, dim=-1) + 1.0
        self.mrrs.append(torch.mean(1.0 / batch_ranks))

    def compute(self):
        return torch.mean(torch.tensor(self.mrrs))
