import torch
from torch import nn


class NPairsLoss(nn.Module):
    """
    The N-Pairs Loss.
    It measures the loss given predicted tensors x1, x2 both with shape [batch_size, hidden_size],
    and target tensor y which is the identity matrix with shape  [batch_size, batch_size].
    """

    def __init__(self, alpha=100):
        super(NPairsLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha

    def similarities(self, x1, x2):
        """
        Calculates the cosine similarity matrix for every pair (i, j),
        where i is an embedding from x1 and j is another embedding from x2.

        :param x1: a tensors with shape [batch_size, hidden_size].
        :param x2: a tensors with shape [batch_size, hidden_size].
        :return: the cosine similarity matrix with shape [batch_size, batch_size].
        """
        x1 = x1 / torch.norm(x1, dim=1, keepdim=True)
        x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)
        return self.alpha * torch.matmul(x1, x2.t())

    def forward(self, predict, target):
        """
        Computes the N-Pairs Loss between the target and predictions.
        :param predict: the prediction of the model,
        Contains the batches x1 (image embeddings) and x2 (description embeddings).
        :param target: the identity matrix with shape  [batch_size, batch_size].
        :return: N-Pairs Loss value.
        """
        x1, x2 = predict
        predict = self.similarities(x1, x2)
        self.show(predict)
        # by construction the probability distribution must be concentrated on the diagonal of the similarities matrix.
        # so, Cross Entropy can be used to measure the loss.
        return self.ce(predict, target)
