import torch
from torch import nn


class MultipleNegativesRankingLoss(nn.Module):

    def __init__(self):
        super(MultipleNegativesRankingLoss, self).__init__()

    def forward(self, r1, r2):
        return self.multiple_negatives_ranking_loss(r1, r2)

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

    # Multiple Negatives Ranking Loss
    # Paper: https://arxiv.org/pdf/1705.00652.pdf
    #   Efficient Natural Language Response Suggestion for Smart Reply
    #   Section 4.4
    def multiple_negatives_ranking_loss(self, x1, x2):
        """
        Compute the loss over a batch with two embeddings per example.

        Each pair is a positive example. The negative examples are all other embeddings in embeddings_b with each embedding
        in embedding_a.

        See the paper for more information: https://arxiv.org/pdf/1705.00652.pdf
        (Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4)

        :param embeddings_a:
            Tensor of shape (batch_size, embedding_dim)
        :param embeddings_b:
            Tensor of shape (batch_size, embedding_dim)
        :return:
            The scalar loss
        """

        scores = torch.matmul(x1, x2.t())
        diagonal_mean = torch.mean(torch.diag(scores))
        mean_log_row_sum_exp = torch.mean(torch.logsumexp(scores, dim=1))
        return -diagonal_mean + mean_log_row_sum_exp
