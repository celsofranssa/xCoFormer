import torch
from torch import nn


class TripletLoss(nn.Module):
    """
    The N-Pairs Loss.
    It measures the loss given predicted tensors x1, x2 both with shape [batch_size, hidden_size],
    and target tensor y which is the identity matrix with shape  [batch_size, batch_size].
    """

    def __init__(self, hparams):
        super(TripletLoss, self).__init__()
        self.triplet_loss = nn.TripletMarginLoss()
        self.name = hparams.name

    def forward(self, anchor, positive):
        """
        Computes the N-Pairs Loss between the target and prediction.
        :param positive:
        :param anchor:
        :param predict: the prediction of the model,
        Contains the batches x1 (image embeddings) and x2 (description embeddings).
        :param target: the identity matrix with shape  [batch_size, batch_size].
        :return: N-Pairs Loss value.
        """
        shuffle_index = torch.randperm(anchor.shape[0])

        negative = positive[[shuffle_index]]
        return self.triplet_loss(anchor, positive, negative)
