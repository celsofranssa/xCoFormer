import torch
from torch import nn
from pytorch_metric_learning import miners, losses

from source.miner.RelevanceMiner import RelevanceMiner


class NTXentLoss(nn.Module):

    def __init__(self, params):
        super(NTXentLoss, self).__init__()
        self.miner = RelevanceMiner(params.miner)
        self.criterion = losses.NTXentLoss(temperature=params.criterion.temperature)

    def forward(self, desc_idx, desc_rpr, code_idx, code_rpr):

        """
        Computes the NTXentLoss.
        """
        miner_outs = self.miner.mine(desc_ids=desc_idx, code_ids=torch.flatten(code_idx))
        # print(f"\n\n miner outs{miner_outs}\n\n")
        # print(f"\n\n desc_idx({desc_idx.shape}){desc_idx}\n\n")
        # print(f"\n\n code_idx({code_idx.shape}){code_idx}\n\n")
        return self.criterion(desc_rpr, None, miner_outs, code_rpr, None)

