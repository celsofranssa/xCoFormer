import pickle

import torch
from pytorch_metric_learning.miners import BaseMiner


class RelevanceMiner(BaseMiner):

    def __init__(self, params):
        super().__init__()
        with open(f"{params.relevance_map.dir}relevance_map.pkl", "rb") as relevance_map_file:
            self.relevance_map = pickle.load(relevance_map_file)

    def mine(self, desc_ids, code_ids):
        a1, p, a2, n = [], [], [], []
        for i, desc_idx in enumerate(desc_ids.tolist()):
            for j, code_idx in enumerate(code_ids.tolist()):
                if code_idx in self.relevance_map[desc_idx]:
                    a1.append(i)
                    p.append(j)
                else:
                    a2.append(i)
                    n.append(j)

        return torch.tensor(
            a1, device=desc_ids.device), torch.tensor(
            p, device=desc_ids.device), torch.tensor(
            a2, device=desc_ids.device), torch.tensor(
            n, device=desc_ids.device)

    def output_assertion(self, output):
        pass
