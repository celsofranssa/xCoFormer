import pickle

import torch
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__(compute_on_cpu=True)
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.ranking = {}

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for text_idx, labels_ids in data.items():
            d = {}
            for label_idx in labels_ids:
                d[f"code_{label_idx}"] = 1.0
            relevance_map[f"desc_{text_idx}"] = d
        return relevance_map

    def update(self, descs_ids, descs_rprs, codes_ids, codes_rprs):

        scores = torch.einsum("ab,cb->ac", descs_rprs, codes_rprs) * self.params.scale

        for i, text_idx in enumerate(descs_ids.tolist()):
            for j, label_idx in enumerate(codes_ids.tolist()):
                if f"desc_{text_idx}" not in self.ranking:
                    self.ranking[f"desc_{text_idx}"] = {}
                self.ranking[f"desc_{text_idx}"][f"code_{label_idx}"] = scores[i][j].item() + self.ranking[
                    f"desc_{text_idx}"].get(
                    f"code_{label_idx}", 0)

    def compute(self):
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in self.ranking.keys()}),
            Run(self.ranking),
            ["mrr@1", "mrr@5", "mrr@10"]
        )
        return m

    def reset(self) -> None:
        self.ranking = {}
