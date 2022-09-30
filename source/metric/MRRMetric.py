import pickle
import nmslib
import torch
from omegaconf import OmegaConf
from ranx import Qrels, Run, evaluate
from torchmetrics import Metric


class MRRMetric(Metric):
    def __init__(self, params):
        super(MRRMetric, self).__init__(compute_on_cpu=True)
        self.params = params
        self.relevance_map = self._load_relevance_map()
        self.descs = []
        self.codes = []

    def _load_relevance_map(self):
        with open(f"{self.params.relevance_map.dir}relevance_map.pkl", "rb") as relevances_file:
            data = pickle.load(relevances_file)
        relevance_map = {}
        for desc_idx, code_ids in data.items():
            d = {}
            for code_idx in code_ids:
                d[f"code_{code_idx}"] = 1.0
            relevance_map[f"desc_{desc_idx}"] = d
        return relevance_map

    def update(self, desc_idx, desc_rpr, code_idx, code_rpr):

        for desc_idx, desc_rpr in zip(
                desc_idx.tolist(),
                desc_rpr.tolist()):
            self.descs.append({"desc_idx": desc_idx, "desc_rpr": desc_rpr})

        for code_idx, code_rpr in zip(
                code_idx.tolist(),
                code_rpr.tolist()):
            self.codes.append({"code_idx": code_idx, "code_rpr": code_rpr})

    def init_index(self):

        # initialize a new index, using a HNSW index on l2 space
        index = nmslib.init(method='hnsw', space='l2')

        for code in self.codes:
            index.addDataPoint(id=code["code_idx"], data=code["code_rpr"])

        index.createIndex(
            index_params=OmegaConf.to_container(self.params.index),
            print_progress=False
        )

        return index

    def retrieve(self, index, num_nearest_neighbors):
        ranking = {}
        index.setQueryTimeParams({'efSearch': 2048})
        for desc in self.descs:
            desc_idx = desc["desc_idx"]
            retrieved_ids, distances = index.knnQuery(desc["desc_rpr"], k=num_nearest_neighbors)
            for code_idx, distance in zip(retrieved_ids, distances):
                if f"desc_{desc_idx}" not in ranking:
                    ranking[f"desc_{desc_idx}"] = {}
                ranking[f"desc_{desc_idx}"][f"code_{code_idx}"] = 1.0 / (distance + 1e-9)

        return ranking

    def compute(self):
        # index
        index = self.init_index()

        # retrive
        ranking = self.retrieve(index, num_nearest_neighbors=self.params.num_nearest_neighbors)

        # eval
        m = evaluate(
            Qrels({key: value for key, value in self.relevance_map.items() if key in ranking.keys()}),
            Run(ranking),
            ["mrr"]
        )

        return m

    def reset(self) -> None:
        self.descs = []
        self.codes = []
