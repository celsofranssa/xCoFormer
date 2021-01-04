import json
from statistics import mean

import nmslib
import numpy as np
import torch
from omegaconf import OmegaConf


class EvalHelper:
    def __init__(self, hparams):
        self.hparams = hparams

    def mrr_at_k(self, positions, k, num_samples):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        # positions_at_k = [p for p in positions if p <= k]
        positions_at_k = [p if p <= k else 0 for p in positions]
        rrank = 0.0
        for pos in positions_at_k:
            if pos != 0:
                rrank += 1.0 / pos
        return rrank / num_samples
        # return rrank / len(positions_at_k)

    def recall_at_k(self, positions, k, num_samples):
        """
        Evaluates the Recall considering only the positions up to k
        :param positions:
        :param k:
        :param num_samples:
        :return:
        """
        return 1.0 * sum(i <= k for i in positions) / num_samples

    def checkpoint_stats(self, stats, stats_path):
        with open(stats_path, "w") as stats_file:
            for data in stats:
                stats_file.write(f"{json.dumps(data)}\n")

    def load_predictions(self):
        # load predictions
        predictions = torch.load(self.hparams.model.predictions.path)

        descs = []
        codes = []

        for prediction in predictions:
            descs.append(prediction["r1"])
            codes.append(prediction["r2"])

        return descs, codes

    def init_index(self, codes):
        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='hnsw', space='cosinesimil')
        index.addDataPointBatch(codes)
        index.createIndex()
        return index

    def retrieve(self, index, descs, k=100):
        # retrieve
        neighbours = index.knnQueryBatch(descs, k=k)

        r_rank = []
        positions = []
        index_error = 0
        for qid, neighbour in enumerate(neighbours):
            rids, distances = neighbour
            try:
                p = np.where(rids == qid)[0][0]
                positions.append(p + 1)
                r_rank.append(1.0 / (p + 1))
            except IndexError:
                index_error += 1

        print("Out of Rank: ", index_error)
        print("MRR: ", mean(r_rank))
        return positions

    def rank(self):

        # load predictions
        descs, codes = self.load_predictions()

        index = self.init_index(codes)

        return self.retrieve(index, descs)

    def perform_eval(self):
        print(OmegaConf.to_yaml(self.hparams))

        thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
        stats = []
        positions = self.rank()

        for k in thresholds:
            stats.append(
                {
                    "k": k,
                    "metric": "MRR",
                    "value": self.mrr_at_k(positions, k,self.hparams.data.test.num_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )
            stats.append(
                {
                    "k": k,
                    "metric": "SSR",
                    "value": self.recall_at_k(positions, k, self.hparams.data.test.num_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )
        stats_path = self.hparams.stats.dir + self.hparams.model.name + "_" + self.hparams.data.name + ".stats"
        self.checkpoint_stats(stats, stats_path)
