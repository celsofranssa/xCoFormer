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

    def mrr(self, ranking):
        """
        Evaluates the MMR considering only the positions up to k.
        :param positions:
        :param num_samples:
        :return:
        """
        return np.mean(ranking)

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

    def checkpoint_ranking(self, ranking, ranking_path):
        with open(ranking_path, "w") as ranking_file:
            ranking_file.write(f"{ranking}\n")

    def load_predictions(self):
        # load predictions
        return torch.load(self.hparams.model.predictions.path)

    def init_index(self, predictions):
        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='brute_force', space='cosinesimil')

        for prediction in predictions:
            index.addDataPoint(id=prediction["idx"], data=prediction["code_repr"])

        index.createIndex()
        return index

    def retrieve(self, index, predictions, k=100):
        # retrieve
        ranking = []
        for prediction in predictions:
            target_idx = prediction["idx"]
            ids, distances = index.knnQuery(prediction["desc_repr"], k=k)
            ids = ids.tolist()
            if target_idx in ids:
                ranking.append(ids.index(target_idx) + 1)
            else:
                ranking.append(1e9)
        return ranking

    def get_ranking(self):

        # load predictions
        predictions = self.load_predictions()

        index = self.init_index(predictions)

        return self.retrieve(index, predictions, k=self.hparams.data.num_test_samples)

    def perform_eval(self):
        thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, self.hparams.data.num_test_samples]
        stats = []
        ranking = self.get_ranking()

        for k in thresholds:
            stats.append(
                {
                    "k": k,
                    "metric": "MRR",
                    "value": self.mrr_at_k(ranking, k, self.hparams.data.num_test_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )
            stats.append(
                {
                    "k": k,
                    "metric": "Recall",
                    "value": self.recall_at_k(ranking, k, self.hparams.data.num_test_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )
        print(self.mrr(ranking))

        stats_path = self.hparams.stats.dir + self.hparams.model.name + "_" + self.hparams.data.name + ".stats"
        ranking_path = self.hparams.rankings.dir + self.hparams.model.name + "_" + self.hparams.data.name + ".ranking"

        self.checkpoint_stats(stats, stats_path)
        self.checkpoint_ranking(ranking, ranking_path)
