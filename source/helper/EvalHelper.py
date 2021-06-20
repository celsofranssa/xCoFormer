import json
from statistics import mean

import nmslib
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


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
            json.dump(ranking, ranking_file)

    def load_predictions(self):
        # load predictions
        return torch.load(self.hparams.model.predictions.path)

    def init_index(self, predictions):
        M = 30
        efC = 100
        num_threads = 4
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='hnsw', space='cosinesimil')

        for prediction in tqdm(predictions, desc="Indexing"):
            index.addDataPoint(id=prediction["idx"], data=prediction["code_repr"])

        index.createIndex(index_time_params)
        return index

    def retrieve(self, index, predictions, k):
        # retrieve
        ranking = {}
        for prediction in tqdm(predictions, desc="Searching"):
            target_idx = prediction["idx"]
            ids, distances = index.knnQuery(prediction["desc_repr"], k=k)
            ids = ids.tolist()
            if target_idx in ids:
                ranking[target_idx] = ids.index(target_idx) + 1
            else:
                ranking[target_idx] = 1e9
        return ranking

    def get_ranking(self, k):

        # load predictions
        predictions = self.load_predictions()

        # index data
        index = self.init_index(predictions)

        # retrieve
        return self.retrieve(index, predictions, k=k)

    def perform_eval(self):

        thresholds = [1, 5, 10]
        stats = []
        ranking = self.get_ranking(k=thresholds[-1])

        for k in thresholds:
            stats.append(
                {
                    "k": k,
                    "metric": "MRR",
                    "value": self.mrr_at_k(ranking.values(), k, self.hparams.data.num_test_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )
            stats.append(
                {
                    "k": k,
                    "metric": "Recall",
                    "value": self.recall_at_k(ranking.values(), k, self.hparams.data.num_test_samples),
                    "model": self.hparams.model.name,
                    "datasets": self.hparams.data.name
                }
            )

        stats_path = self.hparams.stats.dir + self.hparams.model.name + "_" + self.hparams.data.name + ".stats"
        ranking_path = self.hparams.rankings.dir + self.hparams.model.name + "_" + self.hparams.data.name + ".ranking"

        self.checkpoint_stats(stats, stats_path)
        self.checkpoint_ranking(ranking, ranking_path)
