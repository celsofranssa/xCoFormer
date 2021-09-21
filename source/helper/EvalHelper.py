import json
import pickle
from pathlib import Path
import nmslib
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class EvalHelper:
    def __init__(self, params):
        self.params = params

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

    def checkpoint_stats(self, stats):
        """
        Checkpoints stats on disk.
        :param stats: dataframe
        """
        stats.to_csv(
            self.params.stat.dir + self.params.model.name + "_" + self.params.data.name + ".stat",
            sep='\t',index=False,header=True)


    def checkpoint_ranking(self, ranking):
        ranking_path = f"{self.params.ranking.dir}" \
                       f"{self.params.model.name}_" \
                       f"{self.params.data.name}.rnk"
        with open(ranking_path, "wb") as ranking_file:
            pickle.dump(ranking, ranking_file)

    def load_predictions(self, fold):

        predictions_paths = sorted(
            Path(f"{self.params.prediction.dir}fold_{fold}/").glob("*.prd")
        )

        predictions = []
        for path in tqdm(predictions_paths, desc="Loading predictions"):
            predictions.extend(torch.load(path))

        return predictions

    def init_index(self, predictions):
        M = 30
        efC = 100
        num_threads = 4
        index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

        # initialize a new index, using a HNSW index on Cosine Similarity
        index = nmslib.init(method='hnsw', space='cosinesimil')

        for prediction in tqdm(predictions, desc="Indexing"):
            index.addDataPoint(id=prediction["idx"], data=prediction["code_rpr"])

        index.createIndex(index_time_params)
        return index

    def retrieve(self, index, predictions, k):
        # retrieve
        ranking = {}
        for prediction in tqdm(predictions, desc="Searching"):
            target_idx = prediction["idx"]
            ids, distances = index.knnQuery(prediction["desc_rpr"], k=k)
            ids = ids.tolist()
            if target_idx in ids:
                ranking[target_idx] = ids.index(target_idx) + 1
            else:
                ranking[target_idx] = 1e9
        return ranking

    def get_ranking(self, k, fold):

        # load prediction
        predictions = self.load_predictions(fold)

        # index data
        index = self.init_index(predictions)

        # retrieve
        return self.retrieve(index, predictions, k=k)

    def perform_eval(self):

        stats = pd.DataFrame(columns=["fold"])
        rankings = {}
        thresholds = [1, 5, 10]

        for fold in self.params.data.folds:
            ranking = self.get_ranking(k=thresholds[-1], fold=fold)

            for k in thresholds:
                stats.at[fold, f"MRR@{k}"] = self.mrr_at_k(ranking.values(), k, len(ranking))
                stats.at[fold, f"RCL@{k}"] = self.recall_at_k(ranking.values(), k, len(ranking))

            rankings[fold]=ranking

        # update fold colum
        stats["fold"] = stats.index

        self.checkpoint_stats(stats)
        self.checkpoint_ranking(rankings)


