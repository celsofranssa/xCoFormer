import json
from statistics import mean

import nmslib
import torch
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


def rank(data_dir):
    train_path = data_dir + "train.jsonl"
    codes = []
    descs = []
    with open(train_path, "r") as train_file:
        for line in train_file:
            sample = json.loads(line)
            codes.append(sample["code"])
            descs.append(sample["desc"])

    # train vectorizers
    code_vectorizer = TfidfVectorizer(max_features=50000)
    desc_vectorizer = TfidfVectorizer(max_features=50000)
    code_vectorizer.fit_transform(codes)
    desc_vectorizer.fit_transform(descs)

    # test codes and descs
    test_path = data_dir + "test.jsonl"
    codes = []
    descs = []
    with open(test_path, "r") as test_file:
        for line in test_file:
            sample = json.loads(line)
            codes.append(code_vectorizer.transform([sample["code"]]).toarray()[0])
            descs.append(desc_vectorizer.transform([sample["desc"]]).toarray()[0])

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(codes)
    index.createIndex()

    # retrieve
    neighbours = index.knnQueryBatch(descs, k=100, num_threads=64)

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


def mrr_at_k(positions, k):
    # positions_at_k = [p for p in positions if p <= k]
    positions_at_k = [p if p <= k else 0 for p in positions]
    rrank = 0.0
    for pos in positions_at_k:
        if pos != 0:
            rrank += 1.0 / pos
    return rrank / len(positions_at_k)


def ssr_at_k(positions, k, num_samples):
    return 1.0 * sum(i <= k for i in positions) / num_samples


def checkpoint_stats(stats, stats_path):
    with open(stats_path, "w") as stats_file:
        for data in stats:
            stats_file.write(f"{json.dumps(data)}\n")


def eval(data, num_samples, model):
    thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    stats = []
    data_dir = "../resources/datasets/" + data + "/"
    positions = rank(data_dir)

    for k in thresholds:
        stats.append(
            {
                "k": k,
                "metric": "MRR",
                "value": mrr_at_k(positions, k),
                "model": model,
                "datasets": data
            }
        )
        stats.append(
            {
                "k": k,
                "metric": "SSR",
                "value": ssr_at_k(positions, k, num_samples),
                "model": model,
                "datasets": data
            }
        )
    stats_path = "../resources/stats/" + model + "_" + data + ".stats"
    checkpoint_stats(stats, stats_path)


if __name__ == '__main__':
    data="java_v01"
    model="TF-IDF"
    num_samples = 71892
    # data = "python_v01"
    # model = "TF-IDF"
    # num_samples = 1411
    eval(data, num_samples, model)
