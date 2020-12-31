import json
from statistics import mean

import hydra
import nmslib
import numpy as np
import os
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer

from source.DataModule.CodeDescDataModule import CodeDescDataModule
from source.model.JointEncoder import JointEncoder


def get_logger(hparams):
    return pl_loggers.TensorBoardLogger(
        save_dir=hparams.log.dir,
        name=f"{hparams.model.name}_{hparams.data.name}_exp"
    )


def get_model_checkpoint(hparams):
    return ModelCheckpoint(
        monitor="val_mrr",
        dirpath=hparams.model_checkpoint.dir,
        filename=hparams.model.name + "_" + hparams.data.name + "_{epoch:02d}_{val_mrr:.2f}",
        save_top_k=1,
        mode="max"
    )


def get_early_stopping_callback(hparams):
    return EarlyStopping(
        monitor='val_mrr',
        patience=3,
        mode='max'
    )


def get_tokenizer(hparams):
    return AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )


def train(trainer, model, datamodule):
    # training
    datamodule.setup('fit')
    trainer.fit(model, datamodule=datamodule)


def test(trainer, model, datamodule):
    # testing
    datamodule.setup('test')
    trainer.test(datamodule=datamodule)


def fit(hparams):
    print(OmegaConf.to_yaml(hparams))
    #logger
    tb_logger = get_logger(hparams)

    # checkpoint callback
    checkpoint_callback = get_model_checkpoint(hparams)

    # early stopping callback
    early_stopping_callback = get_early_stopping_callback(hparams)

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    dm = CodeDescDataModule(hparams.data, x1_tokenizer, x2_tokenizer)

    model = JointEncoder(hparams.model)

    print(OmegaConf.to_yaml(hparams))

    trainer = Trainer(
        fast_dev_run=hparams.trainer.fast_dev_run,
        max_epochs=hparams.trainer.max_epochs,
        gpus=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # training
    dm.setup('fit')
    trainer.fit(model, datamodule=dm)

    # testing
    dm.setup('test')
    trainer.test(datamodule=dm)


def rank(predictions_path):
    predictions = torch.load(predictions_path)

    descs = []
    codes = []

    for prediction in predictions:
        descs.append(prediction["r1"])
        codes.append(prediction["r2"])

    # initialize a new index, using a HNSW index on Cosine Similarity
    index = nmslib.init(method='hnsw', space='cosinesimil')
    index.addDataPointBatch(codes)
    index.createIndex()

    # retrieve
    neighbours = index.knnQueryBatch(descs, k=100, num_threads=4)

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


def eval(hparams):
    print(OmegaConf.to_yaml(hparams))

    thresholds = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    stats = []
    positions = rank(hparams.model.predictions.path)

    for k in thresholds:
        stats.append(
            {
                "k": k,
                "metric": "MRR",
                "value": mrr_at_k(positions, k),
                "model": hparams.model.name,
                "datasets": hparams.data.name
            }
        )
        stats.append(
            {
                "k": k,
                "metric": "SSR",
                "value": ssr_at_k(positions, k, hparams.data.test.num_samples),
                "model": hparams.model.name,
                "datasets": hparams.data.name
            }
        )
    stats_path = hparams.stats.dir + hparams.model.name + "_" + hparams.data.name + ".stats"
    checkpoint_stats(stats, stats_path)


@hydra.main(config_path="configs/", config_name="config.yaml")
def start(hparams):
    os.chdir(hydra.utils.get_original_cwd())

    print(os.getcwd())
    if "fit" in hparams.tasks:
        fit(hparams)
    if "predict" in hparams.tasks:
        test(hparams)
    if "eval" in hparams.tasks:
        eval(hparams)


if __name__ == '__main__':
    start()
