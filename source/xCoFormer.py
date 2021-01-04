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
from source.Helper.EvalHelper import EvalHelper
from source.model.JointEncoder import JointEncoder


def get_logger(hparams):
    return pl_loggers.TensorBoardLogger(
        save_dir=hparams.log.dir,
        name=f"{hparams.model.name}_{hparams.data.name}_exp"
    )


def get_model_checkpoint_callback(hparams):
    return ModelCheckpoint(
        monitor="val_mrr",
        dirpath=hparams.model_checkpoint.dir,
        filename=hparams.model.name + "_" + hparams.data.name,
        save_top_k=1,
        save_weights_only=True,
        mode="max"
    )


def get_early_stopping_callback(hparams):
    return EarlyStopping(
        monitor='val_mrr',
        patience=hparams.patience,
        min_delta=hparams.min_delta,
        mode='max'
    )


def get_tokenizer(hparams):
    return AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )


def fit(hparams):
    # print(OmegaConf.to_yaml(hparams, resolve=True))
    print(hparams.model.predictions)
    # logger
    # tb_logger = get_logger(hparams)
    #
    # # checkpoint callback
    # checkpoint_callback = get_model_checkpoint_callback(hparams)
    #
    # # early stopping callback
    # early_stopping_callback = get_early_stopping_callback(hparams.trainer)
    #
    # # tokenizers
    # x1_tokenizer = get_tokenizer(hparams.model)
    # x2_tokenizer = x1_tokenizer
    #
    # dm = CodeDescDataModule(hparams.data, x1_tokenizer, x2_tokenizer)
    #
    # model = JointEncoder(hparams.model)
    #
    # trainer = Trainer(
    #     fast_dev_run=hparams.trainer.fast_dev_run,
    #     max_epochs=hparams.trainer.max_epochs,
    #     gpus=1,
    #     logger=tb_logger,
    #     callbacks=[checkpoint_callback, early_stopping_callback]
    # )
    #
    # # training
    # dm.setup('fit')
    # trainer.fit(model, datamodule=dm)
    #
    # # testing
    # dm.setup('test')
    # trainer.test(datamodule=dm)

def predict(hparams):
    print("Not implemented yet")


def eval(hparams):
    print(OmegaConf.to_yaml(hparams))
    evaluator = EvalHelper(hparams)
    evaluator.perform_eval()


def explain(hparams):
    print("Not implemented yet")



@hydra.main(config_path="configs/", config_name="config.yaml")
def perform_tasks(hparams):
    os.chdir(hydra.utils.get_original_cwd())

    if "fit" in hparams.tasks:
        fit(hparams)
    if "predict" in hparams.tasks:
        predict(hparams)
    if "eval" in hparams.tasks:
        eval(hparams)
    if "explain" in hparams.tasks:
        explain(hparams)


if __name__ == '__main__':
    perform_tasks()
