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
from source.helper.EvalHelper import EvalHelper
from source.helper.ExpHelper import get_sample
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
    tokenizer = AutoTokenizer.from_pretrained(
        hparams.tokenizer.architecture
    )
    if hparams.tokenizer.architecture ==


def fit(hparams):
    print("Fitting with the following parameters:\n", OmegaConf.to_yaml(hparams))

    # logger
    tb_logger = get_logger(hparams)

    # checkpoint callback
    checkpoint_callback = get_model_checkpoint_callback(hparams)

    # early stopping callback
    early_stopping_callback = get_early_stopping_callback(hparams.trainer)

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    dm = CodeDescDataModule(hparams.data, x1_tokenizer, x2_tokenizer)

    model = JointEncoder(hparams.model)

    trainer = Trainer(
        fast_dev_run=hparams.trainer.fast_dev_run,
        max_epochs=hparams.trainer.max_epochs,
        gpus=1,
        enable_pl_optimizer=True,
        logger=tb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )

    # training
    dm.setup('fit')
    trainer.fit(model, datamodule=dm)

    # testing
    dm.setup('test')
    trainer.test(model, datamodule=dm)


def predict(hparams):
    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    # data
    dm = CodeDescDataModule(hparams.data, x1_tokenizer, x2_tokenizer)

    # model
    model = JointEncoder.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt"
    )

    # trainer
    trainer = Trainer(
        fast_dev_run=hparams.trainer.fast_dev_run,
        max_epochs=hparams.trainer.max_epochs,
        enable_pl_optimizer=True,
        gpus=1
    )

    # testing
    dm.setup('test')
    trainer.test(model=model, datamodule=dm)


def eval(hparams):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(hparams))
    evaluator = EvalHelper(hparams)
    evaluator.perform_eval()


def explain(hparams):
    print("using the following parameters:\n", OmegaConf.to_yaml(hparams))
    # override some of the params with new values
    model = JointEncoder.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt",
        **hparams.model
    )

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    x1_length = hparams.data.x1_length
    x2_length = hparams.data.x2_length

    desc, code = get_sample(
        hparams.attentions.sample_id,
        hparams.attentions.dir + hparams.data.name + "_samples.jsonl"
    )

    x1 = x1_tokenizer.encode(text=desc, max_length=x1_length, padding="max_length",
                             truncation=True)
    x2 = x2_tokenizer.encode(text=code, max_length=x2_length, padding="max_length",
                             truncation=True)

    # predict
    model.eval()

    r1_attentions, r2_attentions = model(torch.tensor([x1]), torch.tensor([x2]))

    attentions = {
        "x1": desc,
        "x1_tokens": x1_tokenizer.convert_ids_to_tokens(x1),
        "x2": code,
        "x2_tokens": x2_tokenizer.convert_ids_to_tokens(x2),
        "r1_attentions": r1_attentions,
        "r2_attentions": r2_attentions
    }
    torch.save(obj=attentions,
               f=hparams.attentions.dir +
                 hparams.model.name +
                 "_" +
                 hparams.data.name +
                 "_attentions.pt")


@hydra.main(config_path="configs/", config_name="config.yaml")
def perform_tasks(hparams):
    os.chdir(hydra.utils.get_original_cwd())
    hparams = update_hparams(hparams)

    if "fit" in hparams.tasks:
        fit(hparams)
    if "predict" in hparams.tasks:
        predict(hparams)
    if "eval" in hparams.tasks:
        eval(hparams)
    if "explain" in hparams.tasks:
        explain(hparams)


def update_hparams(hparams):
    # update predictions
    hparams.model.predictions.path = f"../resources/predictions/bert_{hparams.data.name}_predictions.pt"

    # update steps
    # num_steps = hparams.data.train.num_samples / hparams.data.batch_size
    # num_epochs = hparams.trainer.max_epochs
    # hparams.model.steps = num_epochs * num_steps
    return hparams


if __name__ == '__main__':
    perform_tasks()
