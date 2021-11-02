import json
from statistics import mean

import hydra
import os
import torch
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from transformers import AutoTokenizer

from source.DataModule.BiEncoderDataModule import BiEncoderDataModule
from source.callback.PredictionWriter import PredictionWriter
from source.helper.EvalHelper import EvalHelper
from source.model.BiEncoderModel import BiEncoderModel
from source.model.SiEncoderModel import SiEncoderModel


def get_logger(params, fold):
    return loggers.TensorBoardLogger(
        save_dir=params.log.dir,
        name=f"{params.model.name}_{params.data.name}_{fold}_exp"
    )


def get_model_checkpoint_callback(params, fold):
    return ModelCheckpoint(
        monitor="val_MRR",
        dirpath=params.model_checkpoint.dir,
        filename=f"{params.model.name}_{params.data.name}_{fold}",
        save_top_k=1,
        save_weights_only=True,
        mode="max"
    )


def get_early_stopping_callback(params):
    return EarlyStopping(
        monitor='val_MRR',
        patience=params.trainer.patience,
        min_delta=params.trainer.min_delta,
        mode='max'
    )


def get_tokenizer(params):
    tokenizer = AutoTokenizer.from_pretrained(
        params.architecture
    )
    if params.architecture == "gpt2":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return tokenizer


def fit(params):
    for fold in params.data.folds:
        print(f"Fitting {params.model.name} over {params.data.name} (fold {fold}) with fowling self.params\n"
              f"{OmegaConf.to_yaml(params)}\n")
        # Initialize a trainer
        trainer = pl.Trainer(
            fast_dev_run=params.trainer.fast_dev_run,
            max_epochs=params.trainer.max_epochs,
            precision=params.trainer.precision,
            gpus=params.trainer.gpus,
            progress_bar_refresh_rate=params.trainer.progress_bar_refresh_rate,
            logger=get_logger(params, fold),
            callbacks=[
                get_model_checkpoint_callback(params, fold),  # checkpoint_callback
                get_early_stopping_callback(params),  # early_stopping_callback
            ]
        )
        # Train the ⚡ model
        trainer.fit(
            model=SiEncoderModel(params.model),
            datamodule=BiEncoderDataModule(
                params.data,
                get_tokenizer(params.model.desc_tokenizer),
                get_tokenizer(params.model.code_tokenizer),
                fold=fold)
        )


def predict(params):
    for fold in params.data.folds:
        # data
        dm = BiEncoderDataModule(
            params.data,
            get_tokenizer(params.model.desc_tokenizer),
            get_tokenizer(params.model.code_tokenizer),
            fold=fold)

        # model
        model = SiEncoderModel.load_from_checkpoint(
            checkpoint_path=f"{params.model_checkpoint.dir}{params.model.name}_{params.data.name}_{fold}.ckpt"
        )

        params.prediction.fold = fold
        # trainer
        trainer = pl.Trainer(
            gpus=params.trainer.gpus,
            callbacks=[PredictionWriter(params.prediction)]
        )

        # predicting
        dm.prepare_data()
        dm.setup("predict")

        print(f"Predicting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
              f"{OmegaConf.to_yaml(params)}\n")
        trainer.predict(
            model=model,
            datamodule=dm,

        )


def eval(params):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(params))
    evaluator = EvalHelper(params)
    evaluator.perform_eval()


@hydra.main(config_path="settings", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "fit" in params.tasks:
        fit(params)
    if "predict" in params.tasks:
        predict(params)
    if "eval" in params.tasks:
        eval(params)


if __name__ == '__main__':
    perform_tasks()
