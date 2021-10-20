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

from source.DataModule.CoEncoderDataModule import CoEncoderDataModule
from source.callback.PredictionWriter import PredictionWriter
from source.helper.EvalHelper import EvalHelper
from source.helper.ExpHelper import get_sample
from source.model.CoEncoderModel import CoEncoderModel


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
        print(f"Fitting {params.model.name} over {params.data.name} (fold {fold}) with fowling params\n"
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
            model=CoEncoderModel(params.model),
            datamodule=CoEncoderDataModule(
                params.data,
                get_tokenizer(params.model.desc_tokenizer),
                get_tokenizer(params.model.code_tokenizer),
                fold=fold)
        )



def predict(params):
    for fold in params.data.folds:
        # data
        dm = CoEncoderDataModule(
                params.data,
                get_tokenizer(params.model.desc_tokenizer),
                get_tokenizer(params.model.code_tokenizer),
                fold=fold)

        # model
        model = CoEncoderModel.load_from_checkpoint(
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


def explain(hparams):
    print("using the following parameters:\n", OmegaConf.to_yaml(hparams))
    # override some of the params with new values
    model = CoEncoderModel.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt",
        **hparams.model
    )

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    x1_length = hparams.data.desc_max_length
    x2_length = hparams.data.code_max_length

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


def sim(hparams):
    print("using the following parameters:\n", OmegaConf.to_yaml(hparams))
    # override some of the params with new values
    model = CoEncoderModel.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt",
        **hparams.model
    )

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    x1_length = hparams.data.desc_max_length
    x2_length = hparams.data.code_max_length

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

    r1, r2 = model(torch.tensor([x1]), torch.tensor([x2]))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    print(cos(r1, r2))




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
    if "explain" in params.tasks:
        explain(params)
    if "sim" in params.tasks:
        sim(params)




if __name__ == '__main__':
    perform_tasks()
