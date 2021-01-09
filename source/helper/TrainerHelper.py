from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from transformers import AutoTokenizer


class TrainerHelper:

    def get_logger(self, hparams):
        return pl_loggers.TensorBoardLogger(
            save_dir=hparams.log.dir,
            name=f"{hparams.model.name}_{hparams.data.name}_exp"
        )

    def get_model_checkpoint_callback(self, hparams):
        return ModelCheckpoint(
            monitor="val_mrr",
            dirpath=hparams.model_checkpoint.dir,
            filename=hparams.model.name + "_" + hparams.data.name,
            save_top_k=1,
            save_weights_only=True,
            mode="max"
        )

    def get_early_stopping_callback(self, hparams):
        return EarlyStopping(
            monitor='val_mrr',
            patience=hparams.patience,
            min_delta=hparams.min_delta,
            mode='max'
        )

    def get_tokenizer(self, hparams):
        return AutoTokenizer.from_pretrained(
            hparams.tokenizer.architecture
        )
