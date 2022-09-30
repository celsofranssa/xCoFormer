import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule
from hydra.utils import instantiate

from source.metric.MRRMetric import MRRMetric


class DescCodeModel(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):

        super(DescCodeModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.desc_encoder = instantiate(hparams.desc_encoder)
        self.code_encoder = instantiate(hparams.code_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric(hparams.metric)


    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        desc_idx, desc, code_idx, code = batch["desc_idx"], batch["desc"], batch["code_idx"], batch["code"]
        desc_rpr, code_rpr = self.desc_encoder(desc), self.code_encoder(code)

        train_loss = self.loss(
            desc_idx,
            desc_rpr,
            code_idx,
            code_rpr
        )

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        desc_idx, desc, code_idx, code = batch["desc_idx"], batch["desc"], batch["code_idx"], batch["code"]
        desc_rpr, code_rpr = self.desc_encoder(desc), self.code_encoder(code)

        self.mrr.update(
            desc_idx,
            desc_rpr,
            code_idx,
            code_rpr
        )

    def validation_epoch_end(self, outs):
        self.log("val_MRR", self.mrr.compute(), prog_bar=True)
        self.mrr.reset()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        idx, desc, code = batch["idx"], batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)

        return {
            "idx": idx,
            "desc_rpr": desc_repr,
            "code_rpr": code_repr
        }

    def test_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.log("test_MRR", self.mrr(desc_repr, code_repr), prog_bar=True)

    def test_epoch_end(self, outs):
        self.mrr.compute()

    def get_desc_encoder(self):
        return self.desc_encoder

    def get_code_encoder(self):
        return self.desc_encoder

    def configure_optimizers(self):
        if self.hparams.tag_training:
            return self._configure_tgt_optimizers()
        else:
            return self._configure_std_optimizers()

    def _configure_tgt_optimizers(self):
        # optimizers
        desc_optimizer = torch.optim.AdamW(self.desc_encoder.parameters(), lr=self.hparams.desc_lr, betas=(0.9, 0.999),
                                           eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        code_optimizer = torch.optim.AdamW(self.code_encoder.parameters(), lr=self.hparams.code_lr,
                                            betas=(0.9, 0.999),
                                            eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)


        return (
            {"optimizer": desc_optimizer, "frequency": self.hparams.desc_frequency_opt},
            {"optimizer": code_optimizer, "frequency": self.hparams.code_frequency_opt},
        )

    def _configure_std_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999),
                                      eps=1e-08, weight_decay=self.hparams.weight_decay, amsgrad=True)

        return (
            {"optimizer": optimizer}
        )

