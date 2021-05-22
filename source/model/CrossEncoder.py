import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule

from source.metric.MRRMetric import MRRMetric


class CrossEncoder(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):
        super(CrossEncoder, self).__init__()
        self.hparams = hparams

        # encoders
        self.encoder = self.get_encoder(hparams.desc_encoder, hparams.desc_encoder_hparams)

        # loss function
        self.loss_fn = self.get_loss(hparams.loss, hparams.loss_hparams)

        # metric
        self.mrr = MRRMetric()

    @staticmethod
    def get_encoder(encoder, encoder_hparams):
        encoder_module, encoder_class = encoder.rsplit('.', 1)
        encoder_module = importlib.import_module(encoder_module)
        return getattr(encoder_module, encoder_class)(encoder_hparams)

    @staticmethod
    def get_loss(loss, loss_hparams):
        loss_module, loss_class = loss.rsplit('.', 1)
        loss_module = importlib.import_module(loss_module)
        return getattr(loss_module, loss_class)(loss_hparams)

    def forward(self, x1, x2):
        r1 = self.encoder(x1)
        r2 = self.encoder(x2)
        return r1, r2

    def configure_optimizers(self):
        return torch.optim.AdamW(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=self.hparams.weight_decay, amsgrad=True)

    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        train_loss = self.loss(desc_repr, code_repr)
        return train_loss

    def validation_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.log("val_mrr", self.mrr(desc_repr, code_repr), prog_bar=True)
        self.log("val_loss", self.loss(desc_repr, code_repr), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log('m_val_mrr', self.mrr.compute())

    def test_step(self, batch, batch_idx):
        idx, desc, code = batch["idx"], batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.write_prediction_dict({
            "idx": idx,
            "desc_repr": desc_repr,
            "code_repr": code_repr
        }, self.hparams.predictions.path)
        self.log('test_mrr', self.mrr(desc_repr, code_repr), prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('m_test_mrr', self.mrr.compute())

    def get_desc_encoder(self):
        return self.desc_encoder

    def get_code_encoder(self):
        return self.desc_encoder

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
