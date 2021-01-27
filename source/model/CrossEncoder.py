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
        self.encoder = self.get_encoder(hparams.x1_encoder, hparams.x1_encoder_hparams)

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
        # optimizers
        optimizers = [
            torch.optim.Adam(self.encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=0, amsgrad=True)
        ]

        # schedulers
        step_size_up = 0.03 * self.num_training_steps

        schedulers = [
            torch.optim.lr_scheduler.CyclicLR(optimizers[0], mode='triangular2', base_lr=1e-7, max_lr=1e-3,
                                              step_size_up=step_size_up, cycle_momentum=False)
        ]
        return optimizers, schedulers

    def training_step(self, batch, batch_idx, optimizer_idx=0):

        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        train_loss = self.loss_fn(r1, r2)
        return train_loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        self.log("val_mrr", self.mrr(r1, r2), prog_bar=True)
        self.log("val_loss", self.loss_fn(r1, r2), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log('m_val_mrr', self.mrr.compute())

    def test_step(self, batch, batch_idx):
        id, x1, x2 = batch["id"], batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        self.write_prediction_dict({
            "id": id,
            "r1": r1,
            "r2": r2
        }, self.hparams.predictions.path)
        self.log('test_mrr', self.mrr(r1, r2), prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('m_test_mrr', self.mrr.compute())

    
    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
