import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule

from source.metric.MRRMetric import MRRMetric


class CoEncoder(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):

        super(CoEncoder, self).__init__()
        self.hparams = hparams

        # encoders
        self.desc_encoder = self.get_encoder(hparams.desc_encoder, hparams.desc_encoder_hparams)
        self.code_encoder = self.get_encoder(hparams.code_encoder, hparams.code_encoder_hparams)

        # loss function
        self.loss = self.get_loss(hparams.loss, hparams.loss_hparams)

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

    def configure_optimizers(self):

        # optimizers
        optimizers = [
            torch.optim.AdamW(self.desc_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=self.hparams.weight_decay, amsgrad=True),
            torch.optim.AdamW(self.code_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                             weight_decay=self.hparams.weight_decay, amsgrad=True)
        ]

        # schedulers
        step_size_up = 0.03 * self.num_training_steps

        schedulers = [
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[0],
                mode='triangular2',
                base_lr=self.hparams.base_lr,
                max_lr=self.hparams.max_lr,
                step_size_up=step_size_up,
                cycle_momentum=False),
            torch.optim.lr_scheduler.CyclicLR(
                optimizers[1],
                mode='triangular2',
                base_lr=self.hparams.base_lr,
                max_lr=self.hparams.max_lr,
                step_size_up=step_size_up,
                cycle_momentum=False)
        ]
        return optimizers, schedulers

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
            on_tpu, using_native_amp, using_lbfgs
    ):
        # update generator every step
        if optimizer_idx == 0:
            if batch_idx % 2 == 0:
                optimizer.step(closure=optimizer_closure)

        # update discriminator every 2 steps
        if optimizer_idx == 1:
            if batch_idx % 2 != 0:
                optimizer.step(closure=optimizer_closure)


    def forward(self, desc, code):
        r1 = self.desc_encoder(desc)
        r2 = self.code_encoder(code)
        return r1, r2

    def training_step(self, batch, batch_idx, optimizer_idx):
        desc, code = batch["desc"], batch["code"]
        r1, r2 = self(desc, code)
        train_loss = self.loss(r1, r2)
        return train_loss

    def validation_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        r1, r2 = self(desc, code)
        self.log("val_mrr", self.mrr(r1, r2), prog_bar=True)
        self.log("val_loss", self.loss(r1, r2), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.log('m_val_mrr', self.mrr.compute())

    def test_step(self, batch, batch_idx):
        idx, desc, code = batch["idx"], batch["desc"], batch["code"]
        r1, r2 = self(desc, code)
        self.write_prediction_dict({
            "idx": idx,
            "r1": r1,
            "r2": r2
        }, self.hparams.predictions.path)
        self.log('test_mrr', self.mrr(r1, r2), prog_bar=True)

    def test_epoch_end(self, outs):
        self.log('m_test_mrr', self.mrr.compute())

    def get_x1_encoder(self):
        return self.desc_encoder

    def get_x2_encoder(self):
        return self.desc_encoder

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
