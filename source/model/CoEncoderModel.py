import importlib

import torch
from pytorch_lightning.core.lightning import LightningModule
from hydra.utils import instantiate

from source.metric.MRRMetric import MRRMetric


class CoEncoderModel(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):

        super(CoEncoderModel, self).__init__()
        self.save_hyperparameters(hparams)

        # encoders
        self.desc_encoder = instantiate(hparams.desc_encoder)
        self.code_encoder = instantiate(hparams.code_encoder)

        # loss function
        self.loss = instantiate(hparams.loss)

        # metric
        self.mrr = MRRMetric()


    def forward(self, desc, code):
        desc_repr = self.desc_encoder(desc)
        code_repr = self.code_encoder(code)
        return desc_repr, code_repr

    def training_step(self, batch, batch_idx, optimizer_idx):

        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        train_loss = self.loss(desc_repr, code_repr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss

    def validation_step(self, batch, batch_idx):
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        self.log("val_MRR", self.mrr(desc_repr, code_repr), prog_bar=True)
        self.log("val_LOSS", self.loss(desc_repr, code_repr), prog_bar=True)

    def validation_epoch_end(self, outs):
        self.mrr.compute()

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

    # Alternating schedule for optimizer steps (e.g. GANs)
    def optimizer_step(
            self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure,
            on_tpu=False, using_native_amp=False, using_lbfgs=False,
    ):
        # update generator every step
        if optimizer_idx == 0:
            if batch_idx % 2 == 0:
                optimizer.step(closure=optimizer_closure)

        # update discriminator every 2 steps
        if optimizer_idx == 1:
            if batch_idx % 2 != 0:
                optimizer.step(closure=optimizer_closure)

    def configure_optimizers(self):
        # optimizers
        optimizers = [
            torch.optim.AdamW(self.desc_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=self.hparams.weight_decay, amsgrad=True),

            torch.optim.AdamW(self.code_encoder.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08,
                              weight_decay=self.hparams.weight_decay, amsgrad=True)
        ]

        # schedulers
        step_size_up = round(0.03 * self.num_training_steps)
        schedulers = [
            torch.optim.lr_scheduler.CyclicLR(optimizers[0], mode='triangular2', base_lr=self.hparams.base_lr,
                                              max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                              cycle_momentum=False),
            torch.optim.lr_scheduler.CyclicLR(optimizers[1], mode='triangular2', base_lr=self.hparams.base_lr,
                                              max_lr=self.hparams.max_lr, step_size_up=step_size_up,
                                              cycle_momentum=False)
        ]

        return optimizers, schedulers

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and number of epochs."""
        steps_per_epochs = len(self.train_dataloader()) / self.trainer.accumulate_grad_batches
        max_epochs = self.trainer.max_epochs
        return steps_per_epochs * max_epochs
