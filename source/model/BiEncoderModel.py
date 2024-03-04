import torch
from hydra.utils import instantiate
from pytorch_lightning import LightningModule
from transformers import get_linear_schedule_with_warmup

from source.metric.MRRMetric import MRRMetric


class BiEncoderModel(LightningModule):

    def __init__(self, hparams):

        super(BiEncoderModel, self).__init__()
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
        desc, code = batch["desc"], batch["code"]
        desc_repr, code_repr = self(desc, code)
        train_loss = self.loss(desc_repr, code_repr)

        # log training loss
        self.log('train_LOSS', train_loss)

        return train_loss
    #
    # def validation_step(self, batch, batch_idx):
    #     desc_repr, code_repr = self(batch["desc"], batch["code"])
    #     self.log_dict(self.mrr(batch["desc_idx"], desc_repr, batch["code_idx"], code_repr), prog_bar=True)
    #
    # def on_validation_epoch_end(self):
    #     self.mrr.compute()

    def validation_step(self, batch, batch_idx):
        desc_repr, code_repr = self(batch["desc"], batch["code"])
        self.mrr(batch["desc_idx"], desc_repr, batch["code_idx"], code_repr)

    def on_validation_epoch_end(self):
        self.log_dict(self.mrr.compute(), prog_bar=True)
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
            return self._configure_ctg_optimizers()
        else:
            return self._configure_std_optimizers()

    def _configure_ctg_optimizers(self):
        # optimizers
        desc_optimizer = torch.optim.AdamW(self.desc_encoder.parameters(), lr=self.hparams.lr, eps=1e-8)
        code_optimizer = torch.optim.AdamW(self.code_encoder.parameters(), lr=self.hparams.lr, eps=1e-8)

        # schedulers
        desc_scheduler = get_linear_schedule_with_warmup(desc_optimizer, num_warmup_steps=0,
                                                         num_training_steps=self.trainer.estimated_stepping_batches)
        code_scheduler = get_linear_schedule_with_warmup(code_optimizer, num_warmup_steps=0,
                                                         num_training_steps=self.trainer.estimated_stepping_batches)

        return (
            {"optimizer": desc_optimizer, "lr_scheduler": desc_scheduler, "frequency": self.hparams.desc_frequency_opt},
            {"optimizer": code_optimizer, "lr_scheduler": code_scheduler, "frequency": self.hparams.code_frequency_opt},
        )

    def _configure_std_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                    num_training_steps=self.trainer.estimated_stepping_batches)

        return (
            {"optimizer": optimizer,
             "lr_scheduler": {"scheduler": scheduler, "interval": "step", "name": "SCHDLR"}},
        )