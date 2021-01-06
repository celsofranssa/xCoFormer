import importlib

import torch
from omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule

from source.loss.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from source.loss.TripletLoss import TripletLoss
from source.metric.MRRMetric import MRRMetric


class JointEncoder(LightningModule):
    """Encodes the code and desc into an same space of embeddings."""

    def __init__(self, hparams):
        super(JointEncoder, self).__init__()
        self.hparams = hparams
        self.x1_encoder = self.get_encoder(hparams.x1_encoder, hparams.x1_encoder_hparams)
        self.x2_encoder = self.get_encoder(hparams.x2_encoder, hparams.x2_encoder_hparams)
        self.loss_fn = MultipleNegativesRankingLoss()
        self.mrr = MRRMetric()

    def get_encoder(self, encoder, encoder_hparams):
        encoder_module, encoder_class = encoder.rsplit('.', 1)
        encoder_module = importlib.import_module(encoder_module)
        return getattr(encoder_module, encoder_class)(encoder_hparams)

    def get_loss(self, loss, loss_hparams):
        encoder_module, encoder_class = loss.rsplit('.', 1)
        encoder_module = importlib.import_module(encoder_module)
        return getattr(encoder_module, encoder_class)(loss_hparams)

    def forward(self, x1, x2):
        r1 = self.x1_encoder(x1)
        r2 = self.x2_encoder(x2)
        return r1, r2

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.hparams.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True
        )

    def training_step(self, batch, batch_idx):
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
