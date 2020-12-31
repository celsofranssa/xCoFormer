import torch
from pytorch_lightning import TrainResult, EvalResult
from pytorch_lightning.core.lightning import LightningModule

from source.loss.MultipleNegativesRankingLoss import MultipleNegativesRankingLoss
from source.metric.MRRMetric import MRRMetric


class DualEncoder(LightningModule):
    """Encodes the code and docstring into an same space of embeddings."""

    def __init__(self, config, x1_encoder, x2_encoder):
        super(DualEncoder, self).__init__()
        self.config = config
        self.x1_encoder = x1_encoder
        self.x2_encoder = x2_encoder
        self.loss_fn = MultipleNegativesRankingLoss()
        self.mrr = MRRMetric(name="MRR")

    def forward(self, x1, x2):
        r1 = self.x1_encoder(x1)
        r2 = self.x2_encoder(x2)
        return r1, r2

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=1e-7, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True
        )

    def training_step(self, batch, batch_idx):
        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        train_loss = self.loss_fn(r1, r2)
        train_mrr = self.mrr(r1, r2)

        result = TrainResult(minimize=train_loss)
        result.log('train_loss', train_loss, prog_bar=True)
        result.log('train_mrr', train_mrr, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        val_loss = self.loss_fn(r1, r2)
        val_mrr = self.mrr(r1, r2)

        result = EvalResult(checkpoint_on=val_loss)

        # logging
        result.log('val_loss', val_loss, prog_bar=True)
        result.log('val_mrr', val_mrr, prog_bar=True)

        return result

    def test_step(self, batch, batch_idx):
        x1, x2 = batch["x1"], batch["x2"]
        r1, r2 = self(x1, x2)
        test_loss = self.loss_fn(r1, r2)
        test_mrr = self.mrr(r1, r2)

        result = EvalResult(checkpoint_on=test_loss)

        # logging
        result.log('test_loss', test_loss, prog_bar=True)
        result.log('test_mrr', test_mrr, prog_bar=True)

        # store predictions
        # predictions_to_write = {'r1': r1, 'r2': r2}
        # result.write_dict(predictions_to_write, self.config.predictions.path)

        return result
