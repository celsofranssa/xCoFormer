import importlib

from pytorch_lightning import LightningModule
from transformers import RobertaModel

class RoBERTaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(RoBERTaEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )
        self.pooling = self.get_pooling(hparams.pooling, hparams.pooling_hparams)

    @staticmethod
    def get_pooling(pooling, pooling_hparams):
        pooling_module, pooling_class = pooling.rsplit('.', 1)
        pooling_module = importlib.import_module(pooling_module)
        return getattr(pooling_module, pooling_class)(pooling_hparams)

    def forward(self, features):
        attention_mask = (features != 1).int()
        encoder_outputs = self.encoder(features, attention_mask)

        return self.pooling(
            attention_mask,
            encoder_outputs
        )

