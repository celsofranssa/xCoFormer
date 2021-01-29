from pytorch_lightning import LightningModule
from transformers import RobertaModel


class RoBERTaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(RoBERTaEncoder, self).__init__()
        self.roberta_encoder = RobertaModel.from_pretrained(hparams.architecture)

    def forward(self, features):
        attention_mask = (features > 1).int()
        return self.roberta_encoder(features, attention_mask).pooler_output
