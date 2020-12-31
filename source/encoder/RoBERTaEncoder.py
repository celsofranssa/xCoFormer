from pytorch_lightning import LightningModule
from transformers import RobertaModel

from source.model.AveragePooling import AveragePooling


class RoBERTaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(RoBERTaEncoder, self).__init__()
        self.roberta_encoder = RobertaModel.from_pretrained(hparams.architecture)
        self.pooling = AveragePooling()

    def forward(self, features):
        # print(features)
        attention_mask = (features > 1).int()
        hidden_states = self.roberta_encoder(features, attention_mask)[0]

        return self.pooling(
            attention_mask,
            hidden_states
        )
