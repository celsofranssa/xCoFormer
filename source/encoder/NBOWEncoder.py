import importlib

from pytorch_lightning import LightningModule
from torch import nn

from source.encoder.EncoderOutput import EncoderOutput
from source.pooling.AveragePooling import AveragePooling


class NBOWEncoder(LightningModule):
    """Encodes the input as embeddings/neural bag of words."""

    def __init__(self, hparams):
        super(NBOWEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        self.pooling = self.get_pooling(hparams.pooling, hparams.pooling_hparams)

    @staticmethod
    def get_pooling(pooling, pooling_hparams):
        pooling_module, pooling_class = pooling.rsplit('.', 1)
        pooling_module = importlib.import_module(pooling_module)
        return getattr(pooling_module, pooling_class)(pooling_hparams)

    def forward(self, x):
        attention_mask = (x > 0).int()
        last_hidden_state = self.embedding(x)


        return self.pooling(
            attention_mask,
            EncoderOutput(last_hidden_state, None)
        )
