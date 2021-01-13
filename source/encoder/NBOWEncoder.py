from pytorch_lightning import LightningModule
from torch import nn

from source.model.AveragePooling import AveragePooling


class NBOWEncoder(LightningModule):
    """Encodes the input as embeddings/neural bag of words."""

    def __init__(self, hparams):
        super(NBOWEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        self.pool = AveragePooling()

    def forward(self, x):
        attention_mask = (x > 0).int()
        outputs = self.embedding(x)
        return self.pool(attention_mask, outputs)