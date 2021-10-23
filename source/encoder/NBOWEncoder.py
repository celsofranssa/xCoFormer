import importlib

from pytorch_lightning import LightningModule
from torch import nn

from source.encoder.EncoderOutput import EncoderOutput
from source.pooling.AveragePooling import AveragePooling


class NBOWEncoder(LightningModule):
    """Encodes the input as embeddings/neural bag of words."""

    def __init__(self, vocabulary_size, representation_size, pooling):
        super(NBOWEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=representation_size
        )

        self.pooling = pooling


    def forward(self, x):
        attention_mask = (x > 0).int()
        last_hidden_state = self.embedding(x)


        return self.pooling(
            attention_mask,
            EncoderOutput(last_hidden_state, None)
        )
