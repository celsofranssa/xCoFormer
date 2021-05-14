import importlib

import torch.nn
from pytorch_lightning import LightningModule
from torch import nn

from source.encoder.EncoderOutput import EncoderOutput
from source.pooling.AveragePooling import AveragePooling


class SelfAttEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(SelfAttEncoder, self).__init__()

        self.value_layer = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        self.query_layer = nn.Linear(hparams.representation_size, hparams.representation_size)
        self.key_layer = nn.Linear(hparams.representation_size, hparams.representation_size)

        self.multihead_attn = torch.nn.MultiheadAttention(hparams.representation_size, hparams.num_heads,
                                                          dropout=hparams.dropout)

        self.pooling = self.get_pooling(hparams.pooling, hparams.pooling_hparams)

    @staticmethod
    def get_pooling(pooling, pooling_hparams):
        pooling_module, pooling_class = pooling.rsplit('.', 1)
        pooling_module = importlib.import_module(pooling_module)
        return getattr(pooling_module, pooling_class)(pooling_hparams)

    def forward(self, x):
        attention_mask = (x > 0).int()

        value = torch.transpose(self.value_layer(x), 0, 1)
        query = self.query_layer(value)
        key = self.key_layer(value)

        last_hidden_state, pooler_output = self.multihead_attn(query, key, value)
        last_hidden_state = torch.transpose(last_hidden_state, 0, 1)


        return self.pooling(
            attention_mask,
            EncoderOutput(last_hidden_state, pooler_output))
