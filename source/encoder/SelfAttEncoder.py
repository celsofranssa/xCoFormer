import importlib

import torch.nn
from pytorch_lightning import LightningModule
from torch import nn

from source.encoder.EncoderOutput import EncoderOutput
from source.pooling.AveragePooling import AveragePooling


class SelfAttEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, vocabulary_size, representation_size, num_heads, dropout, pooling):
        super(SelfAttEncoder, self).__init__()

        self.value_layer = nn.Embedding(
            num_embeddings=vocabulary_size,
            embedding_dim=representation_size
        )

        self.query_layer = nn.Linear(representation_size, representation_size)
        self.key_layer = nn.Linear(representation_size, representation_size)

        self.multihead_attn = torch.nn.MultiheadAttention(representation_size, num_heads,
                                                          dropout=dropout)

        self.pooling = pooling

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
