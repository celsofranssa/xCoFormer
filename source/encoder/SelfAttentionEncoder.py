import torch.nn
from pytorch_lightning import LightningModule
from torch import nn

from source.pooling.AveragePooling import AveragePooling


class SelfAttentionEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(SelfAttentionEncoder, self).__init__()

        self.value_layer = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        self.query_layer = nn.Linear(hparams.representation_size, hparams.representation_size)
        self.key_layer = nn.Linear(hparams.representation_size, hparams.representation_size)

        self.multihead_attn = torch.nn.MultiheadAttention(hparams.representation_size, hparams.num_heads,
                                                          dropout=hparams.dropout)

        self.pool = AveragePooling()

    def forward(self, x):
        attention_mask = (x > 0).int()

        value = torch.transpose(self.value_layer(x), 0, 1)
        query = self.query_layer(value)
        key = self.key_layer(value)

        attn_output, _ = self.multihead_attn(query, key, value)
        attn_output = torch.transpose(attn_output, 0, 1)

        return self.pool(attention_mask, attn_output)
