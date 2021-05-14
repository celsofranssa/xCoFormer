import importlib

import torch.nn
from pytorch_lightning import LightningModule
from torch import nn

from source.encoder.EncoderOutput import EncoderOutput
from source.pooling.AveragePooling import AveragePooling


class GRUEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(GRUEncoder, self).__init__()

        self.embedding = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        self.rnn = torch.nn.GRU(
            input_size=hparams.representation_size,
            hidden_size=hparams.hidden_size,
            batch_first=True,
            bidirectional=True)

        self.pooling = self.get_pooling(hparams.pooling, hparams.pooling_hparams)

    @staticmethod
    def get_pooling(pooling, pooling_hparams):
        pooling_module, pooling_class = pooling.rsplit('.', 1)
        pooling_module = importlib.import_module(pooling_module)
        return getattr(pooling_module, pooling_class)(pooling_hparams)

    def forward(self, x):
        attention_mask = (x > 0).int()
        emb_outs = self.embedding(x)
        last_hidden_state, pooler_output = self.rnn(emb_outs)

        return self.pooling(
            attention_mask,
            EncoderOutput(last_hidden_state, pooler_output)
        )
