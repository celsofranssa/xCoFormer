import importlib

from pytorch_lightning import LightningModule
from transformers import GPTNeoModel


class GPTNeoEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, num_embeddings, pad_id, pooling):
        super(GPTNeoEncoder, self).__init__()
        self.encoder = GPTNeoModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        self.encoder.resize_token_embeddings(num_embeddings)
        self.pad_id=pad_id
        self.pooling = pooling

    def forward(self, features):
        attention_mask = (features != self.pad_id).int()
        encoder_outputs = self.encoder(features)

        return self.pooling(
            attention_mask,
            encoder_outputs
        )
