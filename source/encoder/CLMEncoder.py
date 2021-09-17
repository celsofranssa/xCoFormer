import importlib

from pytorch_lightning import LightningModule
from transformers import RobertaModel



class CLMEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, pooling):
        super(CLMEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        self.pooling = pooling

    def forward(self, features):
        attention_mask = (features != 1).int()
        encoder_outputs = self.encoder(features, attention_mask)

        return self.pooling(
            attention_mask,
            encoder_outputs
        )
