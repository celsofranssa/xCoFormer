
from pytorch_lightning import LightningModule
from transformers import FNetModel



class FNetEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture, output_attentions, pooling):
        super(FNetEncoder, self).__init__()
        self.encoder = FNetModel.from_pretrained(
            architecture,
            output_attentions=output_attentions
        )
        self.pooling = pooling

    def forward(self, features):
        attention_mask = (features != 3).int()
        encoder_outputs = self.encoder(features)

        return self.pooling(
            attention_mask,
            encoder_outputs
        )
