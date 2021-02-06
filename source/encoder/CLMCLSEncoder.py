from pytorch_lightning import LightningModule
from transformers import RobertaModel


class CLMCLSEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(CLMCLSEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )

    def forward(self, features):
        attention_mask = (features != 1).int()
        return self.encoder(features, attention_mask).pooler_output
