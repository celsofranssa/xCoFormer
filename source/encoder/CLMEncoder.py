from pytorch_lightning import LightningModule
from transformers import RobertaModel


class CLMEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(CLMEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )

    def forward(self, features):
        attention_mask = (features != 1).int()
        hidden_states = self.encoder(features, attention_mask).last_hidden_state

        return self.pooling(
            attention_mask,
            hidden_states
        )
