from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from transformers import XLNetModel

from source.model.AveragePooling import AveragePooling


class XLNetEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(XLNetEncoder, self).__init__()
        self.bert_encoder = XLNetModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )
        self.pooling = AveragePooling()

    def forward(self, features):
        attention_mask = (features != 5).int()
        hidden_states = self.bert_encoder(features, attention_mask).last_hidden_state

        return self.pooling(
            attention_mask,
            hidden_states
        )
