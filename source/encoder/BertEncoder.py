from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from transformers import BertModel

from source.model.AveragePooling import AveragePooling


class BertEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(BertEncoder, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )
        self.pooling = AveragePooling()

    def forward(self, features):
        attention_mask = (features > 0).int()
        return self.bert_encoder(features, attention_mask).pooler_output
