from pytorch_lightning import LightningModule
from transformers import BertModel


class BertExplainer(LightningModule):
    """This encoder is used to export Bert attention after be trained."""

    def __init__(self, hparams):
        super(BertExplainer, self).__init__()
        self.bert_encoder = BertModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )

    def forward(self, features):

        attention_mask = (features > 0).int()
        return self.bert_encoder(features, attention_mask).attentions


