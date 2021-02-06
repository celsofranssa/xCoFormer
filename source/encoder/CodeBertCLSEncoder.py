from pytorch_lightning import LightningModule
from transformers import RobertaModel


class CodeBertCLSEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, hparams):
        super(CodeBertCLSEncoder, self).__init__()
        self.codebert_encoder = RobertaModel.from_pretrained(
            hparams.architecture,
            output_attentions=hparams.output_attentions
        )

    def forward(self, features):
        attention_mask = (features != 1).int()
        return self.codebert_encoder(features, attention_mask).pooler_output
