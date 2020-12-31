from pytorch_lightning import LightningModule
from transformers import RobertaModel

from source.model.AveragePooling import AveragePooling


class CodeBertaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self):
        super(CodeBertaEncoder, self).__init__()
        self.codeberta_encoder = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
        self.pooling = AveragePooling()

    def forward(self, features):
        # print(features)
        attention_mask = (features > 2).int()
        hidden_states = self.codeberta_encoder(features, attention_mask)[0]

        return self.pooling(
            attention_mask,
            hidden_states
        )
