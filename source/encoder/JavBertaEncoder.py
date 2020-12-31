from pytorch_lightning import LightningModule
from transformers import RobertaModel

from source.model.AveragePooling import AveragePooling


class JavBertaEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self):
        super(JavBertaEncoder, self).__init__()
        self.javberta_encoder = RobertaModel.from_pretrained(
            "/home/celso/projects/semantic_code_search/resources/model/javBerta/")
        self.pooling = AveragePooling()

    def forward(self, features):
        # print(features)
        attention_mask = (features > 2).int()
        hidden_states = self.codeberta_encoder(features, attention_mask)[0]

        return self.pooling(
            attention_mask,
            hidden_states
        )
