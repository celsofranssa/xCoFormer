from pytorch_lightning import LightningModule

from source.model.AveragePooling import AveragePooling


class Encoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, encoder):
        super(Encoder, self).__init__()
        self.encoder = encoder
        self.pooling = AveragePooling()

    def forward(self, features):
        # print(features)
        attention_mask = (features > 0).int()
        hidden_states = self.encoder(features, attention_mask)[0]

        return self.pooling(
            attention_mask,
            hidden_states
        )
