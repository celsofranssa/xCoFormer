import torch
from pytorch_lightning import LightningModule
from transformers import BertModel


class BERTEncoder(LightningModule):
    """Encodes the input as embeddings."""

    def __init__(self, architecture):
        super(BERTEncoder, self).__init__()
        self.encoder = BertModel.from_pretrained(
            architecture
        )

    def forward(self, input):
        outputs = self.encoder(input, attention_mask=input.gt(0))[0]
        outputs = (outputs * input.gt(0)[:, :, None]).sum(1) / input.gt(0).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)
