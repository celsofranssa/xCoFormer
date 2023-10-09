import torch
from pytorch_lightning import LightningModule
from transformers import RobertaModel


class STDEncoder(LightningModule):
    def __init__(self, architecture):
        super(STDEncoder, self).__init__()
        self.encoder = RobertaModel.from_pretrained(
            architecture
        )


    def forward(self, input):
        outputs = self.encoder(input, attention_mask=input.ne(1))[0]
        outputs = (outputs * input.ne(1)[:, :, None]).sum(1) / input.ne(1).sum(-1)[:, None]
        return torch.nn.functional.normalize(outputs, p=2, dim=1)

