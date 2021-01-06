import torch
from pytorch_lightning import LightningModule


class MaxPooling(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, attention_mask, hidden_states):
        """
        :param attention_mask:
        :param hidden_states:
        :return:
        """
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * attention_mask, 1)
        sum_attention_mask = torch.sum(attention_mask, 1)
        return sum_hidden_stats.max(2)
        #return torch.div(
        #    sum_hidden_states,
        #    torch.clamp(sum_attention_mask, min=1e-9)
        #)
