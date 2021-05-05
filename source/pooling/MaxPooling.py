import torch
from pytorch_lightning import LightningModule


class MaxPooling(LightningModule):
    """
    Performs max pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, attention_mask, encoder_outputs):
        """
        :param attention_mask:
        :param hidden_states:
        :return:
        """
        hidden_states = encoder_outputs.last_hidden_state
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        hidden_states = 1e9 * (attention_mask - 1) + hidden_states
        #hidden_states[attention_mask == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(hidden_states, 1)[0]


