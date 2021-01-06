import torch
from pytorch_lightning import LightningModule


class AveragePooling(LightningModule):
    """
    Performs average pooling on the last hidden-states transformer output.
    """

    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, attention_mask, hidden_states):
        """
        :param attention_mask:
        :param hidden_states:
        :return:
        """
        attention_mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_hidden_states = torch.sum(hidden_states * attention_mask, 1)
        sum_attention_mask = torch.sum(attention_mask, 1)
        temp = torch.div(
            sum_hidden_states,
            torch.clamp(sum_attention_mask, min=1e-9)
        )
        #print(f'avg--------------------------\n{temp}')
        #print(f'hidden_states: {hidden_states}')
        #temp = torch.nn.MaxPool2d(hidden_states)
        #print(f'max----------------- {temp}')
        
        #docs = []
        #for index_doc in range(len(hidden_states)): #
        #    docs.append( hidden_states[index_doc].amax(dim=0).cpu().detach().numpy() ) #max of column, vector tokens
        #temp = torch.Tensor(docs).to('cuda:0')
        #print(f'temp -----------{temp}')
        #print(f'max--------------------------\n{ torch.amax( temp, 1) }')
        return temp
