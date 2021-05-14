import torch.nn
from pytorch_lightning import LightningModule
from torch import nn


class CNNEncoder(LightningModule):
    """Represents a text as a dense vector."""

    def __init__(self, hparams):
        super(CNNEncoder, self).__init__()

        # embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=hparams.vocabulary_size,
            embedding_dim=hparams.representation_size
        )

        kernel_sizes = hparams.kernel_sizes

        # convolutional layers
        self.convs = nn.ModuleList([
            self.get_conv_layer(hparams.representation_size, hparams.out_channels, kernel_size, hparams.max_length)
            for kernel_size in kernel_sizes])

        self.linear = nn.Linear(3 * 4000, hparams.representation_size)

    def get_conv_layer(self, representation_size, out_channels, kernel_size, sentence_length):
        """
        Defines a convolutional block.
        """
        return nn.Sequential(
            nn.Conv1d(in_channels=representation_size, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool1d(sentence_length - kernel_size + 1, stride=1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        r1 = self.embedding(x)
        r1 = torch.transpose(r1, 2, 1)

        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(r1))

        # concatenates the outputs from each convolutional layer
        cat = torch.cat(conv_outputs, 1)

        # flatten
        flatten_cat = torch.flatten(cat, start_dim=1)

        return self.linear(flatten_cat)
