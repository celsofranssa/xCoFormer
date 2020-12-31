import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.CodeSearchDataset import CodeSearchDataset


class CodeDescDataModule(pl.LightningDataModule):
    """

    """

    def __init__(self, hparams, x1_tokenizer, x2_tokenizer):
        super().__init__()
        self.hparams = hparams
        self.x1_tokenizer = x1_tokenizer
        self.x2_tokenizer = x2_tokenizer

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = CodeSearchDataset(
                path=self.hparams.train.path,
                x1_tokenizer=self.x1_tokenizer,
                x2_tokenizer=self.x2_tokenizer,
                x1_length=self.hparams.x1_length,
                x2_length=self.hparams.x2_length
            )

            self.val_dataset = CodeSearchDataset(
                path=self.hparams.val.path,
                x1_tokenizer=self.x1_tokenizer,
                x2_tokenizer=self.x2_tokenizer,
                x1_length=self.hparams.x1_length,
                x2_length=self.hparams.x2_length
            )

        if stage == 'test' or stage is None:
            self.test_dataset = CodeSearchDataset(
                path=self.hparams.val.path,
                x1_tokenizer=self.x1_tokenizer,
                x2_tokenizer=self.x2_tokenizer,
                x1_length=self.hparams.x1_length,
                x2_length=self.hparams.x2_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            drop_last=True,
            num_workers=self.hparams.num_workers
        )
