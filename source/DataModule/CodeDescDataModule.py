import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.CodeSearchDataset import CodeSearchDataset


class CodeDescDataModule(pl.LightningDataModule):
    """

    """

    def __init__(self, params, desc_tokenizer, code_tokenizer):
        super().__init__()
        self.params = params
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer

    def prepare_data(self):
        pass

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            self.train_dataset = CodeSearchDataset(
                data_path=self.params.dir + "train.jsonl",
                desc_tokenizer=self.desc_tokenizer,
                code_tokenizer=self.code_tokenizer,
                desc_max_length=self.params.desc_max_length,
                code_max_length=self.params.code_max_length
            )

            self.val_dataset = CodeSearchDataset(
                data_path=self.params.dir + "val.jsonl",
                desc_tokenizer=self.desc_tokenizer,
                code_tokenizer=self.code_tokenizer,
                desc_max_length=self.params.desc_max_length,
                code_max_length=self.params.code_max_length
            )

        if stage == 'test' or stage is None:
            self.test_dataset = CodeSearchDataset(
                data_path=self.params.dir + "test.jsonl",
                desc_tokenizer=self.desc_tokenizer,
                code_tokenizer=self.code_tokenizer,
                desc_max_length=self.params.desc_max_length,
                code_max_length=self.params.code_max_length
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.params.batch_size,
            drop_last=True,
            num_workers=self.params.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.params.batch_size,
            drop_last=True,
            num_workers=self.params.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.params.batch_size,
            drop_last=True,
            num_workers=self.params.num_workers
        )
