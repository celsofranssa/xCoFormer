import pickle

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from source.Dataset.ZSDescCodeDataset import ZSDescCodeDataset


class ZSDescCodeDataModule(pl.LightningDataModule):
    """
    CodeSearch Zero-Shot DataModule
    """

    def __init__(self, params, desc_tokenizer, code_tokenizer):
        super(ZSDescCodeDataModule, self).__init__()
        self.params = params
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer

    def prepare_data(self):
        with open(self.params.dir + f"samples.pkl", "rb") as dataset_file:
            self.samples = pickle.load(dataset_file)

    def setup(self, stage=None):
        self.predict_dataset = ZSDescCodeDataset(
            samples=self.samples,
            desc_tokenizer=self.desc_tokenizer,
            code_tokenizer=self.code_tokenizer,
            desc_max_length=self.params.desc_max_length,
            code_max_length=self.params.code_max_length
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers
        )
