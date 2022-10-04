import json
import pickle

import torch
from torch.utils.data import Dataset


class ZSDescCodeDataset(Dataset):
    """CodeSearch Dataset.
    """

    def __init__(self, samples, desc_tokenizer, code_tokenizer, desc_max_length, code_max_length):
        super(ZSDescCodeDataset, self).__init__()
        self.samples = samples
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer
        self.desc_max_length = desc_max_length
        self.code_max_length = code_max_length

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "desc_idx": sample["desc_idx"],
            "desc": torch.tensor(
                self.desc_tokenizer.encode(text=sample["desc"], max_length=self.desc_max_length, padding="max_length",
                                           truncation=True)
            ),
            "code_idx": sample["code_idx"],
            "code": torch.tensor(
                self.code_tokenizer.encode(text=sample["code"], max_length=self.code_max_length, padding="max_length",
                                           truncation=True)
            )
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(
            self.samples[idx]
        )
