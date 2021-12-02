import json
import pickle

import torch
from torch.utils.data import Dataset


class BiEncoderDataset(Dataset):
    """CodeSearch Dataset.
    """

    def __init__(self, samples, ids_path, desc_tokenizer, code_tokenizer, desc_max_length, code_max_length):
        super(BiEncoderDataset, self).__init__()
        self.samples = samples
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer
        self.desc_max_length = desc_max_length
        self.code_max_length = code_max_length
        self._load_ids(ids_path)

    def _load_ids(self, ids_path):
        with open(ids_path, "rb") as ids_file:
            self.ids = pickle.load(ids_file)

    def _encode(self, sample):
        return {
            "idx": sample["idx"],
            "desc": torch.tensor(
                self.desc_tokenizer.encode(text=sample["desc"], max_length=self.desc_max_length, padding="max_length",
                                           truncation=True)
            ),
            "code": torch.tensor(
                self.code_tokenizer.encode(text=sample["code"], max_length=self.code_max_length, padding="max_length",
                                           truncation=True)
            )
        }

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        return self._encode(
            self.samples[sample_id]
        )
