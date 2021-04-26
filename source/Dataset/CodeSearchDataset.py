import json

import torch
from torch.utils.data import Dataset


class CodeSearchDataset(Dataset):
    """CodeSearch Dataset.
    """

    def __init__(self, data_path, desc_tokenizer, code_tokenizer, desc_max_length, code_max_length):
        """
        """
        self.samples = []
        self.desc_tokenizer = desc_tokenizer
        self.code_tokenizer = code_tokenizer
        self.desc_max_length = desc_max_length
        self.code_max_length = code_max_length
        self._init_dataset(data_path)

    def _init_dataset(self, dataset_path):

        with open(dataset_path, "r") as dataset_file:
            for line in dataset_file:
                sample = json.loads(line)
                self.samples.append({
                    "idx": sample["idx"],
                    "desc": sample["desc"],
                    "code": sample["code"]
                })

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
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(self.samples[idx])
