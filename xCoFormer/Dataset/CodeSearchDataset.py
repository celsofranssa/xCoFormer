import json

import torch
from torch.utils.data import Dataset


class CodeSearchDataset(Dataset):
    """CodeSearch Dataset.
    """

    def __init__(self, path, x1_tokenizer, x2_tokenizer, x1_length, x2_length):
        """
        """
        self.samples = []
        self.x1_tokenizer = x1_tokenizer
        self.x2_tokenizer = x2_tokenizer
        self.x1_length = x1_length
        self.x2_length = x2_length
        self._init_dataset(path)

    def _init_dataset(self, dataset_path):
        with open(dataset_path, "r") as dataset_file:
            for idx, line in enumerate(dataset_file):
                sample = json.loads(line)
                self.samples.append({
                    "id": idx,
                    "x1": sample["desc"],
                    "x2": sample["code"]
                })

    def _encode(self, sample):
        return {
            "id": sample["id"],
            "x1": torch.tensor(
                self.x1_tokenizer.encode(text=sample["x1"], max_length=self.x1_length, padding="max_length",
                                         truncation=True)
            ),
            "x2": torch.tensor(
                self.x2_tokenizer.encode(text=sample["x2"], max_length=self.x2_length, padding="max_length",
                                         truncation=True)
            )
        }

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self._encode(self.samples[idx])
