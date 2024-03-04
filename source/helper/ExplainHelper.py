import json

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from source.model.BiEncoderModel import BiEncoderModel


class ExplainHelper:

    def __init__(self, params):
        self.params = params

    def perform_explain(self):
        print("using the following parameters:\n", OmegaConf.to_yaml(self.params))

        # override some of the params with new values
        model = BiEncoderModel.load_from_checkpoint(
            checkpoint_path=self.params.model_checkpoint.dir + self.params.model.name + "_" + self.params.data.name + ".ckpt",
            **self.params.model
        )

        # tokenizers
        x1_tokenizer = self.get_tokenizer(self.params.model)
        x2_tokenizer = x1_tokenizer

        x1_length = self.params.data.desc_max_length
        x2_length = self.params.data.code_max_length

        desc, code = self.get_sample(
            self.params.attentions.sample_id,
            self.params.attentions.dir + self.params.data.name + "_samples.jsonl"
        )

        x1 = x1_tokenizer.encode(text=desc, max_length=x1_length, padding="max_length",
                                 truncation=True)
        x2 = x2_tokenizer.encode(text=code, max_length=x2_length, padding="max_length",
                                 truncation=True)

        # predict
        model.eval()

        r1_attentions, r2_attentions = model(torch.tensor([x1]), torch.tensor([x2]))

        attentions = {
            "x1": desc,
            "x1_tokens": x1_tokenizer.convert_ids_to_tokens(x1),
            "x2": code,
            "x2_tokens": x2_tokenizer.convert_ids_to_tokens(x2),
            "r1_attentions": r1_attentions,
            "r2_attentions": r2_attentions
        }
        torch.save(obj=attentions,
                   f=self.params.attentions.dir +
                     self.params.model.name +
                     "_" +
                     self.params.data.name +
                     "_attentions.pt")

    def get_sample(self, sample_id, samples_path):
        """
        Gets code and desc to be analysed by model attention patterns.
        :param sample_id:
        :param samples_path:
        :return:
        """

        with open(samples_path, "r") as samples_file:
            lines = samples_file.readlines()
            if sample_id >= len(lines):
                sample_id = 0
            sample = json.loads(lines[sample_id])

        return sample["desc"], sample["code"]

    def get_tokenizer(self, params):
        tokenizer = AutoTokenizer.from_pretrained(
            params.architecture
        )
        if params.architecture == "gpt2":
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        return tokenizer
