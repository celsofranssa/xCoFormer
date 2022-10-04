from omegaconf import OmegaConf
import pytorch_lightning as pl
from transformers import AutoTokenizer
from source.DataModule.ZSDescCodeDataModule import ZSDescCodeDataModule
from source.callback.PredictionWriter import PredictionWriter
from source.model.DescCodeModel import DescCodeModel


class ZSPredictHelper:
    
    def __init__(self, params):
        self.params=params

    def perform_predict(self):
        dm = ZSDescCodeDataModule(
                self.params.data,
                self.get_tokenizer(self.params.model.desc_tokenizer),
                self.get_tokenizer(self.params.model.code_tokenizer))

        # model
        model = DescCodeModel(self.params.model)
        model.eval()

        trainer = pl.Trainer(
                gpus=self.params.trainer.gpus,
                callbacks=[PredictionWriter(self.params.prediction)]
            )

        print(f"Predicting {self.params.model.name} over {self.params.data.name} with fowling params\n"
              f"{OmegaConf.to_yaml(self.params)}\n")
        trainer.predict(
            model=model,
            datamodule=dm,

        )

    def get_tokenizer(self, params):
        return AutoTokenizer.from_pretrained(
            params.architecture
        )

