import hydra
import os
import torch
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.ExpHelper import get_sample
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.model.CoEncoderModel import CoEncoderModel


def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()


def predict(params):
    predict_helper = PredictHelper(params)
    predict_helper.perform_predict()

def eval(params):
    print("Evaluating with the following parameters:\n", OmegaConf.to_yaml(params))
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()

def explain(hparams):
    print("using the following parameters:\n", OmegaConf.to_yaml(hparams))
    # override some of the params with new values
    model = CoEncoderModel.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt",
        **hparams.model
    )

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    x1_length = hparams.data.desc_max_length
    x2_length = hparams.data.code_max_length

    desc, code = get_sample(
        hparams.attentions.sample_id,
        hparams.attentions.dir + hparams.data.name + "_samples.jsonl"
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
               f=hparams.attentions.dir +
                 hparams.model.name +
                 "_" +
                 hparams.data.name +
                 "_attentions.pt")


def sim(hparams):
    print("using the following parameters:\n", OmegaConf.to_yaml(hparams))
    # override some of the params with new values
    model = CoEncoderModel.load_from_checkpoint(
        checkpoint_path=hparams.model_checkpoint.dir + hparams.model.name + "_" + hparams.data.name + ".ckpt",
        **hparams.model
    )

    # tokenizers
    x1_tokenizer = get_tokenizer(hparams.model)
    x2_tokenizer = x1_tokenizer

    x1_length = hparams.data.desc_max_length
    x2_length = hparams.data.code_max_length

    desc, code = get_sample(
        hparams.attentions.sample_id,
        hparams.attentions.dir + hparams.data.name + "_samples.jsonl"
    )

    x1 = x1_tokenizer.encode(text=desc, max_length=x1_length, padding="max_length",
                             truncation=True)
    x2 = x2_tokenizer.encode(text=code, max_length=x2_length, padding="max_length",
                             truncation=True)

    # predict
    model.eval()

    r1, r2 = model(torch.tensor([x1]), torch.tensor([x2]))
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    print(cos(r1, r2))




@hydra.main(config_path="settings", config_name="settings.yaml")
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    if "fit" in params.tasks:
        fit(params)
    if "predict" in params.tasks:
        predict(params)
    if "eval" in params.tasks:
        eval(params)
    if "explain" in params.tasks:
        explain(params)
    if "sim" in params.tasks:
        sim(params)




if __name__ == '__main__':
    perform_tasks()
