import hydra
import os

from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.ExplainHelper import ExplainHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper


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


def explain(params):
    explain_helper = ExplainHelper(params)
    explain_helper.perform_explain()


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


if __name__ == '__main__':
    perform_tasks()
