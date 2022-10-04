import hydra
import os
from omegaconf import OmegaConf
from source.helper.EvalHelper import EvalHelper
from source.helper.ExplainHelper import ExplainHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.ZSPredictHelper import ZSPredictHelper


def fit(params):
    fit_helper = FitHelper(params)
    fit_helper.perform_fit()

def predict(params):
    predict_helper = PredictHelper(params)
    predict_helper.perform_predict()

def eval(params):
    eval_helper = EvalHelper(params)
    eval_helper.perform_eval()

def explain(params):
    explain_helper = ExplainHelper(params)
    explain_helper.perform_explain()


def zs_predict(params):
    zs_predict_helper = ZSPredictHelper(params)
    zs_predict_helper.perform_predict()


@hydra.main(config_path="settings", config_name="settings.yaml", version_base=None)
def perform_tasks(params):
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)
    if "fit" in params.tasks:
        fit(params)

    if "predict" in params.tasks:
        predict(params)

    if "zs_predict" in params.tasks:
        zs_predict(params)

    if "eval" in params.tasks:
        eval(params)

    if "explain" in params.tasks:
        explain(params)

if __name__ == '__main__':
    perform_tasks()