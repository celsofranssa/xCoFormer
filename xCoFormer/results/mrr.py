from statistics import mean

import hydra
from annoy import AnnoyIndex
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from source.Dataset.CodeSearchDataset import CodeSearchDataset
from util.last_versions.JointEncoder import JointEncoder


def reload_model(cfg):
    reloaded_model = JointEncoder.load_from_checkpoint(cfg.checkpoint.path + cfg.evaluation.model_name)
    reloaded_model.freeze()
    return reloaded_model


def get_dataloader(cfg):
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    dataset = CodeSearchDataset(
        path=cfg.dataset.test_path,
        tokenizer=tokenizer,
        max_length=cfg.preprocessing.max_length)

    dataloader = DataLoader(dataset, batch_size=1, drop_last=True,
                            num_workers=cfg.preprocessing.num_workers)

    num_samples = len(dataset)

    return dataloader, num_samples


def get_representations(reloaded_model, test_dataloader):
    code_representations = []
    desc_representations = []
    i = 0
    for sample in test_dataloader:
        x1, x2 = sample["x1"], sample["x2"]
        x1, x2 = reloaded_model(x1, x2)
        code_representations.append(x1)
        desc_representations.append(x2)
        i += 1
        print(i)
    return code_representations, desc_representations


def build_index(representations, representation_size, num_trees=10, metric="angular"):
    """
    Builds an index.
    :param representations: a collection of representation.
    :param representation_size: the dimension of the representation.
    :param num_trees: number of trees to be used.
    :param metric: the similarity metric.
    :return: an index.
    """
    index = AnnoyIndex(representation_size, metric)
    for i, code in enumerate(representations):
        index.add_item(i, code.squeeze())
    index.build(num_trees)  # 10 trees
    return index


def mrr(desc_representations, index, n_items):
    reciprocal_ranks = []

    for i, desc in enumerate(desc_representations):
        results = index.get_nns_by_vector(desc.squeeze(), n_items)
        reciprocal_ranks.append(1.0 / (results.index(i) + 1))

    return mean(reciprocal_ranks)


@hydra.main(config_path="../configs/config.yaml")
def evaluate_mrr(cfg):
    # reload model
    reloaded_model = reload_model(cfg)

    eval_dataloader, n_items = get_dataloader(cfg)

    # representation
    code_representations, desc_representations = get_representations(reloaded_model, eval_dataloader)

    # index
    index = build_index(code_representations, representation_size=768)

    # statics
    test_mrr = mrr(desc_representations, index, n_items)
    print(test_mrr)


if __name__ == '__main__':
    evaluate_mrr()
