import json


def get_sample(sample_id, samples_path):
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
