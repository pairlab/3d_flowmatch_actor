from functools import partial

from .peract import PeractDataPreprocessor
from .rlbench import RLBenchDataPreprocessor


def fetch_data_preprocessor(dataset_name):
    dataset_name = dataset_name.lower()
    if 'mesa' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=128)
    if 'peract2' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=256)
    if 'peract' in dataset_name:
        return partial(PeractDataPreprocessor, orig_imsize=256)
    if 'rlbench' in dataset_name:
        return partial(RLBenchDataPreprocessor, orig_imsize=256)
    return None
