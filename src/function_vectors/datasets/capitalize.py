import functools

from function_vectors.datasets import loader
from function_vectors.datasets.in_context_learning import (
    InContextLearning,
    ZeroShotLearning,
)

CAPITALIZE_FILE_NAME = "capitalize.json"

ICLCapitalizeDataset = functools.partial(
    InContextLearning, dataset=loader.load(loader.PACKAGE_NAME, CAPITALIZE_FILE_NAME)
)

ZSLCapitalizeDataset = functools.partial(
    ZeroShotLearning, dataset=loader.load(loader.PACKAGE_NAME, CAPITALIZE_FILE_NAME)
)
