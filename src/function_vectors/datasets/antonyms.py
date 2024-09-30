import functools

from function_vectors.datasets import loader
from function_vectors.datasets.in_context_learning import (
    InContextLearning,
    ZeroShotLearning,
)

ANTONYMS_FILE_NAME = "antonyms.json"

ICLAntonymsDataset = functools.partial(
    InContextLearning, dataset=loader.load(loader.PACKAGE_NAME, ANTONYMS_FILE_NAME)
)

ZSLAntonymsDataset = functools.partial(
    ZeroShotLearning, dataset=loader.load(loader.PACKAGE_NAME, ANTONYMS_FILE_NAME)
)
