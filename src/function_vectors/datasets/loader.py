import importlib
import json

PACKAGE_NAME = "function_vectors.datasets.resources"


def load(package_name: str, file_name: str) -> list[tuple[str, str]]:
    with importlib.resources.open_text(package_name, file_name) as file:
        data = json.loads(file.read())

    return [(instance["input"], instance["output"]) for instance in data]
