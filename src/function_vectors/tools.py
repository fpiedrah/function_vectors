import typing

import rich
from rich.table import Table

from function_vectors import datasets


def stringify_new_lines(string: str) -> str:
    return string.replace("\n", "\\n")


def chart_results(
    dataset: datasets.ICL,
    continuations: list[str],
    function_vector_continuations: typing.Optional[list[str]] = None,
) -> None:
    table = Table(title=("RESULTS"))

    column_names = ["FEATURES", "LABELS", "CONTINUATIONS", "INTERVENTION"]

    if not function_vector_continuations:
        column_names.pop()

    for column_name in column_names:
        table.add_column(column_name, justify="center")

    data = [
        dataset.features,
        dataset.labels,
        continuations,
        function_vector_continuations,
    ]
    if not function_vector_continuations:
        data.pop()

    rows = zip(*data)
    for row in rows:
        table.add_row(*row)

    rich.print(table)
