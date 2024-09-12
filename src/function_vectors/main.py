import pprint

import nnsight

from function_vectors import datasets, tools
from function_vectors.function_vector import (
    apply_function_vector,
    construct_function_vector,
)

if __name__ == "__main__":
    model = nnsight.LanguageModel("google/gemma-2-2b")

    in_context_dataset = datasets.ICL(
        data=datasets.ANTONYMS, num_instances=20, num_demonstrations=4, random_seed=0
    )
    zero_shot_dataset = datasets.ICL(
        data=datasets.ANTONYMS, num_instances=20, num_demonstrations=0, random_seed=43
    )

    function_vector, in_context_continuations = construct_function_vector(
        model, in_context_dataset, layer=15
    )
    zero_shot_continuations, function_vector_continuations = apply_function_vector(
        model, zero_shot_dataset, function_vector, layer=15
    )

    zero_shot_continuations = [tools.stringify_new_lines(continuation) for continuation in zero_shot_continuations]
    function_vector_continuations = [tools.stringify_new_lines(continuation) for continuation in function_vector_continuations]

    tools.chart_results(in_context_dataset, in_context_continuations)
    tools.chart_results(
        zero_shot_dataset, zero_shot_continuations, function_vector_continuations
    )
