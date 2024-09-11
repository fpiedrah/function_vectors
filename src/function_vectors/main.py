import pprint
import nnsight

from function_vectors import datasets
from function_vectors.function_vector import construct_function_vector, apply_function_vector


if __name__ == "__main__":
    model = nnsight.LanguageModel("meta-llama/Meta-Llama-3.1-8B")

    in_context_dataset = datasets.ICL(datasets.ANTONYMS, 50)
    zero_shot_dataset = datasets.ICL(datasets.ANTONYMS, 20, 0)

    function_vector, in_context_continuations = construct_function_vector(model, in_context_dataset)
    zero_shot_continuations, function_vector_continuations = apply_function_vector(
        model, zero_shot_dataset, function_vector, 15
    )

    print("IN CONTEXT")
    pprint.pprint(list(zip(in_context_dataset.features, in_context_dataset.labels, in_context_continuations)))

    print("ZERO SHOT")
    pprint.pprint(list(zip(zero_shot_dataset.features, zero_shot_dataset.labels, zero_shot_continuations)))

    print("FUNCTION VECTOR")
    pprint.pprint(list(zip(zero_shot_dataset.features, zero_shot_dataset.labels, function_vector_continuations)))
