import nnsight
import torch

from function_vectors import datasets, tools
from function_vectors.function_vector import (
    apply_function_vector,
    construct_function_vector,
)


def function_vectors(model_name: str) -> None:
    model = nnsight.LanguageModel(model_name)

    in_context_dataset = datasets.ICL(
        data=datasets.ANTONYM_PAIRS,
        num_instances=20,
        num_demonstrations=4,
        random_seed=0,
    )
    zero_shot_dataset = datasets.ICL(
        data=datasets.ANTONYM_PAIRS,
        num_instances=20,
        num_demonstrations=0,
        random_seed=43,
    )

    function_vector, in_context_continuations = construct_function_vector(
        model, in_context_dataset, layer=15
    )
    zero_shot_continuations, function_vector_continuations = apply_function_vector(
        model, zero_shot_dataset, function_vector, layer=15
    )

    zero_shot_continuations = [
        tools.stringify_new_lines(continuation)
        for continuation in zero_shot_continuations
    ]
    function_vector_continuations = [
        tools.stringify_new_lines(continuation)
        for continuation in function_vector_continuations
    ]

    tools.chart_results(in_context_dataset, in_context_continuations)
    tools.chart_results(
        zero_shot_dataset, zero_shot_continuations, function_vector_continuations
    )


def instructed_function_vectors(model_name: str) -> None:
    model = nnsight.LanguageModel(model_name)

    in_context_dataset = datasets.ICL(
        data=datasets.ANTONYM_PAIRS,
        num_instances=20,
        num_demonstrations=4,
        random_seed=0,
    )
    zero_shot_dataset = datasets.ICL(
        data=datasets.ANTONYM_PAIRS,
        num_instances=20,
        num_demonstrations=0,
        random_seed=43,
    )

    in_context_function_vector, in_context_continuations = construct_function_vector(
        model, in_context_dataset, layer=15
    )

    zero_shot_continuations, function_vector_continuations = apply_function_vector(
        model, zero_shot_dataset, in_context_function_vector, layer=15
    )

    tools.chart_results(in_context_dataset, in_context_continuations)
    tools.chart_results(
        zero_shot_dataset, zero_shot_continuations, function_vector_continuations
    )

    model = nnsight.LanguageModel(model_name)

    instructed_zero_shot_dataset = datasets.InstructedICL(
        instruction=datasets.ANTONYM_INSTRUCTION,
        data=datasets.ANTONYM_PAIRS,
        num_instances=20,
        num_demonstrations=0,
        random_seed=0,
    )

    instructed_zero_shot_function_vector, instructed_zero_shot_continuations = (
        construct_function_vector(model, instructed_zero_shot_dataset, layer=15)
    )

    zero_shot_continuations, instructed_zero_shot_function_vector_continuations = (
        apply_function_vector(
            model, zero_shot_dataset, instructed_zero_shot_function_vector, layer=15
        )
    )

    tools.chart_results(
        instructed_zero_shot_dataset, instructed_zero_shot_continuations
    )
    tools.chart_results(
        zero_shot_dataset,
        instructed_zero_shot_continuations,
        instructed_zero_shot_function_vector_continuations,
    )

    cosine_similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    score = cosine_similarity(
        in_context_function_vector, instructed_zero_shot_function_vector
    )

    from rich.console import Console

    console = Console()

    console.print(f"[red bold]COSINE SIMILARITY: [blue bold]{score}")
