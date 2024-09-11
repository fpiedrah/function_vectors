import nnsight
import torch

from function_vectors import datasets


def construct_function_vector(
    model: nnsight.LanguageModel, dataset: datasets.ICL, layer: int = -1
) -> tuple[torch.Tensor, list[str]]:
    with model.trace() as tracer:
        with tracer.invoke(dataset.prompts):
            function_vector = (
                model.model.layers[layer].output[0][:, -1].mean(dim=0).save()
            )
            continuation_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

    continuations = model.tokenizer.batch_decode(continuation_ids.value)

    return (function_vector.value, continuations)


def apply_function_vector(
    model: nnsight.LanguageModel,
    dataset: datasets.ICL,
    function_vector: torch.Tensor,
    layer: int,
):
    ZERO_SHOT_SIZE = 0

    if not dataset.num_demonstrations == ZERO_SHOT_SIZE:
        raise ValueError(
            f"Expected zero-shot (0 demonstrations), but got {dataset.num_demonstrations}."
        )

    with model.trace() as tracer:
        with tracer.invoke(dataset.prompts):
            zero_shot_continuation_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

        with tracer.invoke(dataset.prompts):
            model.model.layers[layer].output[0][:, -1] += function_vector

            function_vector_continuation_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

    zero_shot_continuations = model.tokenizer.batch_decode(zero_shot_continuation_ids.value)
    function_vector_continuations = model.tokenizer.batch_decode(function_vector_continuation_ids.value)

    return (zero_shot_continuations, function_vector_continuations)
