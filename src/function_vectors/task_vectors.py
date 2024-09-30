import nnsight
import torch

from function_vectors.datasets.in_context_learning import InContextLearning


def construct_task_vector(
    model: nnsight.LanguageModel,
    dataset: InContextLearning,
    layer_index: int,
    use_inference_fabric: bool = False,
) -> tuple[torch.Tensor, list[str]]:
    with model.trace(remote=use_inference_fabric) as tracer:
        with tracer.invoke(dataset.prompts):
            layer_output = model.model.layers[layer_index].output[0][:, -1]
            task_vector = layer_output.mean(dim=0).save()
            next_token_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

    predicted_tokens = model.tokenizer.batch_decode(next_token_ids.value)

    return (task_vector.value, predicted_tokens)


def apply_task_vector(
    model: nnsight.LanguageModel,
    dataset: InContextLearning,
    task_vector: torch.Tensor,
    layer_index: int,
    use_inference_fabric: bool = False,
) -> tuple[list[str], list[str]]:
    if dataset.num_context_examples != 0:
        raise ValueError(
            f"Expected zero-shot (0 context examples), but got {dataset.num_context_examples}."
        )

    with model.trace(remote=use_inference_fabric) as tracer:
        with tracer.invoke(dataset.prompts):
            zero_shot_next_token_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

        with tracer.invoke(dataset.prompts):
            model.model.layers[layer_index].output[0][:, -1] += task_vector
            modified_next_token_ids = model.lm_head.output[:, -1].argmax(dim=-1).save()

    zero_shot_predictions = model.tokenizer.batch_decode(zero_shot_next_token_ids.value)
    modified_predictions = model.tokenizer.batch_decode(modified_next_token_ids.value)

    return (zero_shot_predictions, modified_predictions)
