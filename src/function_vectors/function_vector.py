import typing

import einops
import nnsight
import torch

from function_vectors import datasets


def calculate_average_indirect_effect(
    model: nnsight.LanguageModel,
    dataset: datasets.InContextLearning,
    corrupted_dataset: datasets.InContextLearning,
    layer_indices: typing.Optional[list[int]],
) -> torch.Tensor:
    heads_per_layer = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    head_dimension = hidden_size // (heads_per_layer + 1)  # Note: Verify the +1 logic

    num_samples = dataset.num_instances
    correct_completion_ids = [
        tokens[0] for tokens in model.tokenizer(dataset.test_labels)["input_ids"]
    ]

    if layer_indices is None:
        layer_indices = range(len(num_layers))

    head_indices = list(range(heads_per_layer))

    with model.trace() as tracer:

        clean_head_outputs = {}
        with tracer.invoke(dataset.prompts):
            for layer in layer_indices:
                layer_self_attention = model.model.layers[layer].self_attn
                head_outputs = layer_self_attention.o_proj.input[:, -1]

                reshaped_head_outputs = head_outputs.reshape(
                    num_samples, heads_per_layer, head_dimension
                ).mean(dim=0)

                for head in head_indices:
                    clean_head_outputs[(layer, head)] = reshaped_head_outputs[head]

            logits_clean = model.lm_head.output[:, -1]

        with tracer.invoke(dataset.prompts):
            logits_corrupted = model.lm_head.output[:, -1]
            corrupted_log_probs = logits_corrupted.log_softmax(dim=-1)[
                torch.arange(num_samples), correct_completion_ids
            ].save()

        intervention_log_probabilities = {}
        for layer in layer_indices:
            for head in head_indices:
                with tracer.invoke(corrupted_dataset.prompts):
                    layer_self_attention = model.model.layers[layer].self_attn
                    head_outputs = layer_self_attention.o_proj.input[:, -1]

                    reshaped_head_outputs = head_outputs.reshape(
                        num_samples, heads_per_layer, head_dimension
                    )

                    reshaped_head_outputs[:, head] = clean_head_outputs[(layer, head)]

                    logits_intervened = model.lm_head.output[:, -1]
                    intervention_log_probabilities[(layer, head)] = (
                        logits_intervened.log_softmax(dim=-1)[
                            torch.arange(num_samples), correct_completion_ids
                        ].save()
                    )

    intervention_tensor = einops.rearrange(
        torch.stack([prob.value for prob in intervention_log_probabilities.values()]),
        "(num_layers num_heads) num_samples -> num_layers num_heads num_samples",
        num_layers=len(layer_indices),
    )

    difference = intervention_tensor - corrupted_log_probs

    return difference.mean(dim=-1)


def batched_average_indirect_effect(
    model: nnsight.LanguageModel,
    dataset: datasets.InContextLearning,
    corrupted_dataset: datasets.InContextLearning,
    batch_size: int,
):
    num_layers = model.config.num_hidden_layers
    heads_per_layer = model.config.num_attention_heads

    def _get_batches(num_layers: int, batch_size: int) -> typing.Iterator[range]:
        for index in range(0, num_layers, batch_size):
            yield range(num_layers)[index : index + batch_size]

    difference_tensor = torch.empty((0, heads_per_layer))

    for layer_indices in _get_batches(num_layers, batch_size):
        difference_tensor = torch.concat(
            [
                difference_tensor,
                calculate_average_indirect_effect(
                    model, dataset, corrupted_dataset, layer_indices
                ),
            ]
        )

    return difference_tensor
