import torch
import nnsight

from function_vectors import datasets


def calculate_average_indirect_effect(
    model: nnsight.LanguageModel, dataset: datasets.InContextLearning, random_seed: int
):
    attention_heads_per_layer = model.config.num_attention_heads
    hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    attention_head_dimension = hidden_size // (attention_heads_per_layer + 1) # Why + 1? (WTF)
    total_attention_heads = attention_heads_per_layer * hidden_layers

    num_samples = dataset.num_instances
    corrupted_dataset = datasets.corrupt(dataset, random_seed)
    correct_completion_ids = [
        tokens[0] for tokens in model.tokenizer(dataset.test_labels)["input_ids"]
    ]

    with model.trace() as tracer:

        clean_head_outputs = {}
        with tracer.invoke(dataset.prompts):
            for layer in range(hidden_layers):
                head_outputs = model.model.layers[layer].self_attn.o_proj.input[:, -1]
                reshaped_head_outputs = head_outputs.reshape(
                    num_samples, attention_heads_per_layer, attention_head_dimension
                ).mean(dim=0)

                for head in range(attention_heads_per_layer):
                    clean_head_outputs[(layer, head)] = reshaped_head_outputs[head]

            logits_clean = model.lm_head.output[:, -1]

        with tracer.invoke(dataset.prompts):
            logits_corrupted = model.lm_head.output[:, -1]
            corrupted_log_probabilities = logits_corrupted.log_softmax(dim=-1)[
                torch.arange(num_samples), correct_completion_ids
            ].save()

        intervention_log_probabilities = {}
        for layer in range(hidden_layers):
            for head in range(attention_heads_per_layer):
                with tracer.invoke(corrupted_dataset.prompts):
                    layer_instance = model.model.layers[layer]
                    head_outputs = layer_instance.self_attn.o_proj.input[:, -1]

                    head_outputs.reshape(
                        num_samples, attention_heads_per_layer, attention_head_dimension
                    )[:, head] = clean_head_outputs[(layer, head)]

                    logits_intervened = model.lm_head.output[:, -1]
                    intervention_log_probabilities[(layer, head)] = (
                        logits_intervened.log_softmax(dim=-1)[
                            torch.arange(num_samples), correct_completion_ids
                        ].save()
                    )

    intervention_log_probabilities = einops.rearrange(
        torch.stack([value.value for value in intervention_log_probs.values()]),
        "(hidden_layers attention_heads) num_samples -> hidden_layers attention_heads num_samples",
        hidden_layers=hidden_layers,
    )

    difference = intervention_log_probabilities - corrupted_log_probabilities

    return difference.mean(dim=-1)
