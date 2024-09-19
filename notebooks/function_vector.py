import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import random

    import nnsight

    from function_vectors import datasets
    from function_vectors.function_vector import batched_average_indirect_effect

    return batched_average_indirect_effect, datasets, nnsight, random


@app.cell
def __(random):
    MODEL_NAME = "google/gemma-2-2b"
    RANDOM_SEED = 42
    BATCH_SIZE = 1

    random.seed(RANDOM_SEED)
    return BATCH_SIZE, MODEL_NAME, RANDOM_SEED


@app.cell
def __(MODEL_NAME, datasets, nnsight, random):
    model = nnsight.LanguageModel(MODEL_NAME)

    dataset = datasets.AntonymsDataset(
        instructions=None,
        num_context_examples=5,
        num_instances=20,
        random_seed=random.randint(0, 100),
        template=datasets.COLON_TEMPLATE,
    )

    corrupted_dataset = datasets.corrupt(dataset, random_seed=random.randint(0, 100))
    return corrupted_dataset, dataset, model


@app.cell
def __(
    BATCH_SIZE,
    batched_average_indirect_effect,
    corrupted_dataset,
    dataset,
    model,
):
    average_indirect_effect = batched_average_indirect_effect(
        model, dataset, corrupted_dataset, BATCH_SIZE
    )

    average_indirect_effect
    return (average_indirect_effect,)


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
