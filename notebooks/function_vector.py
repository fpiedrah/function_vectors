import marimo

__generated_with = "0.8.15"
app = marimo.App(width="medium")


@app.cell
def __():
    import random

    import nnsight

    from function_vectors import datasets
    from function_vectors.function_vector import calculate_average_indirect_effect
    return calculate_average_indirect_effect, datasets, nnsight, random


@app.cell
def __(random):
    MODEL_NAME = "google/gemma-2-2b"
    RANDOM_SEED = 42

    random.seed(RANDOM_SEED)
    return MODEL_NAME, RANDOM_SEED


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
    return dataset, model


@app.cell
def __(dataset):
    dataset.prompts[0]
    return


@app.cell
def __(calculate_average_indirect_effect, dataset, model, random):
    average_indirect_effect = calculate_average_indirect_effect(
        model, dataset, random_seed=random.randint(0, 100)
    )

    average_indirect_effect
    return average_indirect_effect,


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
