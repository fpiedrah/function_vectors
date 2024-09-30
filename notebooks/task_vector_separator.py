import marimo

__generated_with = "0.8.22"
app = marimo.App(width="medium")


@app.cell
def __():
    import marimo
    import nnsight
    from sklearn.metrics import accuracy_score

    from function_vectors.datasets.antonyms import (
        ICLAntonymsDataset,
        ZSLAntonymsDataset,
    )
    from function_vectors.datasets.capitalize import (
        ICLCapitalizeDataset,
        ZSLCapitalizeDataset,
    )
    from function_vectors.datasets.in_context_learning import (
        COLON_TEMPLATE,
        DASH_TEMPLATE,
        HASH_TEMPLATE,
        PIPE_TEMPLATE,
        SEPARATOR_META_TEMPLATE,
    )
    from function_vectors.task_vectors import apply_task_vector, construct_task_vector

    return (
        COLON_TEMPLATE,
        DASH_TEMPLATE,
        HASH_TEMPLATE,
        ICLAntonymsDataset,
        ICLCapitalizeDataset,
        PIPE_TEMPLATE,
        SEPARATOR_META_TEMPLATE,
        ZSLAntonymsDataset,
        ZSLCapitalizeDataset,
        accuracy_score,
        apply_task_vector,
        construct_task_vector,
        marimo,
        nnsight,
    )


@app.cell
def __(nnsight):
    NUM_CONTEX_EXAMPLES = 5
    NUM_INSTANCES = 30
    LAYER_INDEX = 10

    MODEL_NAME = "meta-llama/Llama-3.2-1B"
    model = nnsight.LanguageModel(MODEL_NAME)
    return (
        LAYER_INDEX,
        MODEL_NAME,
        NUM_CONTEX_EXAMPLES,
        NUM_INSTANCES,
        model,
    )


@app.cell
def __(
    COLON_TEMPLATE,
    ICLAntonymsDataset,
    LAYER_INDEX,
    NUM_CONTEX_EXAMPLES,
    NUM_INSTANCES,
    ZSLAntonymsDataset,
    accuracy_score,
    apply_task_vector,
    construct_task_vector,
    marimo,
    model,
):
    colon_in_context_dataset = ICLAntonymsDataset(
        instructions=None,
        num_context_examples=NUM_CONTEX_EXAMPLES,
        num_instances=NUM_INSTANCES,
        template=COLON_TEMPLATE,
        random_seed=101,  # REFACTOR THIS
    )

    colon_task_vector, colon_in_context_predicted = construct_task_vector(
        model, colon_in_context_dataset, LAYER_INDEX
    )

    colon_zero_shot_dataset = ZSLAntonymsDataset(
        instructions=None,
        num_instances=NUM_INSTANCES,
        template=COLON_TEMPLATE,
        random_seed=105,  # REFACTOR THIS
    )

    colon_zero_shot_predicted, colon_intervention_predicted = apply_task_vector(
        model,
        colon_zero_shot_dataset,
        colon_task_vector,
        LAYER_INDEX,
    )

    marimo.md(
        f"""
        ## COLON SEPARATED ANTONYMS DATASET
        ---

        **PROMPT EXAMPLE:** {colon_in_context_dataset.prompts[0]} \n
        **ICL ACCURACY:** {
            accuracy_score(
                colon_in_context_dataset.test_labels, 
                colon_in_context_predicted,
            )
        } \n
        **ZSL ACCURACY:** {
            accuracy_score(
                colon_zero_shot_dataset.test_labels, 
                colon_zero_shot_predicted,
            )
        } \n
        **INTERVENTION ACCURACY:** {
            accuracy_score(
                colon_zero_shot_dataset.test_labels, 
                colon_intervention_predicted,
            )
        }
        """
    )
    return (
        colon_in_context_dataset,
        colon_in_context_predicted,
        colon_intervention_predicted,
        colon_task_vector,
        colon_zero_shot_dataset,
        colon_zero_shot_predicted,
    )


@app.cell
def __(
    DASH_TEMPLATE,
    ICLAntonymsDataset,
    LAYER_INDEX,
    NUM_CONTEX_EXAMPLES,
    NUM_INSTANCES,
    ZSLAntonymsDataset,
    accuracy_score,
    apply_task_vector,
    construct_task_vector,
    marimo,
    model,
):
    dash_in_context_dataset = ICLAntonymsDataset(
        instructions=None,
        num_context_examples=NUM_CONTEX_EXAMPLES,
        num_instances=NUM_INSTANCES,
        template=DASH_TEMPLATE,
        random_seed=101,  # REFACTOR THIS
    )

    dash_task_vector, dash_in_context_predicted = construct_task_vector(
        model, dash_in_context_dataset, LAYER_INDEX
    )

    dash_zero_shot_dataset = ZSLAntonymsDataset(
        instructions=None,
        num_instances=NUM_INSTANCES,
        template=DASH_TEMPLATE,
        random_seed=105,  # REFACTOR THIS
    )

    dash_zero_shot_predicted, dash_intervention_predicted = apply_task_vector(
        model,
        dash_zero_shot_dataset,
        dash_task_vector,
        LAYER_INDEX,
    )

    marimo.md(
        f"""
        ## DASH SEPARATED ANTONYMS DATASET
        ---

        **PROMPT EXAMPLE:** {dash_in_context_dataset.prompts[0]} \n
        **ICL ACCURACY:** {
            accuracy_score(
                dash_in_context_dataset.test_labels, 
                dash_in_context_predicted,
            )
        } \n
        **ZSL ACCURACY:** {
            accuracy_score(
                dash_zero_shot_dataset.test_labels, 
                dash_zero_shot_predicted,
            )
        } \n
        **INTERVENTION ACCURACY:** {
            accuracy_score(
                dash_zero_shot_dataset.test_labels, 
                dash_intervention_predicted,
            )
        }
        """
    )
    return (
        dash_in_context_dataset,
        dash_in_context_predicted,
        dash_intervention_predicted,
        dash_task_vector,
        dash_zero_shot_dataset,
        dash_zero_shot_predicted,
    )


@app.cell
def __(
    HASH_TEMPLATE,
    ICLAntonymsDataset,
    LAYER_INDEX,
    NUM_CONTEX_EXAMPLES,
    NUM_INSTANCES,
    ZSLAntonymsDataset,
    accuracy_score,
    apply_task_vector,
    construct_task_vector,
    dash_in_context_dataset,
    marimo,
    model,
):
    hash_in_context_dataset = ICLAntonymsDataset(
        instructions=None,
        num_context_examples=NUM_CONTEX_EXAMPLES,
        num_instances=NUM_INSTANCES,
        template=HASH_TEMPLATE,
        random_seed=101,  # REFACTOR THIS
    )

    hash_task_vector, hash_in_context_predicted = construct_task_vector(
        model, dash_in_context_dataset, LAYER_INDEX
    )

    hash_zero_shot_dataset = ZSLAntonymsDataset(
        instructions=None,
        num_instances=NUM_INSTANCES,
        template=HASH_TEMPLATE,
        random_seed=105,  # REFACTOR THIS
    )

    hash_zero_shot_predicted, hash_intervention_predicted = apply_task_vector(
        model,
        hash_zero_shot_dataset,
        hash_task_vector,
        LAYER_INDEX,
    )

    marimo.md(
        f"""
        ## HASH SEPARATED ANTONYMS DATASET
        ---

        **PROMPT EXAMPLE:** {hash_in_context_dataset.prompts[0]} \n
        **ICL ACCURACY:** {
            accuracy_score(
                hash_in_context_dataset.test_labels, 
                hash_in_context_predicted,
            )
        } \n
        **ZSL ACCURACY:** {
            accuracy_score(
                hash_zero_shot_dataset.test_labels, 
                hash_zero_shot_predicted,
            )
        } \n
        **INTERVENTION ACCURACY:** {
            accuracy_score(
                hash_zero_shot_dataset.test_labels, 
                hash_intervention_predicted,
            )
        }
        """
    )
    return (
        hash_in_context_dataset,
        hash_in_context_predicted,
        hash_intervention_predicted,
        hash_task_vector,
        hash_zero_shot_dataset,
        hash_zero_shot_predicted,
    )


@app.cell
def __(
    ICLCapitalizeDataset,
    LAYER_INDEX,
    NUM_CONTEX_EXAMPLES,
    NUM_INSTANCES,
    PIPE_TEMPLATE,
    ZSLAntonymsDataset,
    accuracy_score,
    apply_task_vector,
    construct_task_vector,
    marimo,
    model,
):
    pipe_in_context_dataset = ICLCapitalizeDataset(
        instructions=None,
        num_context_examples=NUM_CONTEX_EXAMPLES,
        num_instances=NUM_INSTANCES,
        template=PIPE_TEMPLATE,
        random_seed=101,  # REFACTOR THIS
    )

    pipe_task_vector, pipe_in_context_predicted = construct_task_vector(
        model, pipe_in_context_dataset, LAYER_INDEX
    )

    pipe_zero_shot_dataset = ZSLAntonymsDataset(
        instructions=None,
        num_instances=NUM_INSTANCES,
        template=PIPE_TEMPLATE,
        random_seed=105,  # REFACTOR THIS
    )

    pipe_zero_shot_predicted, pipe_intervention_predicted = apply_task_vector(
        model,
        pipe_zero_shot_dataset,
        pipe_task_vector,
        LAYER_INDEX,
    )

    marimo.md(
        f"""
        ## PIPE SEPARATED ANTONYMS DATASET
        ---

        **PROMPT EXAMPLE:** {pipe_in_context_dataset.prompts[0]} \n
        **ICL ACCURACY:** {
            accuracy_score(
                pipe_in_context_dataset.test_labels, 
                pipe_in_context_predicted,
            )
        } \n
        **ZSL ACCURACY:** {
            accuracy_score(
                pipe_zero_shot_dataset.test_labels, 
                pipe_zero_shot_predicted,
            )
        } \n
        **INTERVENTION ACCURACY:** {
            accuracy_score(
                pipe_zero_shot_dataset.test_labels, 
                pipe_intervention_predicted,
            )
        }
        """
    )
    return (
        pipe_in_context_dataset,
        pipe_in_context_predicted,
        pipe_intervention_predicted,
        pipe_task_vector,
        pipe_zero_shot_dataset,
        pipe_zero_shot_predicted,
    )


@app.cell
def __(
    COLON_TEMPLATE,
    ICLCapitalizeDataset,
    LAYER_INDEX,
    NUM_CONTEX_EXAMPLES,
    NUM_INSTANCES,
    ZSLCapitalizeDataset,
    accuracy_score,
    apply_task_vector,
    construct_task_vector,
    marimo,
    model,
    pipe_in_context_dataset,
    pipe_task_vector,
):
    capitalize_in_context_dataset = ICLCapitalizeDataset(
        instructions=None,
        num_context_examples=NUM_CONTEX_EXAMPLES,
        num_instances=NUM_INSTANCES,
        template=COLON_TEMPLATE,
        random_seed=101,  # REFACTOR THIS
    )

    capitalize_task_vector, capitalize_in_context_predicted = construct_task_vector(
        model, capitalize_in_context_dataset, LAYER_INDEX
    )

    capitalize_zero_shot_dataset = ZSLCapitalizeDataset(
        instructions=None,
        num_instances=NUM_INSTANCES,
        template=COLON_TEMPLATE,
        random_seed=105,  # REFACTOR THIS
    )

    capitalize_zero_shot_predicted, capitalize_intervention_predicted = (
        apply_task_vector(
            model,
            capitalize_zero_shot_dataset,
            pipe_task_vector,
            LAYER_INDEX,
        )
    )

    marimo.md(
        f"""
        ## COLON SEPARATED CAPITALIZE DATASET
        ---

        **PROMPT EXAMPLE:** {pipe_in_context_dataset.prompts[0]} \n
        **ICL ACCURACY:** {
            accuracy_score(
                capitalize_in_context_dataset.test_labels, 
                capitalize_in_context_predicted,
            )
        } \n
        **ZSL ACCURACY:** {
            accuracy_score(
                capitalize_in_context_dataset.test_labels, 
                capitalize_zero_shot_predicted,
            )
        } \n
        **INTERVENTION ACCURACY:** {
            accuracy_score(
                capitalize_in_context_dataset.test_labels, 
                capitalize_intervention_predicted,
            )
        }
        """
    )
    return (
        capitalize_in_context_dataset,
        capitalize_in_context_predicted,
        capitalize_intervention_predicted,
        capitalize_task_vector,
        capitalize_zero_shot_dataset,
        capitalize_zero_shot_predicted,
    )


@app.cell
def __(
    capitalize_task_vector,
    colon_task_vector,
    dash_task_vector,
    hash_task_vector,
    marimo,
    pipe_task_vector,
):
    from plotly import express
    from scipy.spatial import distance

    task_vectors_labels = [
        "colon_task_vector",
        "dash_task_vector",
        "hash_task_vector",
        "pipe_task_vector",
        "capitalize_task_vector",
    ]
    task_vectors = [
        colon_task_vector.detach().numpy(),
        dash_task_vector.detach().numpy(),
        hash_task_vector.detach().numpy(),
        pipe_task_vector.detach().numpy(),
        capitalize_task_vector.detach().numpy(),
    ]

    similarity_matrix = 1 - distance.cdist(task_vectors, task_vectors, "cosine")

    figure = express.imshow(
        similarity_matrix,
        labels=dict(color="Cosine Similarity"),
        x=task_vectors_labels,
        y=task_vectors_labels,
        color_continuous_scale="Bluered",
    )

    """
    figure.update_layout(
        title="Cosine Similarity Matrix Heatmap",
        xaxis_title="Task Vectors",
        yaxis_title="Task Vectors",
    )
    """

    marimo.md(
        f"""
        ## SIMILARITY MATRIX
        ---

        {marimo.as_html(figure)}
        """
    )
    return (
        distance,
        express,
        figure,
        similarity_matrix,
        task_vectors,
        task_vectors_labels,
    )


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
