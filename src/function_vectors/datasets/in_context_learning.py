import copy
import dataclasses
import functools
import random
import typing

import jinja2

COLON_TEMPLATE = jinja2.Template(
    """{%- if instructions -%}
    {{ instructions ~ '\n' }}
    {%- endif -%}

    {%- for feature, label in context_examples -%}
    {{ feature }}:{{ label ~ ' '}}
    {%- endfor -%}

    {{ test_feature }}:"""
)


QUESTION_ANSWER_TEMPLATE = jinja2.Template(
    """{%- if instructions -%}
    {{ instructions ~ '\n' }}
    {%- endif -%}
    
    {%- for feature, label in context_examples -%}
    Q: {{ feature ~ '\n' }}A: {{ label ~ '\n\n' }}
    {%- endfor -%}

    Q: {{ test_feature ~ '\n' }}A: """
)


@dataclasses.dataclass
class Instance:
    prompt: str
    context_examples: list[tuple[str, str]]
    test_example: tuple[str, str]


class InContextLearning:
    def __init__(
        self,
        dataset: list[tuple[str, str]],
        instructions: typing.Optional[str],
        num_context_examples: int,
        num_instances: int,
        random_seed: int,
        template: jinja2.Template,
    ):
        self._random_seed = random_seed

        self.dataset = dataset
        self.template = template
        self.instructions = instructions

        self.num_context_examples = num_context_examples
        self.num_instances = num_instances

        self.instances = self._generate_instances()

    @property
    @functools.cache
    def prompts(self):
        return [instance.prompt for instance in self.instances]

    @property
    @functools.cache
    def context_examples(self):
        return [instance.context_examples for instance in self.instances]

    @property
    @functools.cache
    def test_examples(self):
        return [instance.test_example for instance in self.instances]

    @property
    @functools.cache
    def test_features(self):
        return [instance.test_example[0] for instance in self.instances]

    @property
    @functools.cache
    def test_labels(self):
        return [instance.test_example[1] for instance in self.instances]

    def __getitem__(self, index: int) -> Instance:
        return self.instances[index]

    def __copy__(self):
        return type(self)(
            dataset=self.dataset,
            instructions=self.instructions,
            num_context_examples=self.num_context_examples,
            num_instances=self.num_instances,
            random_seed=self._random_seed,
            template=self.template,
        )

    def _sample_examples(self) -> tuple[list[tuple[str, str]], tuple[str, str]]:
        context_examples = random.sample(self.dataset, self.num_context_examples)
        test_example = random.choice(self.dataset)
        return context_examples, test_example

    def _format_prompt(
        self, context_examples: list[tuple[str, str]], test_feature: str
    ) -> str:
        return self.template.render(
            instructions=self.instructions,
            context_examples=context_examples,
            test_feature=test_feature,
        )

    def _generate_instances(self) -> list[Instance]:
        instances = []
        for index in range(self.num_instances):
            random.seed(self._random_seed + index)

            context_examples, (test_feature, test_label) = self._sample_examples()
            prompt = self._format_prompt(context_examples, test_feature)

            instances.append(
                Instance(
                    prompt=prompt,
                    context_examples=context_examples,
                    test_example=(test_feature, test_label),
                )
            )

        return instances


def corrupt(dataset: InContextLearning, random_seed: int) -> InContextLearning:
    dataset = copy.copy(dataset)
    instances = []
    for index, instance in enumerate(dataset.instances):
        random.seed(random_seed + index)

        _, random_context_examples = zip(
            *random.sample(dataset.dataset, dataset.num_context_examples)
        )
        context_features, _ = zip(*instance.context_examples)
        corrupted_context_examples = zip(context_features, random_context_examples)
        prompt = dataset._format_prompt(
            corrupted_context_examples, instance.test_example[0]
        )

        instances.append(
            Instance(
                prompt=prompt,
                context_examples=corrupted_context_examples,
                test_example=instance.test_example,
            )
        )

    dataset.instances = instances

    return dataset
