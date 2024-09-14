import dataclasses
import functools
import random


@dataclasses.dataclass
class ICLInstance:
    prompt: str
    demonstrations: list[tuple[str, str]]
    instance: tuple[str, str]


class ICL:
    def __init__(
        self,
        data: list[tuple[str, str]],
        num_instances: int,
        num_demonstrations: int = 5,
        separator: str = ":",
        random_seed: int = 42,
    ):
        self.data = data
        self.num_instances = num_instances
        self.num_demonstrations = num_demonstrations
        self.separator = separator
        self.random_seed = random_seed
        self.in_context_instances = self._generate()

    @property
    @functools.cache
    def prompts(self):
        return [instance.prompt for instance in self.in_context_instances]

    @property
    @functools.cache
    def demonstrations(self):
        return [instance.demonstrations for instance in self.in_context_instances]

    @property
    @functools.cache
    def instances(self):
        return [instance.instance for instance in self.in_context_instances]

    @property
    @functools.cache
    def features(self):
        return [instance.instance[0] for instance in self.in_context_instances]

    @property
    @functools.cache
    def labels(self):
        return [instance.instance[1] for instance in self.in_context_instances]

    def __getitem__(self, index: int) -> ICLInstance:
        return self.in_context_instances[index]

    def _format(self, demonstrations: list[tuple[str, str]], feature: str) -> str:
        string = " ".join(
            [f"{feature}{self.separator}{label}" for feature, label in demonstrations]
        )

        return f"{string} {feature}{self.separator}"

    def _sample(self) -> str:
        demonstrations = random.sample(self.data, self.num_demonstrations)
        instance = random.choice(self.data)

        return (demonstrations, instance)

    def _generate(self) -> list:
        instances = []

        for index in range(self.num_instances):
            random.seed(self.random_seed + index)

            (demonstrations, (feature, label)) = self._sample()

            instance = ICLInstance(
                prompt=self._format(demonstrations, feature),
                demonstrations=demonstrations,
                instance=(feature, label),
            )

            instances.append(instance)

        return instances


class InstructedICL(ICL):
    def __init__(
        self,
        instruction: str,
        data: list[tuple[str, str]],
        num_instances: int,
        num_demonstrations: int = 5,
        separator: str = ":",
        random_seed: int = 42,
    ):
        self.instruction = instruction.format(separator=separator)
        super().__init__(
            data, num_instances, num_demonstrations, separator, random_seed
        )

    def _generate(self):
        return [
            ICLInstance(
                prompt=f"{self.instruction} {instance.prompt}",
                demonstrations=instance.demonstrations,
                instance=instance.instance,
            )
            for instance in super()._generate()
        ]
