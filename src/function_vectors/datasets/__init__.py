from .antonyms import ANTONYM_INSTRUCTION, ANTONYM_PAIRS
from .in_context_learning import ICL, ICLInstance, InstructedICL
from .pig_latin import PIG_LATIN_PAIRS
from .translation import (
    ENGLISH_DEUTSCH_PAIRS,
    ENGLISH_FRENCH_PAIRS,
    ENGLISH_SPANISH_PAIRS,
)

__all__ = [
    ICL,
    ICLInstance,
    InstructedICL,
    ANTONYM_INSTRUCTION,
    ANTONYM_PAIRS,
    ENGLISH_DEUTSCH_PAIRS,
    ENGLISH_FRENCH_PAIRS,
    ENGLISH_SPANISH_PAIRS,
    PIG_LATIN_PAIRS,
]
