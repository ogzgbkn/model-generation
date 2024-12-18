from enum import Enum, auto


class Smell(Enum):
    PASSIVE_VOICE = auto()
    NEGATIVE = auto()
    VAGUE_PRONOUNS = auto()
    LOGICAL_INCONSISTENCIES = auto()
    NUMERICAL_DISCREPANCIES = auto()
    AMBIGUITIES = auto()
    SUBJECTIVE_LANGUAGE = auto()
    OPTIONAL_PARTS = auto()
    WEAK_VERBS = auto()

    @staticmethod
    def from_str(smell: str) -> "Smell":
        try:
            return Smell[smell.upper().replace("-", "_")]
        except KeyError:
            raise ValueError(f"Unknown smell: {smell}")


class SmellType(Enum):
    LEXICAL = auto()
    SEMANTIC = auto()
    SYNTACTIC = auto()

    @staticmethod
    def from_str(smell_type: str) -> "SmellType":
        try:
            return SmellType[smell_type.upper()]
        except KeyError:
            raise ValueError(f"Unknown smell type: {smell_type}")
