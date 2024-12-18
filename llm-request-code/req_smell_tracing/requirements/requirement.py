from req_smell_tracing.requirements.smell import Smell, SmellType


class Requirement:
    def __init__(
        self,
        id: int,
        description: str,
        smell: Smell | None = None,
        smell_type: SmellType | None = None,
    ):
        self.id = id
        self.description = description

        if (smell is None and smell_type is not None) or (
            smell is not None and smell_type is None
        ):
            raise ValueError(
                f"Smell and smell type must be provided together (requirement {id})"
            )

        self.smell = smell
        self.smell_type = smell_type

    def __repr__(self):
        return f"Requirement({self.id}, {self.description}, {self.smell}, {self.smell_type})"

    @staticmethod
    def format_as_list(requirements: list["Requirement"]) -> str:
        return "\n".join(
            [
                f"{req.id}. {req.description}"
                for req in sorted(requirements, key=lambda x: x.id)
            ]
        )
