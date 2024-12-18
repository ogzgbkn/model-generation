import csv
import logging
import os
import tomllib

from req_smell_tracing.requirements.requirement import Requirement
from req_smell_tracing.requirements.smell import Smell, SmellType
from req_smell_tracing.workflow import Context

logger = logging.getLogger(__name__)


class Game:
    """
    Represents a game.

    Attributes:
        name (str): The name of the game.
        path (str): The path to the game's directory.
        requirements (list[Requirement]): The game's requirements.
        requirements_smelly (list[Requirement]): The game's smelly requirements.
        code (dict[str, str]): The game's code, indexed by language.
    """

    def __init__(
        self,
        name: str,
        path: str,
        code_languages: list[str],
        ground_truth_tracing_path: str,
    ):
        """
        Initializes a game.

        Arguments:
            name (str): The name of the game.
            path (str): The path to the game's directory.
            code_languages (list[str]): The languages of the game's code.
            ground_truth_tracing_oath (str): The path of the game's ground truth tracing file.
        """
        self.name = name
        self.path = path
        self.ground_truth_tracing_path = ground_truth_tracing_path

        # Load requirements

        requirements_path = os.path.join(path, "requirements.csv")
        self.requirements = []

        try:
            with open(requirements_path, "r") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    smell = Smell.from_str(row["smell"]) if row["smell"] else None
                    smell_type = (
                        SmellType.from_str(row["smell_type"])
                        if row["smell_type"]
                        else None
                    )

                    self.requirements.append(
                        Requirement(
                            int(row["id"]), row["description"], smell, smell_type
                        )
                    )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Requirements file not found at {requirements_path}"
            )

        if not self.requirements:
            raise ValueError(f"No requirements found in {requirements_path}")

        # Load smelly requirements

        requirements_smelly_path = os.path.join(path, "requirements_smelly.csv")
        self.requirements_smelly = []

        try:
            with open(requirements_smelly_path, "r") as f:
                reader = csv.DictReader(f)

                for row in reader:
                    smell = Smell.from_str(row["smell"]) if row["smell"] else None
                    smell_type = (
                        SmellType.from_str(row["smell_type"])
                        if row["smell_type"]
                        else None
                    )

                    self.requirements_smelly.append(
                        Requirement(
                            int(row["id"]), row["description"], smell, smell_type
                        )
                    )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Requirements smelly file not found at {requirements_smelly_path}"
            )

        if not self.requirements_smelly:
            raise ValueError(
                f"No smelly requirements found in {requirements_smelly_path}"
            )

        # Load code

        self.code = {}

        for code_language in code_languages:
            language_extension = {
                "java": "java",
                "python": "py",
            }[code_language]
            code_path = os.path.join(path, f"code.{language_extension}")

            try:
                with open(code_path, "r") as f:
                    self.code[code_language] = f.read()
            except FileNotFoundError:
                raise FileNotFoundError(f"Code file not found at {code_path}")

    def __repr__(self):
        return f"Game({self.name})"

    def requirements_ids(self) -> list[int]:
        """
        Returns the IDs of the game's requirements.
        """
        return [req.id for req in self.requirements]

    def requirements_smelly_ids(self) -> list[int]:
        """
        Returns the IDs of the game's smelly requirements.
        """
        return [req.id for req in self.requirements_smelly]

    def get_requirements(
        self, requirements_ids: list[int], requirements_smelly_ids: list[int]
    ) -> list[Requirement]:
        """
        Returns the game's requirements and smelly requirements with the given IDs.

        Arguments:
            requirements_ids (list[int]): The IDs of the requirements to get.
            requirements_smelly_ids (list[int]): The IDs of the smelly requirements to get.
        """
        requirements = [req for req in self.requirements if req.id in requirements_ids]
        requirements_smelly = [
            req for req in self.requirements_smelly if req.id in requirements_smelly_ids
        ]

        return requirements + requirements_smelly

    @staticmethod
    def load_games(
        config: dict[str, str | None], context: Context
    ) -> dict[str, "Game"]:
        """
        Loads games.

        Arguments:
            config (dict[str, str | None]): The program's configuration.
            logger (Logger): The program's logger.
            context (Context): The program's context.
        """
        logger.info("Loading games")

        data_path = config["DATA_PATH"]
        games_config_path = os.path.join(data_path, "games", "games.toml")

        try:
            with open(games_config_path, "rb") as f:
                games_config = tomllib.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Games config not found at {games_config_path}")

        games = {}

        for game in games_config["games"]:
            game = Game(
                game["name"],
                os.path.join(data_path, "games", game["path"]),
                game["code_languages"],
                os.path.join(
                    data_path, "games", game["path"], game["ground_truth_tracing_file"]
                ),
            )
            games[game.name] = game

        logger.info(f"Loaded {len(games)} games")

        return games
