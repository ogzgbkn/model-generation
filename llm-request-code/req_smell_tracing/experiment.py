import copy
import datetime
import json
import logging
import os
import random
from typing import Callable

import tomlkit

from req_smell_tracing.game import Game
from req_smell_tracing.llm import GPT, LLM, Ollama, Prompt
from req_smell_tracing.tracing import TracingColumn
from req_smell_tracing.workflow import Context

logger = logging.getLogger(__name__)


class Experiment:
    """
    Represents an experiment that can be run with a specific configuration.

    Attributes:
        name (str): The name of the experiment.
        group (str): The group to which the experiment belongs.
        game (Game): The game associated with the experiment.
        code_language (str): The programming language used in the game's code.
        gpt (GPT): The GPT model used for generating prompts.
        prompts (list[GPTPrompt]): The list of prompts for the experiment.
        requirements (list[int]): The list of requirement IDs for the experiment.
        requirements_smelly (list[int]): The list of smelly requirement IDs for the experiment.
        description (str): The description of the experiment.
        iterations (int): The number of iterations to run the experiment.
        tracing_csv_columns (dict[str, str] | None): The mapping of tracing column names to CSV column names.
    """

    def __init__(
        self,
        name: str = "",
        group: str = "",
        game: Game | None = None,
        timestamp: str = "",
        code_language: str = "",
        llm: LLM | None = None,
        prompts: list[Prompt] = [],
        prompts_filled: bool = False,
        requirements: list[int] = [],
        requirements_smelly: list[int] = [],
        description: str = "",
        iterations: int = 1,
        tracing_csv_columns: dict[str, str] | None = None,
    ):
        """
        Initializes a new instance of the Experiment class.

        Args:
            name (str): The name of the experiment.
            group (str): The group to which the experiment belongs.
            game (Game): The game associated with the experiment.
            code_language (str): The programming language used in the game's code.
            llm (LLM): The LLM model used for generating prompts.
            prompts (list[GPTPrompt]): The list of prompts for the experiment.
            requirements (list[int], optional): The list of requirement IDs for the experiment. Defaults to an empty list.
            requirements_smelly (list[int], optional): The list of smelly requirement IDs for the experiment. Defaults to an empty list.
            description (str, optional): The description of the experiment. Defaults to an empty string.
            iterations (int, optional): The number of iterations to run the experiment. Defaults to 1.
            tracing_csv_columns (dict[str, str] | None, optional): The mapping of tracing column names to CSV column names. Defaults to None.
        """
        self.name = name
        self.group = group
        self.game = copy.copy(game)
        self.timestamp = timestamp
        self.code_language = code_language
        self.llm = copy.copy(llm)
        self.prompts = copy.deepcopy(prompts)
        self.prompts_filled = prompts_filled
        self.requirements = requirements
        self.requirements_smelly = requirements_smelly
        self.description = description
        self.iterations = iterations


    def alter_new(self, **kwargs) -> "Experiment":
        """
        Creates a new Experiment instance with the specified changes.

        Args:
            **kwargs: The changes to apply to the new experiment.

        Returns:
            Experiment: The new experiment instance.
        """
        return Experiment(
            name=kwargs.get("name", self.name),
            group=kwargs.get("group", self.group),
            game=kwargs.get("game", copy.copy(self.game)),
            code_language=kwargs.get("code_language", self.code_language),
            llm=kwargs.get("llm", copy.copy(self.llm)),
            prompts=kwargs.get("prompts", copy.deepcopy(self.prompts)),
            requirements=kwargs.get("requirements", self.requirements),
            requirements_smelly=kwargs.get(
                "requirements_smelly", self.requirements_smelly
            ),
            description=kwargs.get("description", self.description),
            iterations=kwargs.get("iterations", self.iterations)
        )

    def __repr__(self):
        return f"Experiment({self.name})"

    def details_as_dict(self) -> dict[str, any]:
        details_dict = {
            "name": self.name,
            "description": self.description,
            "group": self.group,
            "game": self.game.name,
            "code_language": self.code_language,
            "iterations": self.iterations,
            "llm": {
                "type": type(self.llm).__name__,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "top_p": self.llm.top_p,
                "max_tokens": self.llm.max_tokens,
            },
            "prompts": [{"role": p.role, "content": p.content} for p in self.prompts],
            "requirements": self.requirements,
            "requirements_smelly": self.requirements_smelly,
            "timestamp": self.timestamp
        }

        return _dict_remove_none(details_dict)

    def run(self, dry_run: bool = False, batch: bool = False, timestamp = datetime.datetime.now()):
        logger.info(f"Running experiment {self.name}")

        self.results = []

        if not self.prompts_filled:
            for prompt in self.prompts:
                prompt.fill_content(
                    self.game.get_requirements(
                        self.requirements, self.requirements_smelly
                    )
                )

            self.prompts_filled = True

        for i in range(self.iterations):
            result_id = f"{self.group}/{self.name}/{timestamp}/{i + 1}"
            logger.info(f"Running the iteration: {result_id}")

            if dry_run and not batch:
                res = ""
            else:
                res = self.llm.prompt(result_id, self.prompts, add_to_batch=batch)

            if not batch:
                if res:
                    self.results.append(res)
                else:
                    logger.error(
                        f"LLM prompt failed for experiment {self.name} in iteration {i + 1}"
                    )

    def save_results(self, config: dict[str, str | None], dry_run: bool = False, timestamp = 'Latest'):
        logger.info(f"Saving results for experiment {self.name}")

        llm_type = type(self.llm).__name__

        results_path = os.path.join(
            config["DATA_PATH"], "results", llm_type, timestamp, self.group, self.name
        )

        os.makedirs(results_path, exist_ok=True)

        with open(os.path.join(results_path, "experiment.toml"), "w") as f:
            tomlkit.dump(self.details_as_dict(), f)

        if dry_run:
            return

        for i, result in enumerate(self.results):
            result = json.loads(result)
            result["id"] = f"{self.group}/{self.name}/{self.timestamp}/{i + 1}"

            '''
            with open(os.path.join(results_path, f"result_{i + 1}.json"), "w") as f:
                json.dump(result, f, indent=4)
            '''

            # Instead of dumping the result object directly to the .json file, dump only the string LLM response inside the .txt file
            with open(os.path.join(results_path, f"result_{i + 1}.txt"), "w") as f:
                if "output" in result:
                    f.write(result["output"])
                else:
                    f.write('Generated LLM response does not have the key "output"!')

    def from_generator(
        self,
        generator,
        **generators_params,
    ) -> list["Experiment"]:
        requirements = self.game.requirements_ids()
        requirements_smelly = self.game.requirements_smelly_ids()
        experiments = []

        for i, (requirements, requirements_smelly) in enumerate(
            generator(requirements, requirements_smelly, **generators_params)
        ):
            experiment = Experiment(
                name=f"{self.name}_{(i + 1):02d}",
                group=self.group,
                game=copy.copy(self.game),
                code_language=self.code_language,
                llm=copy.copy(self.llm),
                prompts=copy.deepcopy(self.prompts),
                requirements=requirements,
                requirements_smelly=requirements_smelly,
                description=self.description,
                iterations=self.iterations
            )

            experiments.append(experiment)

        return experiments

    @staticmethod
    def read_experiments(
        config: dict[str, str | None],
        context: Context,
        experiments_defs: list[
            Callable[[list["Experiment"], dict[str, any], dict[str, str | None]], None]
        ],
    ) -> list["Experiment"]:
        logger.info("Loading experiments")

        experiments = []
        games = context["games"]

        for experiments_def in experiments_defs:
            experiments_def(experiments, games, config)

        return experiments

    @staticmethod
    def load_experiments(
        config: dict[str, str | None],
        context: Context,
        file: str,
    ) -> list["Experiment"]:
        """
        Loads the experiments from a JSON file.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.

        Returns:
            list[Experiment]: The list of experiments loaded from the file.
        """
        logger.info("Loading experiments")

        with open(os.path.join(config["DATA_PATH"], file), "r") as f:
            experiments = json.load(f)

        res = []

        for e in experiments:
            match e["llm"]["type"]:
                case "GPT":
                    llm = GPT(config, e["llm"]["model"], e["llm"]["temperature"])
                case "Ollama":
                    llm = Ollama(config, e["llm"]["model"], e["llm"]["temperature"])
                case _:
                    raise ValueError(f"Unknown LLM type: {e['llm']['type']}")

            res.append(
                Experiment(
                    name=e["name"],
                    group=e["group"],
                    game=context["games"][e["game"]],
                    timestamp=e["timestamp"],
                    code_language=e["code_language"],
                    llm=llm,
                    prompts=[
                        Prompt(config, role=p["role"], content=p["content"])
                        for p in e["prompts"]
                    ],
                    prompts_filled=True,
                    requirements=e["requirements"],
                    requirements_smelly=e["requirements_smelly"],
                    description=e["description"],
                    iterations=e["iterations"]
                )
            )

        return res

    @staticmethod
    def save_experiments(
        config: dict[str, str | None],
        context: Context,
        file: str,
    ):
        """
        Saves the experiments from the context to a single JSON file.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
        """
        logger.info("Saving experiments")

        experiments = context["experiments"]

        with open(os.path.join(config["DATA_PATH"], file), "w") as f:
            json.dump(
                [e.details_as_dict() for e in experiments],
                f,
                indent=4,
            )

        logger.info("Experiments saved")

    @staticmethod
    def run_experiments(
        config: dict[str, str | None],
        context: Context,
        dry_run: bool = False,
        batch: bool = False,
        batch_id_file: str = "batch_id.txt",
        save: bool = True,
    ) -> dict[str, list[str]]:
        """
        Runs the experiments from the context and saves the results.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.

        Returns:
            dict[str, list[str]]: A dictionary that contains the results for each experiment by name.
        """
        logger.info("Running experiments")

        experiments = context["experiments"]
        results = {}

        if batch and os.path.exists("batch.jsonl"):
            os.remove("batch.jsonl")

        experiments_run_timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        for experiment in experiments:
            experiment.timestamp = experiments_run_timestamp
            experiment.run(dry_run, batch, timestamp = experiments_run_timestamp)

            if not batch:
                if save:
                    experiment.save_results(config, dry_run, timestamp = experiments_run_timestamp)

                results[experiment.name] = experiment.results
                logger.info(f"Results saved for experiment {experiment.name}")

        if batch:
            batch_id = GPT.run_batch(config, dry_run)

            logger.warning("SAVE THE BATCH ID FOR LATER USE")
            logger.warning(f"===== Batch ID: {batch_id} =====")

        logger.info("Experiments completed")

        return results


class Generators:
    @staticmethod
    def no_smells(requirements, requirements_smelly, **_):
        yield (requirements, [])

    @staticmethod
    def all_smells(requirements, requirements_smelly, **_):
        requirements_without_smelly = list(
            filter(lambda x: x not in requirements_smelly, requirements)
        )

        yield (requirements_without_smelly, requirements_smelly)

    @staticmethod
    def random_smells_percentage(
        requirements, requirements_smelly, percentage=None, variations=1
    ):
        if not percentage:
            raise ValueError(
                "Percentage must be provided for random_smells_percentage generator"
            )

        num_smelly = int(float(len(requirements_smelly)) * percentage)

        for _ in range(variations):
            smelly = random.sample(requirements_smelly, num_smelly)
            non_smelly = list(filter(lambda x: x not in smelly, requirements))

            yield (non_smelly, smelly)

    @staticmethod
    def random_smells(requirements, requirements_smelly, variations=1):
        samples = []

        while len(samples) < variations:
            num_smelly = random.randint(1, len(requirements_smelly) - 1)
            smelly = random.sample(requirements_smelly, num_smelly)
            non_smelly = list(filter(lambda x: x not in smelly, requirements))
            sample = (sorted(non_smelly), sorted(smelly))

            # Only yield unique samples
            if sample in samples:
                continue

            samples.append(sample)

            yield sample

    @staticmethod
    def smell_type_specific_smells(requirements, requirements_smelly, variations=1, smell_type=1, game=None):
        if not game:
            raise Exception("The game is not provided!")
        requirements_with_specific_smell_type = game.requirements_with_specific_smell_type_ids(smell_type)
        requirements_without_smelly = list(
            filter(lambda x: x not in requirements_with_specific_smell_type, requirements)
        )

        yield (requirements_without_smelly, requirements_with_specific_smell_type)

def _dict_remove_none(d: dict[any, any]) -> dict[any, any]:
    new_d = copy.deepcopy(d)

    for k, v in d.items():
        if isinstance(v, dict):
            new_d[k] = _dict_remove_none(v)
        elif v is None:
            del new_d[k]

    return new_d
