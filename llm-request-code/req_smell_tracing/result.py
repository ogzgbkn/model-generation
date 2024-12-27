import copy
import datetime
import glob
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from openai import OpenAI

from req_smell_tracing.experiment import Experiment
from req_smell_tracing.tracing import (
    ComparisonResult,
    Trace,
    Tracing,
    TracingColumn,
    TracingComparator,
)
from req_smell_tracing.workflow import Context

logger = logging.getLogger(__name__)


class ResultMergingStrategy(Enum):
    INTERSECTION = "intersection"
    UNION = "union"


@dataclass
class ResultID:
    group: str
    experiment_name: str
    date: datetime.datetime
    run: int

    def __str__(self):
        """
        Get the result ID as a string.

        Result ID: <group>/<experiment_name>/<date>/<run>
        """
        return f"{self.group}/{self.experiment_name}/{self.date.strftime('%Y-%m-%d_%H%M%S')}/{self.run:02d}"

    def __repr__(self):
        return f"ResultID({self})"

    def get_execution(self):
        """
        Get the execution part of the result ID, which is the result ID without the run number.

        Execution: <group>/<experiment_name>/<date>

        Returns:
            str: The execution part of the result ID.
        """
        return f"{self.group}/{self.experiment_name}/{self.date.strftime('%Y-%m-%d_%H%M%S')}"

    @staticmethod
    def from_string(string: str) -> "ResultID":
        parts = string.split("/")

        if len(parts) != 4:
            raise ValueError(f"Invalid result ID: {string}")

        group = parts[0]
        experiment_name = parts[1]

        try:
            date = datetime.datetime.strptime(parts[2], "%Y-%m-%d_%H%M%S")
        except ValueError:
            raise ValueError(f"Invalid date in result ID: {string}")

        try:
            run = int(parts[3])
        except ValueError:
            raise ValueError(f"Run in result ID is not a number: {string}")

        return ResultID(group, experiment_name, date, run)

    def to_path(self):
        return os.path.join(
            self.group,
            self.experiment_name,
            self.date.strftime("%Y-%m-%d_%H%M%S"),
            f"result_{self.run:02d}.json",
        )


class Result:
    def __init__(
        self,
        id: ResultID,
        tracing: Tracing,
        ground_truth_tracing: Tracing | None = None,
        evaluation: dict[TracingComparator, ComparisonResult] = {},
    ):
        self.id = id
        self.tracing = tracing
        self.ground_truth_tracing = ground_truth_tracing
        self.evaluation = evaluation

    def __repr__(self):
        return f"Result({self.id})"

    def evaluate(self, tracing_comparators: list[TracingComparator], total_loc: int):
        """
        Evaluates the result by comparing the tracing with the ground truth tracing using the
        specified comparators.

        Arguments:
            tracing_comparators (list[TracingComparator]): The comparators to use.
            total_loc (int): The total lines of code in the codebase.
        """
        logger.info(f"Evaluating result {self.id}")

        for tracing_comparator in tracing_comparators:
            self.evaluation[tracing_comparator] = self.tracing.compare(
                self.ground_truth_tracing, tracing_comparator, total_loc
            )

    @staticmethod
    def load_results(
        config: dict[str, str | None],
        context: Context,
        load_groups: list[str] | None = None,
    ) -> list["Result"]:
        """
        Loads results from JSON files.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
            load_groups (list[str] | None): The groups to load results from. If None, all groups are loaded.

        Returns:
            list[Result]: The loaded results.
        """
        logger.info("Loading results")

        results_path = os.path.join(config["DATA_PATH"], "results", "**/*.json")
        result_files = [
            f
            for f in glob.glob(results_path, recursive=True)
            if os.path.split(f)[1].startswith("result_")
        ]
        results = []

        for result_file in result_files:
            with open(result_file, "r") as f:
                result_data = json.load(f)

            result_id = ResultID.from_string(result_data["id"])

            experiment = next(
                filter(
                    lambda x: x.name == result_id.experiment_name
                    and x.group == result_id.group,
                    context["experiments"],
                ),
                None,
            )

            if not experiment:
                logger.warning(
                    f"Experiment {result_id.experiment_name} not found, skipping"
                )
                continue

            with open(experiment.game.ground_truth_tracing_path, "r") as f:
                ground_truth_tracing = f.read()

            results.append(
                Result(
                    result_id,
                    Tracing.from_json(
                        json.dumps(result_data),
                        "requirements",
                        keys={
                            TracingColumn.REQUIREMENT_ID: "requirement_id",
                            TracingColumn.IMPLEMENTED: "implemented",
                            TracingColumn.LINES_OF_CODE: "lines_of_code",
                        },
                    ),
                    Tracing.from_csv(
                        ground_truth_tracing,
                        columns={
                            TracingColumn.REQUIREMENT_ID: "requirement_id",
                            TracingColumn.IMPLEMENTED: "implemented",
                            TracingColumn.LINES_OF_CODE: "lines_of_code",
                        },
                    ),
                )
            )

        logger.info(f"Loaded {len(results)} results")

        return results

    @staticmethod
    def load_batch_results(
        config: dict[str, str | None],
        context: Context,
        file: str,
    ) -> list["Result"]:
        """
        Loads results from JSONL batch file.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
            file (str): The batch file to load results from.

        Returns:
            list[Result]: The loaded results.
        """
        logger.info("Loading batch results")

        batch_path = os.path.join(config["DATA_PATH"], "results", file)
        results = []

        with open(batch_path, "r") as f:
            for line in f.readlines():
                result = json.loads(line)

                result_data = {
                    "id": result["custom_id"],
                    "requirements": json.loads(
                        result["response"]["body"]["choices"][0]["message"]["content"]
                    )["requirements"],
                }

                result_id = ResultID.from_string(result_data["id"])

                experiment = next(
                    filter(
                        lambda x: x.name == result_id.experiment_name
                        and x.group == result_id.group,
                        context["experiments"],
                    ),
                    None,
                )

                if not experiment:
                    logger.warning(
                        f"Experiment {result_id.experiment_name} not found, skipping"
                    )
                    continue

                with open(experiment.game.ground_truth_tracing_path, "r") as f:
                    ground_truth_tracing = f.read()

                results.append(
                    Result(
                        result_id,
                        Tracing.from_json(
                            json.dumps(result_data),
                            "requirements",
                            keys={
                                TracingColumn.REQUIREMENT_ID: "requirement_id",
                                TracingColumn.IMPLEMENTED: "implemented",
                                TracingColumn.LINES_OF_CODE: "lines_of_code",
                            },
                        ),
                        Tracing.from_csv(
                            ground_truth_tracing,
                            columns={
                                TracingColumn.REQUIREMENT_ID: "requirement_id",
                                TracingColumn.IMPLEMENTED: "implemented",
                                TracingColumn.LINES_OF_CODE: "lines_of_code",
                            },
                        ),
                    )
                )

        logger.info(f"Loaded {len(results)} results")

        return results

    @staticmethod
    def save_results_single_file(
        config: dict[str, str | None],
        context: Context,
        file: str,
    ):
        """
        Save results to single JSON file.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
            file (str): The file to save results to.
        """
        logger.info("Saving results to single file")

        results_path = os.path.join(config["DATA_PATH"], "results", file)
        results = []

        for result in context["results"]:
            experiment = next(
                filter(
                    lambda x: x.name == result.id.experiment_name
                    and x.group == result.id.group,
                    context["experiments"],
                ),
                None,
            )

            if not experiment:
                logger.warning(
                    f"Experiment {result.id.experiment_name} not found, skipping"
                )
                continue

            results.append(
                {
                    "id": str(result.id),
                    "game": result.id.group,
                    "experiment": result.id.experiment_name,
                    "timestamp": result.id.date.strftime("%Y-%m-%d_%H%M%S"),
                    "run": result.id.run,
                    "model": experiment.llm.model,
                    "smelly_requirements": experiment.requirements_smelly,
                    "requirements": [t.__dict__ for t in result.tracing.traces],
                }
            )

        with open(results_path, "w") as f:
            json.dump(results, f)

        logger.info("Results saved")

    @staticmethod
    def merge_results(
        config: dict[str, str | None], context: Context, strategy: ResultMergingStrategy
    ) -> dict[str, "Result"]:
        """
        Merges results from context.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
            strategy (ResultMergingStrategy): The strategy to use for merging.
        """
        logger.info("Merging results")

        match strategy:
            case ResultMergingStrategy.INTERSECTION:
                return Result._merge_intersection(
                    context["results"], context["experiments"]
                )
            case ResultMergingStrategy.UNION:
                return Result._merge_union(context["results"], context["experiments"])
            case _:
                raise ValueError(f"Unknown merging strategy: {strategy}")

    @staticmethod
    def _merge_intersection(
        results: list["Result"], experiments: list[Experiment]
    ) -> list["Result"]:
        """
        Merges results using the intersection strategy.

        Arguments:
            results (list[Result]): The results to merge.

        Returns:
            list[Result]: The merged results.
        """
        logger.info("Merging results using the intersection strategy")

        merged_results = []
        result_groups = Result.group_by(results, "execution")

        for group_results in result_groups.values():
            traces = list(
                zip(
                    *[
                        sorted(result.tracing.traces, key=lambda t: t.requirement_id)
                        for result in group_results
                    ]
                )
            )
            merged_traces = []

            for traces_group in traces:
                merged_loc = list(
                    set.intersection(
                        *map(set, map(lambda t: t.lines_of_code, traces_group))
                    )
                )
                merged_traces.append(
                    Trace(
                        traces_group[0].requirement_id,
                        traces_group[0].implemented,
                        merged_loc,
                    )
                )

            result_id = ResultID(
                group_results[0].id.group,
                group_results[0].id.experiment_name,
                group_results[0].id.date,
                1,
            )
            merged_results.append(
                Result(
                    result_id,
                    Tracing(merged_traces),
                    copy.copy(group_results[0].ground_truth_tracing),
                    copy.copy(group_results[0].evaluation),
                )
            )

        return merged_results

    @staticmethod
    def _merge_union(
        results: list["Result"], experiments: list[Experiment]
    ) -> list["Result"]:
        """
        Merges results using the union strategy.

        Arguments:
            results (list[Result]): The results to merge.

        Returns:
            list[Result]: The merged results.
        """
        logger.info("Merging results using the union strategy")

        merged_results = []
        result_groups = Result.group_by(results, "execution")

        for group_results in result_groups.values():
            traces = list(
                zip(
                    *[
                        sorted(result.tracing.traces, key=lambda t: t.requirement_id)
                        for result in group_results
                    ]
                )
            )
            merged_traces = []

            for traces_group in traces:
                merged_loc = list(
                    set.union(*map(set, map(lambda t: t.lines_of_code, traces_group)))
                )
                merged_traces.append(
                    Trace(
                        traces_group[0].requirement_id,
                        traces_group[0].implemented,
                        merged_loc,
                    )
                )

            result_id = ResultID(
                group_results[0].id.group,
                group_results[0].id.experiment_name,
                group_results[0].id.date,
                1,
            )
            merged_results.append(
                Result(
                    result_id,
                    Tracing(merged_traces),
                    copy.copy(group_results[0].ground_truth_tracing),
                    copy.copy(group_results[0].evaluation),
                )
            )

        return merged_results

    @staticmethod
    def evaluate_results(
        config: dict[str, str | None],
        context: Context,
        tracing_comparators: list[TracingComparator],
    ) -> list["Result"]:
        """
        Evaluates results from context.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.

        Returns:
            dict[str, dict[str, float]]: The evaluated results.
        """
        logger.info("Validating results")

        results_evaluated = []

        for experiment_name, result in context["results"].items():
            experiment = next(
                filter(lambda x: x.name == experiment_name, context["experiments"])
            )

            result = copy.copy(result)
            result.evaluate(tracing_comparators, experiment.lines_of_code())
            results_evaluated.append(result)

        return results_evaluated

    @staticmethod
    def save_evaluated_results(
        config: dict[str, str | None],
        context: Context,
        implementation_comparator: TracingComparator,
        per_experiment_comparator: TracingComparator,
    ):
        """
        Saves the results from context.

        Arguments:
            config (dict[str, str | None]): The application's configuration.
            context (Context): The application's context.
        """
        logger.info("Saving results")

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        evaluation_path = os.path.join(
            config["DATA_PATH"],
            "results",
            f"evaluation_{timestamp}.json",
        )
        experiments = context["experiments"]
        evaluation = {"evaluation": []}

        for result in context["evaluated_results"]:
            experiment = next(
                filter(lambda x: x.name == result.experiment_name, experiments)
            )

            evaluation["evaluation"].append(
                {
                    "experiment_name": result.experiment_name,
                    "group": experiment.group,
                    "implementation": result.evaluation[implementation_comparator],
                    "per_requirement": result.evaluation[per_experiment_comparator],
                }
            )

        with open(evaluation_path, "w") as f:
            json.dump(evaluation, f, indent=4)

        logger.info("Results saved")

    @staticmethod
    def group_by(results: list["Result"], key: str) -> dict[str, list["Result"]]:
        """
        Groups results by a key.

        Arguments:
            results (list[Result]): The results to group.
            key (str): The key to group by.

        Returns:
            dict[str, list[Result]]: The grouped results.
        """
        groups = defaultdict(list)

        for result in results:
            match key:
                case "group":
                    groups[result.id.group].append(result)
                case "experiment_name":
                    groups[result.id.experiment_name].append(result)
                case "date":
                    groups[result.id.date].append(result)
                case "run":
                    groups[result.id.run].append(result)
                case "execution":
                    groups[result.id.get_execution()].append(result)
                case _:
                    raise ValueError(f"Unknown key: {key}")

        return groups
    
    @staticmethod
    def write_batch_results(
        config: dict[str, str | None],
        context: Context,
        file: str,
        file_to_read: str):
        # Read the batch ID from the file and store it in a variable
        with open(file_to_read, "r") as batch_id_file:
            batch_id = batch_id_file.read().strip()  # Use strip() to remove any trailing newlines or spaces

        try:
            client = OpenAI(api_key=config["OPENAI_API_KEY"])
            # Check the status of the batch
            batch = client.batches.retrieve(batch_id)
            if batch.status == "completed":
                print(f"Batch {batch_id} is completed. Fetching results...")
                
                # Fetch the results
                results = client.files.content(batch.output_file_id)
                llm_type = "GPT"
                
                # Define the path to save the results
                results_file_path = os.path.join(config["DATA_PATH"], file)
                
                # Write results to the .jsonl file
                with open(results_file_path, "w") as f:
                    f.write(results.text)
                
                print(f"Results saved to {results_file_path}.")
            else:
                print(f"Batch {batch_id} is not completed yet. Current status: {batch_status['status']}.")
        except Exception as e:
            print(f"An error occurred: {e}")

    @staticmethod
    def write_batch_results_readable(
        config: dict[str, str | None],
        context: Context,
        file_to_read: str
    ):
        batch_file_path = os.path.join(config["DATA_PATH"], file_to_read)
        results = {}

        with open(batch_file_path, "r") as f:
            for line in f.readlines():
                result = json.loads(line)
                custom_id = result["custom_id"]
                results[custom_id] = result
        
        for experiment in context["experiments"]:
            llm_type = type(experiment.llm).__name__
            for i in range(experiment.iterations):
                exp_id = f"{experiment.group}/{experiment.name}/{experiment.timestamp}/{i + 1}"
                if exp_id in results:
                    found_result = results[exp_id] if exp_id in results else None
                    if found_result:
                        result_dir = os.path.join(config["DATA_PATH"], "results", llm_type, experiment.timestamp, experiment.group, experiment.name)
                        os.makedirs(result_dir, exist_ok=True)

                        # Instead of dumping the result object directly to the .json file, dump only the string LLM response inside the .txt file
                        with open(os.path.join(result_dir, f"result_{i+1}.txt"), "w") as f:
                            llm_response = json.loads(
                                found_result["response"]["body"]["choices"][0]["message"]["content"]
                            )
                            if "output" in llm_response:
                                f.write(llm_response["output"])
                            else:
                                f.write('Generated LLM response does not have the key "output"!')
                    else:
                        logger.info(f"Result CANNOT be found in the batch for the experiment {exp_id}")        

def _dirlist(path):
    return filter(os.path.isdir, map(lambda x: os.path.join(path, x), os.listdir(path)))
