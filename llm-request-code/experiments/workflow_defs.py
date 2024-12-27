from experiments import new_experiments_def
from req_smell_tracing.experiment import Experiment
from req_smell_tracing.game import Game
from req_smell_tracing.result import Result
from req_smell_tracing.workflow import Phase


def def_workflow(workflow, args):
    with Phase("init", workflow) as phase:
        phase.add("load_games", Game.load_games, to_context="games")
        phase.add(
            "read_experiments",
            Experiment.read_experiments,
            experiments_defs=[new_experiments_def.def_experiments],
            to_context="experiments",
        )

    with Phase("run", workflow) as phase:
        phase.add(
            "run_experiments",
            Experiment.run_experiments,
            to_context="results",
            batch=True,
            batch_id_file="batch_id.txt"
        )
        phase.add(
            "save_experiments", Experiment.save_experiments, file="experiments.json"
        )

    with Phase("load", workflow) as phase:
        phase.add(
            "load_results",
            Result.load_results,
            to_context="results",
            load_groups=[args.group] if args.group else None,
        )

    with Phase("load_batch", workflow) as phase:
        phase.add("load_games", Game.load_games, to_context="games")
        phase.add(
            "load_experiments",
            Experiment.load_experiments,
            file="experiments.json",
            to_context="experiments",
        )
        phase.add(
            "write_batch_results",
            Result.write_batch_results,
            to_context="results_batch",
            file="batch_results.jsonl",
            file_to_read="batch_id.txt"
        )
        phase.add(
            "write_batch_results_readable",
            Result.write_batch_results_readable,
            to_context="results_batch",
            file_to_read="batch_results.jsonl"
        )

        """ phase.add(
            "load_batch_results",
            Result.load_batch_results,
            to_context="results",
            file="batch_results.jsonl",
        ) """

    with Phase("save_batch_results", workflow) as phase:
        phase.add(
            "save_batch_results", Result.save_results_single_file, file="results.json"
        )

    workflow.command("run", ["init", "run"])
    workflow.command("batchtoresults", ["load_batch"])
