import argparse
import json
import logging
import os
import sys
import traceback
from logging import Logger
from typing import Callable

logger = logging.getLogger(__name__)


type Context = dict[str, any]


class Workflow:
    def __init__(self, config: dict[str, str | None]):
        self.config = config
        self.phases = {}
        self.context = {}
        self.commands = {}
        self.args = None

    def command(self, cmd: str, phases: list[str]):
        self.commands[cmd] = phases

    def run_phases(self, phases: list[str]):
        for phase in phases:
            if phase not in self.phases:
                logger.error(f"Unknown phase: {phase}")
                continue

            self.phases[phase].run()
            self.run_phases(self.phases[phase].next_steps)

    def run(self):
        logger.info("Running workflow")

        command = self.args.command

        if command == "env":
            command = self.config["COMMAND"]

        if command not in self.commands:
            logger.error(f"Unknown command: {command}")
            sys.exit(1)

        self.run_phases(self.commands[command])

    @staticmethod
    def load_workflow(
        config: dict[str, str | None],
        workflow_defs: list[Callable[["Workflow", argparse.Namespace], None]],
    ) -> "Workflow":
        logger.info("Loading workflow")

        workflow = Workflow(config)

        parser = argparse.ArgumentParser(
            prog="req_smell_tracing",
            description="Run the requirements smell tracing workflow",
        )
        parser.add_argument("command", help="The command to run (e.g. run, load)")
        parser.add_argument("--group", help="The group of experiments to load")
        workflow.args = parser.parse_args()

        for workflow_def in workflow_defs:
            workflow_def(workflow, workflow.args)

        return workflow


class Phase:
    def __init__(self, name: str, workflow: Workflow):
        self.name = name
        self.steps = {}
        self.next_steps = []
        self.workflow = workflow

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            logger.error(f"Error in phase {self.name}: {exc_val}")
            logger.debug(traceback.format_exc())
            return False

        self.workflow.phases[self.name] = self

        return True

    def add(
        self,
        step_name: str,
        step_fn: Callable[[dict[str, str | None], Logger], any],
        to_context: str | None = None,
        to_file: str | None = None,
        **kwargs,
    ):
        self.steps[step_name] = (step_fn, to_context, to_file, kwargs)

    def then(self, phase_name: str):
        self.next_steps.append(phase_name)

    def run(self):
        logger.info(f"Running phase {self.name}")

        for step_name, (step_fn, to_context, to_file, kwargs) in self.steps.items():
            logger.info(f"Running step {step_name}")

            try:
                res = step_fn(self.workflow.config, self.workflow.context, **kwargs)

                if to_context:
                    self.workflow.context[to_context] = res
                else:
                    self.workflow.context[step_name] = res

                if to_file:
                    path = os.path.join(
                        self.workflow.config["DATA_PATH"], "results", to_file
                    )

                    with open(path, "w") as f:
                        json.dump(res, f, indent=4)

            except Exception as e:
                logger.critical(f"Error while running step {step_name}: {e}")
                logger.debug(traceback.format_exc())
                sys.exit(1)
