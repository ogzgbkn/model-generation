import logging

from dotenv import dotenv_values

from experiments import workflow_defs
from req_smell_tracing.workflow import Workflow

config = dotenv_values(".env")
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)

    workflow = Workflow.load_workflow(
        config, workflow_defs=[workflow_defs.def_workflow]
    )
    workflow.run()


if __name__ == "__main__":
    main()
