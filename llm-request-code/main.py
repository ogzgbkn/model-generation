import logging
import os

from dotenv import dotenv_values

from experiments import workflow_defs
from req_smell_tracing.workflow import Workflow

from pathlib import Path

# Try with full path
full_path = os.path.join(os.path.dirname(__file__), ".env")
config = dotenv_values(full_path)
logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.DEBUG)
    root = Path(__file__).resolve()
    workflow = Workflow.load_workflow(
        config, workflow_defs=[workflow_defs.def_workflow]
    )
    workflow.run()


if __name__ == "__main__":
    main()
