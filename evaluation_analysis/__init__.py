# For relative imports to work in Python 3.6
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .validations import validate_row, validate_last_row
from .rq1_diagram_generators import create_non_smelly_score_percentages, create_non_smelly_reason_counts
from .helpers import get_num_of_requirements