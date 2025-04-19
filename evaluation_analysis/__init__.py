# For relative imports to work in Python 3.6
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .validations import validate_row, validate_last_row
from .general_diagram_generators import create_reason_counts, generate_general_diagrams
from .rq1_diagram_generators import generate_rq1_diagrams
from .rq2_diagram_generators import generate_rq2_diagrams
from .rq3_diagram_generators import generate_rq3_diagrams
from .helpers import get_num_of_requirements