# For relative imports to work in Python 3.6
import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from .validations import validate_row, validate_last_row