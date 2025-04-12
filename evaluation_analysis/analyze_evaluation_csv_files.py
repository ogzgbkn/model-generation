import os
import csv
from collections import defaultdict

from helpers import get_num_of_requirements
from validations import validate_row, validate_last_row
from diagram_generators import create_non_smelly_score_percentages, create_non_smelly_reason_counts


def process_csv_file(results_dict, file_path):
    """
    Process an evaluation CSV file:
      - Extract game name and variant name from the file path.
      - Read and validate each row in the CSV.
      - Store the results in the results dict.
    """
    # Extract variant name and game name
    # Expected path: .../<game_name>/<variant_name>/evaluation.csv
    variant_name = os.path.basename(os.path.dirname(file_path))
    parent_dir = os.path.dirname(os.path.dirname(file_path))
    game_name = os.path.basename(parent_dir)
    num_of_requirements = get_num_of_requirements(game_name)

    # Store game_name and variant_name in variables as required:
    print(f"Processing file: {file_path}")

    # Open and process the CSV file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if line_num != 1 and line_num <= num_of_requirements + 1:
                # Unpack the validated row into the expected fields:
                req_id, smell_type, smell_sub_type, completeness, completeness_reasons, correctness, correctness_reasons = validate_row(row, line_num, file_path)
                if 'random' not in variant_name:
                    results_dict_variant_name = variant_name.removesuffix('_01')
                else:
                    results_dict_variant_name = variant_name
                eval_result = {
                    'requirement_id': req_id,
                    'smell_type': smell_type,
                    'smell_sub_type': smell_sub_type,
                    'completeness': completeness,
                    'completeness_reasons': completeness_reasons,
                    'correctness': correctness,
                    'correctness_reasons': correctness_reasons,
                }
                results_dict[game_name][results_dict_variant_name][req_id] = eval_result
            # Last line of the CSV that shows the completeness and correctness results
            elif line_num == num_of_requirements + 2:
                total_str, total_completeness, total_correctness = validate_last_row(row, line_num, file_path)
                results_dict[game_name][results_dict_variant_name]['total'] = {
                    'total': total_str,
                    'completeness': total_completeness,
                    'correctness': total_correctness,
                }
            else:
                print(f"Skipping invalid row at line {line_num} in {file_path}.")


def analyze_evaluations(target_path):
    nested_dict = lambda: defaultdict(nested_dict)
    results_dict = nested_dict()
    for root, dirs, files in os.walk(target_path):
        for file in files:
            if file == 'evaluation.csv':
                file_path = os.path.join(root, file)
                process_csv_file(results_dict, file_path)
    return results_dict


def draw_diagrams(evals_info):
    create_non_smelly_score_percentages(evals_info)
    create_non_smelly_reason_counts(evals_info)


if __name__ == '__main__':
    # Go one directory up and into 'evaluations'
    # current_path = os.getcwd()
    # Absolute path to the current file
    current_file_path = os.path.abspath(__file__)
    # Directory containing the file (./model-generation/evaluation_analysis)
    current_dir = os.path.dirname(current_file_path)
    # One level up from the file's directory (./model-generation)
    parent_dir = os.path.dirname(current_dir)
    target_path = os.path.abspath(os.path.join(parent_dir, 'evaluations'))

    if os.path.exists(target_path):
        evals_info = analyze_evaluations(target_path)
        draw_diagrams(evals_info)
    else:
        print(f"The directory '{target_path}' does not exist.")