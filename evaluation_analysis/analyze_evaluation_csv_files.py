import os
import csv
from collections import defaultdict

def get_num_of_requirements(game_name):
    games = {
        'dice_game': 25,
        'arkanoid': 19,
        'snake': 14,
        'scopa': 16,
        'pong': 20
    }
    return games[game_name]


def validate_row(row, line_num, file_path):
    """
    Validate a row from the CSV file.
    Expected row format:
       [Requirement ID, Smell Type, Completeness, Completeness-Reasons, Correctness, Correctness-Reasons]
    Returns True if the row is valid, False otherwise.
    """
    # Check for exactly 6 columns:
    if len(row) != 6:
        print(f"Validation error in {file_path} at line {line_num}: Expected 6 columns, got {len(row)}.")
        return False

    # REQUIREMENT ID VALIDATION STARTS
    # Validation on the first column (Requirement ID)
    req_id = row[0].strip()  # remove extra spaces
    try:
        int_req = int(req_id)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Requirement ID '{req_id}' is not a valid integer.")
    # REQUIREMENT ID VALIDATION ENDS

    # SMELL TYPE VALIDATION STARTS
    smell_type = row[1].strip()
    if smell_type not in ['lexical', 'semantic', 'syntactic', '']:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Smell type '{smell_type}' is not a valid smell type.")
    # SMELL TYPE VALIDATION ENDS

    # COMPLETENESS VALIDATION STARTS
    completeness = row[2].strip()
    try:
        int_completeness = int(completeness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Completeness'{completeness}' is not a valid integer.")
    # COMPLETENESS VALIDATION ENDS

    # COMPLETENESS REASONS VALIDATION STARTS
    completeness_reasons_raw = row[3].strip()
    if completeness_reasons_raw != '':
        completeness_reasons = completeness_reasons_raw.replace('"', '')
        completeness_reasons_list = completeness_reasons.split(',')
    else:
        completeness_reasons_list = []
    # COMPLETENESS REASONS VALIDATION ENDS

    # CORRECTNESS VALIDATION STARTS
    correctness = row[4].strip()
    try:
        int_correctness = int(correctness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Correctness '{correctness}' is not a valid integer.")
    # CORRECTNESS VALIDATION ENDS

    # CORRECTNESS REASONS VALIDATION STARTS
    correctness_reasons_raw = row[5].strip()
    if correctness_reasons_raw != '':
        correctness_reasons = correctness_reasons_raw.replace('"', '')
        correctness_reasons_list = correctness_reasons.split(',')
    else:
        correctness_reasons_list = []
    # CORRECTNESS REASONS VALIDATION ENDS

    return int_req, smell_type, int_completeness, completeness_reasons_list, int_correctness, correctness_reasons_list

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
    print(f"Game Name: {game_name}")
    print(f"Variant Name: {variant_name}")

    # Open and process the CSV file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, 1):
            if line_num != 1 and line_num <= num_of_requirements + 1:
                # Unpack the validated row into the expected fields:
                req_id, smell_type, completeness, completeness_reasons, correctness, correctness_reasons = validate_row(row, line_num, file_path)
                if 'random' not in variant_name:
                    results_dict_variant_name = variant_name.removesuffix('_01')
                else:
                    results_dict_variant_name = variant_name
                eval_result = {}
                eval_result['requirement_id'] = req_id
                eval_result['smell_type'] = smell_type
                eval_result['completeness'] = completeness
                eval_result['completeness_reasons'] = completeness_reasons
                eval_result['correctness'] = correctness
                eval_result['correctness_reasons'] = correctness_reasons
                results_dict[game_name][results_dict_variant_name][req_id] = eval_result
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
    test_var = ''


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
    else:
        print(f"The directory '{target_path}' does not exist.")