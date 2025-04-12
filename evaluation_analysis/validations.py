def validate_row(row, line_num, file_path):
    """
    Validate a row from the CSV file.
    Expected row format:
       [Requirement ID, Smell Type, Completeness, Completeness-Reasons, Correctness, Correctness-Reasons]
    Returns True if the row is valid, False otherwise.
    """
    # Check for exactly 6 columns:
    if len(row) != 7:
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

    # SMELL SUB TYPE VALIDATION STARTS
    smell_sub_type = row[2].strip()
    if smell_sub_type not in ['subjective_language', 'optional_parts', 'weak_verbs',
                              'passive_voice', 'negative', 'vague_pronouns',
                              'logical_inconsistencies', 'numerical_discrepancies', 'ambiguities','']:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Smell sub type '{smell_sub_type}' is not a valid smell sub type.")
    # SMELL SUB TYPE VALIDATION ENDS

    # COMPLETENESS VALIDATION STARTS
    completeness = row[3].strip()
    try:
        int_completeness = int(completeness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Completeness'{completeness}' is not a valid integer.")
    # COMPLETENESS VALIDATION ENDS

    # COMPLETENESS REASONS VALIDATION STARTS
    completeness_reasons_raw = row[4].strip()
    if completeness_reasons_raw != '':
        completeness_reasons = completeness_reasons_raw.replace('"', '')
        completeness_reasons_list = completeness_reasons.split(',')
    else:
        completeness_reasons_list = []
    # COMPLETENESS REASONS VALIDATION ENDS

    # CORRECTNESS VALIDATION STARTS
    correctness = row[5].strip()
    try:
        int_correctness = int(correctness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Correctness '{correctness}' is not a valid integer.")
    # CORRECTNESS VALIDATION ENDS

    # CORRECTNESS REASONS VALIDATION STARTS
    correctness_reasons_raw = row[6].strip()
    if correctness_reasons_raw != '':
        correctness_reasons = correctness_reasons_raw.replace('"', '')
        correctness_reasons_list = correctness_reasons.split(',')
    else:
        correctness_reasons_list = []
    # CORRECTNESS REASONS VALIDATION ENDS

    return int_req, smell_type, smell_sub_type, int_completeness, completeness_reasons_list, int_correctness, correctness_reasons_list


def validate_last_row(row, line_num, file_path):
    # Check for exactly 6 columns:
    if len(row) != 7:
        print(f"Validation error in {file_path} at line {line_num}: Expected 6 columns, got {len(row)}.")
        return False
    
    # FIRST COLUMN VALIDATION STARTS
    total_str = row[0].strip()  # remove extra spaces
    if total_str != 'Total':
        raise Exception(f"Validation warning in {file_path} at line {line_num}: First column '{total_str}' is not a valid 'Total' string.")
    # FIRST COLUMN VALIDATION ENDS

    # TOTAL COMPLETENESS VALIDATION STARTS
    total_completeness = row[3].strip()
    try:
        int_total_completeness = int(total_completeness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Total completeness'{total_completeness}' is not a valid integer.")
    # TOTAL COMPLETENESS VALIDATION ENDS

    # TOTAL CORRECTNESS VALIDATION STARTS
    total_correctness = row[5].strip()
    try:
        int_total_correctness = int(total_correctness)
    except ValueError:
        raise Exception(f"Validation warning in {file_path} at line {line_num}: Total completeness'{total_correctness}' is not a valid integer.")
    # TOTAL CORRECTNESS VALIDATION ENDS

    return total_str, total_completeness, total_correctness
