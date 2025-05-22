import os
import csv
import math
import textwrap
import matplotlib.pyplot as plt
import numpy as np

def generate_general_diagrams(evals_info):
    create_reason_counts(evals_info, calc_normalized_reason_counts_general, context = 'All Evaluations', directory = 'general')


def create_reason_counts(evals_info, normalizer_func, context = None, requested_variants = None, directory = ''):
    reasons_dict = {
        'completeness_reasons': {},
        'correctness_reasons': {},
    }

    # Next 3 for loops goes through all evaluations
    for game, results in evals_info.items():
        for variant_name, variant_results in results.items():
            if requested_variants and variant_name not in requested_variants:
                continue
            for req_id_str, req_result in variant_results.items():
                
                reason_types = ['completeness_reasons', 'correctness_reasons']
                for reason_type in reason_types:
                    if reason_type in req_result and req_result[reason_type]:
                        for reason in req_result[reason_type]:
                            if reason in reasons_dict[reason_type]:
                                reasons_dict[reason_type][reason] = reasons_dict[reason_type][reason] + 1
                            else:
                                reasons_dict[reason_type][reason] = 1
    
    context_1, context_2 = "Top 10 Completeness Reasons", "Top 10 Correctness Reasons"
    if context:
        context_1 = f"{context_1} ({context})"
        context_2 = f"{context_2} ({context})"

    normalized_reasons_dict = normalizer_func(reasons_dict)

    # Plot and save diagrams
    plot_top_10_reasons(normalized_reasons_dict['completeness_reasons'], context_1, "completeness_reasons.png", directory)
    plot_top_10_reasons(normalized_reasons_dict['correctness_reasons'], context_2, "correctness_reasons.png", directory)


def plot_top_10_reasons(reason_dict, title, filename, directory = ''):
    # Sort and get top 10
    sorted_items = sorted(reason_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, values = zip(*sorted_items)
    labels = explain_labels(labels)

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 100)
    plt.yticks(range(0, 100 + 1, 10))  # Set y-axis ticks at every 10 units

    plt.xticks(rotation=45, ha='center')  # Tilt for better spacing

    plt.grid(axis='y', linestyle=':', linewidth=1, alpha=0.7)  # Horizontal dotted lines
    plt.title(title)
    plt.xlabel('Reason')
    plt.ylabel('Percentage')
    plt.tight_layout()

    os.makedirs(os.path.join("evaluation_analysis/diagrams", directory), exist_ok=True)
    plt.savefig(os.path.join("evaluation_analysis/diagrams", directory, filename), dpi=300)
    plt.close()


def explain_labels(labels):
    """
    Replace each label in the tuple with its explanation from the dictionary.
    If a label has no explanation, it remains unchanged.
    """
    return tuple(
        textwrap.fill(label_explanations.get(label, label), width = 10)
        for label in labels
    )


def calc_normalized_reason_counts_general(reasons_dict, all_evals = True):
    current_file_path = os.path.abspath(__file__)
    # Directory containing the file (./model-generation/evaluation_analysis)
    current_dir = os.path.dirname(current_file_path)
    # Path to your CSV file
    csv_file_path = os.path.join(current_dir, 'games_component_counts.csv')
    reason_counts = {}

    # Open the CSV file
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)

        # If the CSV has a header and you want to skip it
        header = next(reader, None)

        # Process each row
        for row_number, row in enumerate(reader, start=1):
            print(f"Processing row {row_number}:")
            for col_index, value in enumerate(row):
                if row[0] == 'total' and value != 'total':
                    reasons = header[col_index]
                    reasons_list = reasons.split(',')
                    for reason in reasons_list:
                        reason_counts[reason] = int(value)
                        # There are 7 variants for each game
                        if all_evals:
                            reason_counts[reason] = reason_counts[reason] * 7

    for reason, count in reasons_dict['completeness_reasons'].items():
        reason_prefix = reason[0:3]
        reasons_dict['completeness_reasons'][reason] = (reasons_dict['completeness_reasons'][reason] / reason_counts[reason_prefix]) * 100

    for reason, count in reasons_dict['correctness_reasons'].items():
        reason_prefix = reason[0:3]
        reasons_dict['correctness_reasons'][reason] = (reasons_dict['correctness_reasons'][reason] / reason_counts[reason_prefix]) * 100

    return reasons_dict


label_explanations = {
    '1.1': 'missing actor',
    '1.2': 'missing participant',
    '1.3': 'missing message entirely',
    '1.3.1': 'missing part(s) of message',
    '1.4': 'missing loop entirely',
    '1.4.1': 'missing part(s) of loop cond.',
    '1.5': 'missing alt/else entirely',
    '1.5.1': 'missing part(s) of alt cond.',
    '1.5.2': 'missing an alt',
    '1.5.3': 'missing an else',
    '1.5.4': 'missing an extra else',
    '1.6': 'missing note entirely',
    '1.6.1': 'missing part(s) of note',
    # Correctness reasons
    '2.1': 'wrong actor',
    '2.2': 'wrong participant',
    '2.3.1': 'wrong part(s) of message',
    '2.3.2': 'wrong message direction',
    '2.3.3': 'wrongly located message',
    '2.4.1': 'wrong part(s) of loop cond.',
    '2.4.2': 'wrong loop',
    '2.4.3': 'wrongly located loop',
    '2.4.4': 'wrongly used loop',
    '2.5.1': 'wrong part(s) of alt/else cond.',
    '2.5.2': 'wrong alt/else',
    '2.5.3': 'wrongly located alt/else',
    '2.6.1': 'wrong part(s) of note',
    '2.7': 'no content',
}