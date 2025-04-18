import os
import math
import matplotlib.pyplot as plt
import numpy as np

def generate_general_diagrams(evals_info):
    create_reason_counts(evals_info, directory = 'general')


def create_reason_counts(evals_info, context = None, requested_variants = None, directory = ''):
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
    # Plot and save diagrams
    plot_top_10_reasons(reasons_dict['completeness_reasons'], context_1, "completeness_reasons.png", directory)
    plot_top_10_reasons(reasons_dict['correctness_reasons'], context_2, "correctness_reasons.png", directory)


def plot_top_10_reasons(reason_dict, title, filename, directory = ''):
    # Sort and get top 10
    sorted_items = sorted(reason_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, values = zip(*sorted_items)

    # Calculate dynamic y-axis limit
    max_value = max(values)
    dynamic_max = math.ceil(max_value * 1.25 / 10) * 10  # Round up to next multiple of 10

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, dynamic_max)
    plt.yticks(range(0, dynamic_max + 1, 10))  # Set y-axis ticks at every 10 units

    plt.grid(axis='y', linestyle=':', linewidth=1, alpha=0.7)  # Horizontal dotted lines
    plt.title(title)
    plt.xlabel('Reason Code')
    plt.ylabel('Count')
    plt.tight_layout()

    os.makedirs("diagrams", exist_ok=True)
    plt.savefig(os.path.join("diagrams", directory, filename), dpi=300)
    plt.close()