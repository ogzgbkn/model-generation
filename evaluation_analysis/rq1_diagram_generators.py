import os
import matplotlib.pyplot as plt
import numpy as np

from helpers import get_num_of_requirements

def create_non_smelly_score_percentages(evals_info):
    percentages_dict = {}
    total_completeness_percentage = 0
    total_correctness_percentage = 0
    for game, results in evals_info.items():
        game_req_count = get_num_of_requirements(game)
        no_smells_completeness = results['no_smells']['total']['completeness']
        no_smells_correctness = results['no_smells']['total']['correctness']
        completeness_percentage = (int(no_smells_completeness) / game_req_count) * 100
        correctness_percentage = (int(no_smells_correctness) / game_req_count) * 100
        percentages_dict[game] = {
            'completeness_percentage': completeness_percentage,
            'correctness_percentage': correctness_percentage,
        }
        total_completeness_percentage = total_completeness_percentage + completeness_percentage
        total_correctness_percentage = total_correctness_percentage + correctness_percentage
    
    avg_completeness = total_completeness_percentage / 5
    avg_correctness = total_correctness_percentage / 5
    percentages_dict['average'] = {
        'completeness_percentage': avg_completeness,
        'correctness_percentage': avg_correctness,
    }

    # Ordered keys
    ordered_keys = ['average', 'dice_game', 'arkanoid', 'snake', 'scopa', 'pong']
    completeness = [percentages_dict[k]["completeness_percentage"] for k in ordered_keys]
    correctness = [percentages_dict[k]["correctness_percentage"] for k in ordered_keys]

    # Plot config
    bar_width = 0.35
    group_spacing = 0.4
    num_groups = len(ordered_keys)

    # Compute X positions for group centers
    group_centers = np.arange(num_groups) * (2 * bar_width + group_spacing)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(group_centers - bar_width / 2, completeness, width=bar_width, color='blue', label='Completeness')
    ax.bar(group_centers + bar_width / 2, correctness, width=bar_width, color='red', label='Correctness')

    # Labels and title
    ax.set_ylabel('Percentage')
    ax.set_title('Completeness vs. Correctness Percentages)')

    # Correctly center tick labels under each group
    ax.set_xticks(group_centers)
    ax.set_xticklabels(ordered_keys, rotation=45, ha='center')  # <== ha='center' is key here

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()

    # Vertical separator after 'average'
    separator_x = group_centers[0] + bar_width + group_spacing / 2
    ax.axvline(separator_x, color='gray', linestyle='--', linewidth=1)

    # Optional: Horizontal grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    os.makedirs("diagrams", exist_ok=True)
    plt.savefig("diagrams/non_smelly_completeness_correctness_percentages.png", dpi=300, bbox_inches='tight')


def create_non_smelly_reason_counts(evals_info):
    reasons_dict = {
        'completeness_reasons': {},
        'correctness_reasons': {},
    }

    # Next 3 for loops goes through all evaluations
    for game, results in evals_info.items():
        for variant_name, variant_results in results.items():
            for req_id_str, req_result in variant_results.items():
                
                reason_types = ['completeness_reasons', 'correctness_reasons']
                for reason_type in reason_types:
                    if reason_type in req_result and req_result[reason_type]:
                        for reason in req_result[reason_type]:
                            if reason in reasons_dict[reason_type]:
                                reasons_dict[reason_type][reason] = reasons_dict[reason_type][reason] + 1
                            else:
                                reasons_dict[reason_type][reason] = 1
    
    # Plot and save diagrams
    plot_top_10_reasons(reasons_dict['completeness_reasons'], "Top 10 Completeness Reasons", "completeness_reasons.png")
    plot_top_10_reasons(reasons_dict['correctness_reasons'], "Top 10 Correctness Reasons", "correctness_reasons.png")


def plot_top_10_reasons(reason_dict, title, filename):
    # Sort and get top 10
    sorted_items = sorted(reason_dict.items(), key=lambda x: x[1], reverse=True)[:10]
    labels, values = zip(*sorted_items)

    plt.figure(figsize=(8, 6))
    plt.bar(labels, values, color='skyblue')
    plt.ylim(0, 150)
    plt.yticks(range(0, 151, 10))  # Set y-axis ticks at every 10 units

    plt.grid(axis='y', linestyle=':', linewidth=1, alpha=0.7)  # Horizontal dotted lines
    plt.title(title)
    plt.xlabel('Reason Code')
    plt.ylabel('Count')
    plt.tight_layout()

    os.makedirs("diagrams", exist_ok=True)
    plt.savefig(os.path.join("diagrams", filename), dpi=300)
    plt.close()