import os
import matplotlib.pyplot as plt
import csv
import numpy as np

from helpers import get_num_of_requirements, Reason
from general_diagram_generators import create_reason_counts, calc_normalized_reason_counts_general, plot_top_10_reasons


def generate_rq1_diagrams(evals_info):
    create_non_smelly_score_percentages(evals_info)
    create_reason_counts(evals_info, calc_normalized_reason_counts_no_smells, context = 'Non-smelly Variants Only', requested_variants = ['no_smells'], directory = 'rq1')
    create_non_smelly_score_percentages_both_comp_corr_true(evals_info, directory = 'rq1', file_name = 'completeness_correctness_percentages_both_true.png', context = 'Non-smelly Variants Only')
    create_non_smelly_critical_and_non_critical_reasons_percentage(evals_info)


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
    ax.set_title('Completeness vs. Correctness Percentages (Non-smelly Variants Only)')

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
    
    os.makedirs("evaluation_analysis/diagrams/rq1", exist_ok=True)
    plt.savefig("evaluation_analysis/diagrams/rq1/completeness_correctness_percentages.png", dpi=300, bbox_inches='tight')


def create_non_smelly_score_percentages_both_comp_corr_true(evals_info, directory = '', file_name = '', context = None):
    percentages_dict = {}
    total_comp_corr_percentage = 0
    for game, results in evals_info.items():
        game_req_count = get_num_of_requirements(game)
        game_no_smells_total_comp_corr_one = 0
        # Checking each no-smells variant
        for req_id_str, req_result in results['no_smells'].items():
            if req_id_str != 'total':
                if req_result['completeness'] == 1 and req_result['correctness'] == 1:
                    game_no_smells_total_comp_corr_one = game_no_smells_total_comp_corr_one + 1
        comp_corr_one_percentage = (int(game_no_smells_total_comp_corr_one) / game_req_count) * 100
        percentages_dict[game] = {
            'comp_corr_percentage': comp_corr_one_percentage,
        }
        total_comp_corr_percentage = total_comp_corr_percentage + comp_corr_one_percentage
    
    avg_comp_corr = total_comp_corr_percentage / 5
    percentages_dict['average'] = {
        'comp_corr_percentage': avg_comp_corr
    }

    # Ordered keys
    ordered_keys = ['average', 'dice_game', 'arkanoid', 'snake', 'scopa', 'pong']
    comp_and_corr = [percentages_dict[k]["comp_corr_percentage"] for k in ordered_keys]

    # Plot config
    bar_width = 0.6
    group_spacing = 0.4
    num_groups = len(ordered_keys)
    x_positions = np.arange(num_groups)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot bars
    ax.bar(x_positions, comp_and_corr, width=bar_width, color='red', label='Completeness and Correctness')

    # Labels and title
    ax.set_ylabel('Percentage')
    if context:
        chart_title = f'Comp. and Corr. Both True Percentages ({context})'
    else:
        chart_title = 'Comp. and Corr. Both True Percentages (All Evaluations)'
    ax.set_title(chart_title)

    # Correctly center tick labels under each group
    ax.set_xticks(x_positions)
    ax.set_xticklabels(ordered_keys, rotation=45, ha='center')  # <== ha='center' is key here

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()

    # Optional: Horizontal grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    os.makedirs(f"evaluation_analysis/diagrams/{directory}", exist_ok=True)
    plt.savefig(f"evaluation_analysis/diagrams/{directory}/{file_name}", dpi=300, bbox_inches='tight')


def calc_normalized_reason_counts_no_smells(reasons_dict):
    return calc_normalized_reason_counts_general(reasons_dict, all_evals = False)


def create_non_smelly_critical_and_non_critical_reasons_percentage(evals_info):
    reason = Reason()
    reasons_percentages = reason.calc_critical_and_non_critical_percentages(evals_info, variants = ['no_smells'])

    # Ordered keys
    ordered_keys = ['average', 'dice_game', 'arkanoid', 'snake', 'scopa', 'pong']
    critical_reasons_percentages = [reasons_percentages[k]["critical_reasons_percentage"] for k in ordered_keys]
    non_critical_reasons_percentages = [reasons_percentages[k]["non_critical_reasons_percentage"] for k in ordered_keys]

    # Plot config
    bar_width = 0.35
    group_spacing = 0.4
    num_groups = len(ordered_keys)

    # Compute X positions for group centers
    group_centers = np.arange(num_groups) * (2 * bar_width + group_spacing)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(group_centers - bar_width / 2, critical_reasons_percentages, width=bar_width, color='red', label='Critical')
    ax.bar(group_centers + bar_width / 2, non_critical_reasons_percentages, width=bar_width, color='green', label='Non-critical')

    # Labels and title
    ax.set_ylabel('Percentage')
    ax.set_title('Critical vs. Non-Critical Reasons Percentages (Non-smelly Variants Only)')

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
    
    os.makedirs("evaluation_analysis/diagrams/rq1", exist_ok=True)
    plt.savefig("evaluation_analysis/diagrams/rq1/critical_non_critical_reasons_percentages.png", dpi=300, bbox_inches='tight')