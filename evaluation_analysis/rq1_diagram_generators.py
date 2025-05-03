import os
import matplotlib.pyplot as plt
import csv
import numpy as np

from helpers import get_num_of_requirements
from general_diagram_generators import create_reason_counts, calc_normalized_reason_counts_general, plot_top_10_reasons


def generate_rq1_diagrams(evals_info):
    create_non_smelly_score_percentages(evals_info)
    create_reason_counts(evals_info, calc_normalized_reason_counts_no_smells, context = 'Non-smelly Variants Only', requested_variants = ['no_smells'], directory = 'rq1')
    create_non_smelly_score_percentages_both_comp_corr_true(evals_info)
    # Also creates tolerated reason counts and tolerated comp and corr true
    create_non_smelly_score_percentages_tolerated(evals_info)


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


def create_non_smelly_score_percentages_both_comp_corr_true(evals_info):
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
    chart_title = 'Comp. and Corr. Both True Percentages (Non-Smelly Variants Only)'
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
    
    os.makedirs("evaluation_analysis/diagrams/rq1", exist_ok=True)
    plt.savefig("evaluation_analysis/diagrams/rq1/completeness_correctness_percentages_both_true.png", dpi=300, bbox_inches='tight')


def calc_normalized_reason_counts_no_smells(reasons_dict):
    return calc_normalized_reason_counts_general(reasons_dict, all_evals = False)


def create_non_smelly_score_percentages_tolerated(evals_info):
    comp_corr_one_percentages_dict = {}
    total_comp_corr_one_percentage = 0

    percentages_dict = {}
    reasons_dict = {
        'completeness_reasons': {},
        'correctness_reasons': {},
    }

    tolerated_completeness_reasons = ['1.3', '1.3.1', '1.4.1', '1.5.1', '1.6', '1.6.1']
    tolerated_completeness_reasons_set = set(tolerated_completeness_reasons)
    tolerated_correctness_reasons = ['2.3', '2.3.1', '2.3.2', '2.3.3', '2.4.1', '2.5.1', '2.6', '2.6.1']
    tolerated_correctness_reasons_set = set(tolerated_correctness_reasons)

    total_completeness_percentage = 0
    total_correctness_percentage = 0
    for game, results in evals_info.items():
        total_comp_tol_count = 0
        total_corr_tol_count = 0
        game_req_count = get_num_of_requirements(game)
        game_tolerated_comp_total = 0
        game_tolerated_corr_total = 0
        game_no_smells_total_comp_corr_one = 0
        # Checking each no-smells variant
        for req_id_str, req_result in results['no_smells'].items():
            if req_id_str != 'total':
                tolerate_completeness = False
                tolerate_correctness = False
                
                # Toleration of less impactful smells
                eval_completeness_reasons_set = set(req_result['completeness_reasons'])
                eval_correctness_reasons_set = set(req_result['correctness_reasons'])
                # The sets should not be empty
                if eval_completeness_reasons_set and eval_completeness_reasons_set.issubset(tolerated_completeness_reasons_set):
                    tolerate_completeness = True
                if eval_correctness_reasons_set and eval_correctness_reasons_set.issubset(tolerated_correctness_reasons_set):
                    tolerate_correctness = True
                # If completeness can be tolerated and correctness only has 2.7, correctness can also be tolerated
                elif req_result['correctness_reasons'] == ['2.7'] and tolerate_completeness:
                    tolerate_correctness = True

                if tolerate_completeness:
                    total_comp_tol_count = total_comp_tol_count + 1
                if tolerate_correctness:
                    total_corr_tol_count = total_corr_tol_count + 1

                # For calculating the tolerated no smelly reason percentages
                reason_types = ['completeness_reasons', 'correctness_reasons']
                for reason_type in reason_types:
                    # Skipping the tolerated reasons
                    if (reason_type == 'completeness_reasons' and tolerate_completeness) or (reason_type == 'correctness_reasons' and tolerate_correctness):
                        continue
                    if reason_type in req_result and req_result[reason_type]:
                        for reason in req_result[reason_type]:
                            if reason in reasons_dict[reason_type]:
                                reasons_dict[reason_type][reason] = reasons_dict[reason_type][reason] + 1
                            else:
                                reasons_dict[reason_type][reason] = 1

                completeness_to_aggregate = 1 if tolerate_completeness else req_result['completeness']
                correctness_to_aggregate = 1 if tolerate_correctness else req_result['correctness']

                # For generating comp corr one tolerated percentages
                if completeness_to_aggregate == 1 and correctness_to_aggregate == 1:
                    game_no_smells_total_comp_corr_one = game_no_smells_total_comp_corr_one + 1
                
                game_tolerated_comp_total = game_tolerated_comp_total + completeness_to_aggregate
                game_tolerated_corr_total = game_tolerated_corr_total + correctness_to_aggregate

        print(f"Game: {game}, comp tol count: {total_comp_tol_count}, corr tol count: {total_corr_tol_count}")
        
        # For generating comp corr one tolerated percentages
        comp_corr_one_percentage = (int(game_no_smells_total_comp_corr_one) / game_req_count) * 100
        comp_corr_one_percentages_dict[game] = {
            'comp_corr_percentage': comp_corr_one_percentage,
        }
        total_comp_corr_one_percentage = total_comp_corr_one_percentage + comp_corr_one_percentage
        
        completeness_percentage = (int(game_tolerated_comp_total) / game_req_count) * 100
        correctness_percentage = (int(game_tolerated_corr_total) / game_req_count) * 100
        percentages_dict[game] = {
            'completeness_percentage': completeness_percentage,
            'correctness_percentage': correctness_percentage,
        }
        total_completeness_percentage = total_completeness_percentage + completeness_percentage
        total_correctness_percentage = total_correctness_percentage + correctness_percentage
    
    avg_comp_corr = total_comp_corr_one_percentage / 5
    comp_corr_one_percentages_dict['average'] = {
        'comp_corr_percentage': avg_comp_corr
    }

    create_non_smelly_tolerated_comp_corr_one(comp_corr_one_percentages_dict)
    
    avg_completeness = total_completeness_percentage / 5
    avg_correctness = total_correctness_percentage / 5
    percentages_dict['average'] = {
        'completeness_percentage': avg_completeness,
        'correctness_percentage': avg_correctness,
    }

    normalized_reasons_dict = calc_normalized_reason_counts_general(reasons_dict, all_evals = False)

    # Plot and save diagrams
    plot_top_10_reasons(normalized_reasons_dict['completeness_reasons'], "Top 10 Completeness Reasons (Non-smelly Variants Only)", "completeness_reasons_tolerated.png", 'rq1')
    plot_top_10_reasons(normalized_reasons_dict['correctness_reasons'], "Top 10 Correctness Reasons (Non-smelly Variants Only)", "correctness_reasons_tolerated.png", 'rq1')

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
    plt.savefig("evaluation_analysis/diagrams/rq1/completeness_correctness_percentages_tolerated.png", dpi=300, bbox_inches='tight')


def create_non_smelly_tolerated_comp_corr_one(percentages_dict):
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
    chart_title = 'Comp. and Corr. Both True Percentages (Non-Smelly Variants Only)'
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
    
    os.makedirs("evaluation_analysis/diagrams/rq1", exist_ok=True)
    plt.savefig("evaluation_analysis/diagrams/rq1/completeness_correctness_percentages_tolerated_both_true.png", dpi=300, bbox_inches='tight')