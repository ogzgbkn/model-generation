import os
import math
import matplotlib.pyplot as plt
import numpy as np

from helpers import Reason

def generate_rq2_diagrams(evals_info):
    create_score_percentages(evals_info, 'completeness_correctness_percentages_smell_based')
    create_score_percentages(evals_info, 'completeness_correctness_percentages_smell_based_no_2_7', no_completeness_evals_incl = False, context = 'Generated Requirements Only')
    create_all_smelly_critical_and_non_critical_reasons_percentage(evals_info)

def create_score_percentages(evals_info, file_name, no_completeness_evals_incl = True, context = None):
    comp_and_corr_scores_dict = {
        'smelly': {
            'comp_and_corr': 0,
            'count': 0
        },
        'non-smelly': {
            'comp_and_corr': 0,
            'count': 0
        }
    }
    
    scores_dict = {
        'smelly': {
            'completeness': 0,
            'correctness': 0,
            'count': 0
        },
        'non-smelly': {
            'completeness': 0,
            'correctness': 0,
            'count': 0
        }
    }

    for game, results in evals_info.items():
        for variant_name, variant_results in results.items():
            for req_id_str, req_result in variant_results.items():
                if req_id_str != 'total':
                    # No completeness evals deactivated case
                    if not no_completeness_evals_incl:
                        if '2.7' in req_result['correctness_reasons']:
                            continue
                    evaluation_type = None
                    # Smelly evaluation case
                    if req_result['smell_type']:
                        if req_result['smell_sub_type']:
                            evaluation_type = 'smelly'
                        else:
                            raise Exception('Smell sub type does not exist! Check the eval results.')
                    # Non-smelly evaluation case
                    else:
                        evaluation_type = 'non-smelly'
                    
                    scores_dict[evaluation_type]['completeness'] = scores_dict[evaluation_type]['completeness'] + req_result['completeness']
                    scores_dict[evaluation_type]['correctness'] = scores_dict[evaluation_type]['correctness'] + req_result['correctness']
                    scores_dict[evaluation_type]['count'] = scores_dict[evaluation_type]['count'] + 1

                    comp_and_corr_scores_dict[evaluation_type]['count'] += 1
                    if req_result['completeness'] == 1 and req_result['correctness'] == 1:
                        comp_and_corr_scores_dict[evaluation_type]['comp_and_corr'] += 1

    all_count = scores_dict['smelly']['count'] + scores_dict['non-smelly']['count']
    
    all_completeness = scores_dict['smelly']['completeness'] + scores_dict['non-smelly']['completeness']
    all_completeness_percentage = (all_completeness / all_count) * 100

    all_correctness = scores_dict['smelly']['correctness'] + scores_dict['non-smelly']['correctness']
    all_correctness_percentage = (all_correctness / all_count) * 100
    
    percentages_dict = {
        'all': {
            'completeness_percentage': all_completeness_percentage,
            'correctness_percentage': all_correctness_percentage
        },
        'non-smelly': {
            'completeness_percentage': (scores_dict['non-smelly']['completeness'] / scores_dict['non-smelly']['count']) * 100,
            'correctness_percentage': (scores_dict['non-smelly']['correctness'] / scores_dict['non-smelly']['count']) * 100
        },
        'smelly': {
            'completeness_percentage': (scores_dict['smelly']['completeness'] / scores_dict['smelly']['count']) * 100,
            'correctness_percentage': (scores_dict['smelly']['correctness'] / scores_dict['smelly']['count']) * 100
        }
    }

    comp_and_corr_all_count = comp_and_corr_scores_dict['smelly']['count'] + comp_and_corr_scores_dict['non-smelly']['count']
    all_comp_and_corr = comp_and_corr_scores_dict['smelly']['comp_and_corr'] + comp_and_corr_scores_dict['non-smelly']['comp_and_corr']
    all_comp_and_corr_percentage = (all_comp_and_corr / comp_and_corr_all_count) * 100
    
    comp_and_corr_percentages_dict = {
        'all': {
            'comp_and_corr': all_comp_and_corr_percentage
        },
        'non-smelly': {
            'comp_and_corr': (comp_and_corr_scores_dict['non-smelly']['comp_and_corr'] / comp_and_corr_scores_dict['non-smelly']['count']) * 100
        },
        'smelly': {
            'comp_and_corr': (comp_and_corr_scores_dict['smelly']['comp_and_corr'] / comp_and_corr_scores_dict['smelly']['count']) * 100
        }
    }

    create_percentages_diagram(percentages_dict, file_name, context)
    create_percentages_diagram_both_true(comp_and_corr_percentages_dict, f"{file_name}_both_true", context)


def create_percentages_diagram(percentages_dict, file_name, context = 'None'):
    # Ordered keys
    ordered_keys = ['all', 'non-smelly', 'smelly']
    completeness = [percentages_dict[k]["completeness_percentage"] for k in ordered_keys]
    correctness = [percentages_dict[k]["correctness_percentage"] for k in ordered_keys]
    
    # Plot config
    bar_width = 0.35
    group_spacing = 0.4
    num_groups = len(ordered_keys)

    # Compute X positions for group centers
    group_centers = np.arange(num_groups) * (2 * bar_width + group_spacing)

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot bars
    ax.bar(group_centers - bar_width / 2, completeness, width=bar_width, color='blue', label='Completeness')
    ax.bar(group_centers + bar_width / 2, correctness, width=bar_width, color='red', label='Correctness')

    # Labels and title
    ax.set_ylabel('Percentage')
    if context:
        chart_title = f'Completeness vs. Correctness Percentages ({context})'
    else:
        chart_title = 'Completeness vs. Correctness Percentages (All Evaluations)'
    ax.set_title(chart_title)

    # Correctly center tick labels under each group
    ax.set_xticks(group_centers)
    ax.set_xticklabels(ordered_keys, rotation=45, ha='center')  # <== ha='center' is key here

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()

    # Optional: Horizontal grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    os.makedirs("evaluation_analysis/diagrams/rq2", exist_ok=True)
    plt.savefig(f"evaluation_analysis/diagrams/rq2/{file_name}", dpi=300, bbox_inches='tight')


def create_percentages_diagram_both_true(percentages_dict, file_name, context = 'None'):
    # Ordered keys
    ordered_keys = ['all', 'non-smelly', 'smelly']
    comp_and_corr = [percentages_dict[k]["comp_and_corr"] for k in ordered_keys]
    
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
    
    os.makedirs("evaluation_analysis/diagrams/rq2", exist_ok=True)
    plt.savefig(f"evaluation_analysis/diagrams/rq2/{file_name}", dpi=300, bbox_inches='tight')


def create_all_smelly_critical_and_non_critical_reasons_percentage(evals_info):
    reason = Reason()
    reasons_percentages = reason.calc_critical_and_non_critical_percentages(evals_info, variants = ['all_smells'])

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
    ax.set_title('Critical vs. Non-Critical Reasons Percentages (All-smells Variants Only)')

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
    plt.savefig("evaluation_analysis/diagrams/rq2/critical_non_critical_reasons_percentages_all_smells_variants.png", dpi=300, bbox_inches='tight')