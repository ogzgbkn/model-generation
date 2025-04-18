import os
import math
import matplotlib.pyplot as plt
import numpy as np

def generate_rq2_diagrams(evals_info):
    create_score_percentages(evals_info, 'completeness_correctness_percentages_smell_based')
    create_score_percentages(evals_info, 'completeness_correctness_percentages_smell_based_no_2_7', no_completeness_evals_incl = False, context = 'Implemented Requirements Only')


def create_score_percentages(evals_info, file_name, no_completeness_evals_incl = True, context = None):
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
    
    os.makedirs("diagrams/rq2", exist_ok=True)
    plt.savefig(f"diagrams/rq2/{file_name}", dpi=300, bbox_inches='tight')