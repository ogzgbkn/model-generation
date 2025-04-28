import os
import math
import matplotlib.pyplot as plt
# plt.set_loglevel('debug')
import numpy as np

def generate_rq3_diagrams(evals_info):
    scores_dict = extract_smell_type_based_info(evals_info)
    scores_dict_without_2_7 = extract_smell_type_based_info(evals_info, no_completeness_evals_incl = False)
    create_smell_type_based_score_percentages(scores_dict, 'completeness_correctness_percentages_smell_type_based')
    create_smell_type_based_score_percentages(scores_dict_without_2_7,
                                              'completeness_correctness_percentages_smell_type_based_no_2_7',
                                              'Generated Requirements Only')
    create_smell_sub_type_based_score_percentages(scores_dict, 'completeness_correctness_percentages_smell_sub_type_based')
    create_smell_sub_type_based_score_percentages(scores_dict_without_2_7,
                                                  'completeness_correctness_percentages_smell_sub_type_based_no_2_7',
                                                  'Generated Requirements Only')


def extract_smell_type_based_info(evals_info, no_completeness_evals_incl = True):
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
        },
        'lexical': {
            'subjective_language': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'optional_parts': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'weak_verbs': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            }
        },
        'semantic': {
            'logical_inconsistencies': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'numerical_discrepancies': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'ambiguities': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            }
        },
        'syntactic': {
            'passive_voice': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'negative': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            },
            'vague_pronouns': {
                'completeness': 0,
                'correctness': 0,
                'count': 0
            }
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
                            current_scores = scores_dict[req_result['smell_type']][req_result['smell_sub_type']]
                            new_scores = {
                                'completeness': current_scores['completeness'] + req_result['completeness'],
                                'correctness': current_scores['correctness'] + req_result['correctness'],
                                'count': current_scores['count'] + 1
                            }
                            scores_dict[req_result['smell_type']][req_result['smell_sub_type']] = new_scores
                        else:
                            raise Exception('Smell sub type does not exist! Check the eval results.')
                    # Non-smelly evaluation case
                    else:
                        evaluation_type = 'non-smelly'
                    
                    scores_dict[evaluation_type]['completeness'] = scores_dict[evaluation_type]['completeness'] + req_result['completeness']
                    scores_dict[evaluation_type]['correctness'] = scores_dict[evaluation_type]['correctness'] + req_result['correctness']
                    scores_dict[evaluation_type]['count'] = scores_dict[evaluation_type]['count'] + 1

    return scores_dict


def create_smell_type_based_score_percentages(scores_dict, file_name, context = None):
    # Compute percentages
    ordered_keys = ['non-smelly', 'lexical', 'semantic', 'syntactic']
    completeness = []
    correctness = []

    # Non-smelly
    ns = scores_dict['non-smelly']
    if ns['count'] > 0:
        completeness.append(ns['completeness'] / ns['count'] * 100)
        correctness.append(ns['correctness'] / ns['count'] * 100)
    else:
        completeness.append(0)
        correctness.append(0)

    # Smelly types
    for category in ['lexical', 'semantic', 'syntactic']:
        c, k = compute_avg(scores_dict[category])
        completeness.append(c)
        correctness.append(k)
    
    if context:
        chart_title = f'Completeness vs. Correctness Percentages ({context})'
    else:
        chart_title = 'Completeness vs. Correctness Percentages (All Evaluations)'

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
    ax.set_title(chart_title)

    sample_sizes = aggregate_sample_sizes(scores_dict, ordered_keys)

    # Build new x-tick labels with sample size
    xtick_labels = [f"{key}\n(n={sample_sizes[key]})" for key in ordered_keys]

    # Correctly center tick labels under each group
    ax.set_xticks(group_centers)
    ax.set_xticklabels(xtick_labels, rotation=45, ha='center')  # <== ha='center' is key here

    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 10))
    ax.legend()

    # Vertical separator after 'non-smelly'
    separator_x = group_centers[0] + bar_width + group_spacing / 2
    ax.axvline(separator_x, color='gray', linestyle='--', linewidth=1)

    # Optional: Horizontal grid
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    
    os.makedirs("evaluation_analysis/diagrams/rq3", exist_ok=True)
    plt.savefig(f"evaluation_analysis/diagrams/rq3/{file_name}", dpi=300, bbox_inches='tight')


def compute_avg(group_dict):
    total_c, total_k, total_count = 0, 0, 0
    for sub in group_dict.values():
        total_c += sub['completeness']
        total_k += sub['correctness']
        total_count += sub['count']
    if total_count == 0:
        return 0, 0
    completeness_avg = total_c / total_count * 100
    correctness_avg = total_k / total_count * 100
    return completeness_avg, correctness_avg


def create_smell_sub_type_based_score_percentages(scores_dict, file_name, context = None):
    # Create one diagram per smell type
    for smell_type, subtypes in scores_dict.items():
        if smell_type in ['lexical', 'semantic', 'syntactic']:
            labels, comp_values, corr_values, smell_counts = calculate_percentages(smell_type, subtypes)

            if context:
                chart_title = f'Completeness vs. Correctness Percentages - {smell_type} ({context})'
            else:
                chart_title = f'Completeness vs. Correctness Percentages - {smell_type} (All Evaluations)'

            # Plot config
            bar_width = 0.35
            group_spacing = 0.4
            num_groups = len(labels)

            # Compute X positions for group centers
            group_centers = np.arange(num_groups) * (2 * bar_width + group_spacing)

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot bars
            ax.bar(group_centers - bar_width / 2, comp_values, width=bar_width, color='blue', label='Completeness')
            ax.bar(group_centers + bar_width / 2, corr_values, width=bar_width, color='red', label='Correctness')

            # Labels and title
            ax.set_ylabel('Percentage')
            ax.set_title(chart_title)

            # Correctly center tick labels under each group
            ax.set_xticks(group_centers)
            ax.set_xticklabels([item + f'\n(n={smell_counts[index]})' for index, item in enumerate(labels)], rotation=45, ha='center')  # <== ha='center' is key here

            ax.set_ylim(0, 100)
            ax.set_yticks(np.arange(0, 101, 10))
            ax.legend()

            # Vertical separator after '<smell_type>'
            separator_x = group_centers[0] + bar_width + group_spacing / 2
            ax.axvline(separator_x, color='gray', linestyle='--', linewidth=1)

            # Optional: Horizontal grid
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

            plt.tight_layout()
            
            os.makedirs("evaluation_analysis/diagrams/rq3", exist_ok=True)
            plt.savefig(f"evaluation_analysis/diagrams/rq3/{smell_type}_{file_name}", dpi=300, bbox_inches='tight')


def calculate_percentages(smell_type, group_dict):
    labels = []
    smell_counts = []
    completeness = []
    correctness = []
    total_c = total_k = total_count = 0

    for subtype, metrics in group_dict.items():
        count = metrics['count']
        if count == 0:
            comp = corr = 0
        else:
            comp = metrics['completeness'] / count * 100
            corr = metrics['correctness'] / count * 100

        labels.append(subtype)
        smell_counts.append(count)
        completeness.append(comp)
        correctness.append(corr)

        total_c += metrics['completeness']
        total_k += metrics['correctness']
        total_count += count

    # Type-level averages
    if total_count == 0:
        avg_comp = avg_corr = 0
    else:
        avg_comp = total_c / total_count * 100
        avg_corr = total_k / total_count * 100

    # Prepend type name and averages
    type_name = smell_type  # crude way, can be improved
    labels.insert(0, type_name)
    smell_counts.insert(0, total_count)
    completeness.insert(0, avg_comp)
    correctness.insert(0, avg_corr)

    return labels, completeness, correctness, smell_counts


def aggregate_sample_sizes(scores_dict, ordered_keys):
    sample_sizes = {}
    for key in ordered_keys:
        if 'count' in scores_dict[key]:
            sample_sizes[key] = scores_dict[key]['count']
        else:
            for sub_smell_type, value in scores_dict[key].items():
                if key not in sample_sizes:
                    sample_sizes[key] = value['count']
                else:
                    sample_sizes[key] = sample_sizes[key] + value['count']

    return sample_sizes