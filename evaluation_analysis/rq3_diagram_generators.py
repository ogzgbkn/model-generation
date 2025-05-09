import os
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
# plt.set_loglevel('debug')
import numpy as np
import pandas as pd


def generate_rq3_diagrams(evals_info):
    """ scores_dict = extract_smell_type_based_info(evals_info)
    scores_dict_without_2_7 = extract_smell_type_based_info(evals_info, no_completeness_evals_incl = False)
    create_smell_type_based_score_percentages(scores_dict, 'completeness_correctness_percentages_smell_type_based')
    create_smell_type_based_score_percentages(scores_dict_without_2_7,
                                              'completeness_correctness_percentages_smell_type_based_no_2_7',
                                              'Generated Requirements Only')
    create_smell_sub_type_based_score_percentages(scores_dict, 'completeness_correctness_percentages_smell_sub_type_based')
    create_smell_sub_type_based_score_percentages(scores_dict_without_2_7,
                                                  'completeness_correctness_percentages_smell_sub_type_based_no_2_7',
                                                  'Generated Requirements Only') """
    create_statistical_test(evals_info)


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
            labels, comp_values, corr_values, smell_counts = calculate_percentages(smell_type, subtypes, scores_dict['non-smelly'])

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
            separator_x = group_centers[1] + bar_width + group_spacing / 2
            ax.axvline(separator_x, color='gray', linestyle='--', linewidth=1)

            # Optional: Horizontal grid
            ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

            plt.tight_layout()
            
            os.makedirs("evaluation_analysis/diagrams/rq3", exist_ok=True)
            plt.savefig(f"evaluation_analysis/diagrams/rq3/{smell_type}_{file_name}", dpi=300, bbox_inches='tight')


def calculate_percentages(smell_type, group_dict, non_smelly_scores):
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

    # Inserting non-smelly bars to the very beginning as success criteria
    labels.insert(0, 'non-smelly')
    smell_counts.insert(0, non_smelly_scores['count'])
    completeness.insert(0, non_smelly_scores['completeness'] / non_smelly_scores['count'] * 100)
    correctness.insert(0, non_smelly_scores['correctness'] / non_smelly_scores['count'] * 100)

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


def create_statistical_test(evals_info):

    # Sample data creation
    smell_types = ["lexical", "semantic", "syntactic", ""]
    subtypes = {
        "lexical": ["subjective_language", "optional_parts", "weak_verbs"],
        "semantic": ["logical_inconsistencies", "numerical_discrepancies", "ambiguities"],
        "syntactic": ["passive_voice", "negative", "vague_pronouns"],
        "": [""]
    }
    records = []

    for game, variants in evals_info.items():
        for variant, requirements in variants.items():
            for req_id, details in requirements.items():
                # Considering only generated requirements
                if req_id != 'total' and details['correctness_reasons'] != ['2.7']:
                    record = {
                        'game': game,
                        'variant': variant,
                        'req_id': req_id,
                        'smell_type': details['smell_type'],
                        'smell_sub_type': details['smell_sub_type'],
                        'completeness': details['completeness'],
                        'correctness': details['correctness']
                    }
                    records.append(record)

    df = pd.DataFrame(records)
    
    # Run the analysis
    print("Analyzing relationship between smells and quality metrics...")
    results = analyze_smells(df)

    # Create interpretable table
    interpretation_table = create_interpretable_table(results)
    # print("\nDetailed Interpretation Table:")
    pd.set_option('display.max_colwidth', None)  # Display full text in columns
    # print(interpretation_table[["Test", "Metric", "p-value", "Cramer's V", "Effect Size", "Interpretation"]])
    
    # Save the interpretation table to CSV for better viewing
    interpretation_table.to_csv("smell_analysis_results.csv", index=False)
    # print("\nDetailed results saved to 'smell_analysis_results.csv'")

    pass
    
    """ # Display results
    results_table = display_results(results)
    print("\nResults Summary:")
    print(results_table)
    
    # Create a more readable summary table
    summary = pd.pivot_table(
        results_table, 
        values=["p-value", "Cramer's V"], 
        index=["Test", "Type"], 
        columns=["Metric"]
    )
    
    print("\nSummary Table:")
    print(summary)
    
    # Plot results
    plot_results(results, "completeness")
    plot_results(results, "correctness")
    
    # Print detailed contingency tables for significant findings
    print("\nDetailed Contingency Tables for Significant Findings:")
    for r in results:
        if r["p_value"] < 0.05:
            print(f"\n{r['test_name']} - {r['metric'].capitalize()}")
            print(f"p-value: {r['p_value']:.4f}, Cramer's V: {r['cramers_v']:.4f}")
            print(r["contingency"]) """

    """ records = []
    for game, variants in evals_info.items():
        for variant, requirements in variants.items():
            for req_id, details in requirements.items():
                # Considering only generated requirements
                if req_id != 'total' and details['correctness_reasons'] != ['2.7']:
                    record = {
                        'game': game,
                        'variant': variant,
                        'req_id': req_id,
                        'smell_type': details['smell_type'],
                        'smell_sub_type': details['smell_sub_type'],
                        'completeness': details['completeness'],
                        'correctness': details['correctness']
                    }
                    records.append(record)

    df = pd.DataFrame(records)

    df['has_smell']       = df['smell_type'] != ''
    df['lexical_flag']    = df['smell_type'] == 'lexical'
    df['syntactic_flag']  = df['smell_type'] == 'syntactic'
    df['semantic_flag']   = df['smell_type'] == 'semantic'

    cont_comp_smell = pd.crosstab(df['has_smell'], df['completeness'])
    # Rows: False (clean) / True (smelly)
    # Columns: 0 (incomplete) / 1 (complete)

    cont_corr_smell = pd.crosstab(df['has_smell'], df['correctness'])
    # Same layout: 0 = incorrect, 1 = correct

    # Completeness vs. any smell
    chi2_c, p_c, _, expected_c = chi2_contingency(cont_comp_smell)

    # Correctness vs. any smell
    chi2_k, p_k, _, expected_k = chi2_contingency(cont_corr_smell)

    print(f"Completeness vs. smell: p = {p_c:.3f}")
    print(f"Correctness vs. smell:  p = {p_k:.3f}") """


def analyze_smells(df):
    """
    Analyze all smell types and subtypes in relation to completeness and correctness
    """
    results = []
    
    # Get unique smell types and subtypes
    smell_types = [type_name for type_name in df['smell_type'].unique() if type_name != ""]
    
    # Tests to run
    tests = [
        # Any smell vs no smell
        {"name": "Has any smell", "col": "smell_type", "val": "", "type": "overall"},
        
        # Individual smell types
        *[{"name": f"Smell type: {s_type}", "col": "smell_type", "val": s_type, "type": "type"} 
          for s_type in smell_types],
        
        # Subtypes for each smell type
        *[{"name": f"Subtype: {subtype}", "col": "smell_sub_type", "val": subtype, "type": "subtype"}
          for smell_type in smell_types
          for subtype in df[df['smell_type'] == smell_type]['smell_sub_type'].unique() 
          if subtype != ""]
    ]
    
    # Run tests for both completeness and correctness
    for metric in ['completeness', 'correctness']:
        for test in tests:
            if test["type"] == "overall":
                # For overall test, we're checking if having ANY smell affects scores
                # So we invert the condition (smell_type = "" means no smell)
                modified_df = create_valid_df(df, test["col"], test["val"], negate = True)
                result = run_chi_squared_test(modified_df, test["col"], test["val"], metric, negate = True)
            else:
                modified_df = create_valid_df(df, test["col"], test["val"])
                result = run_chi_squared_test(modified_df, test["col"], test["val"], metric)
            
            results.append({
                "test_name": test["name"],
                "test_type": test["type"],
                "metric": metric,
                "chi_squared": result["chi2"],
                "p_value": result["p_value"],
                "cramers_v": result["cramers_v"],
                "contingency": result["contingency"]
            })
    
    return results


def run_chi_squared_test(df, condition_col, condition_value, score_col, negate=False):
    """
    Run chi-squared test to check if having a particular smell type affects scores
    
    Parameters:
    df (DataFrame): The data
    condition_col (str): Column to check condition (e.g., 'smell_type')
    condition_value (str or list): Value(s) to check in condition_col
    score_col (str): Column with scores to analyze ('completeness' or 'correctness')
    negate (bool): Whether to negate the condition (for testing "any smell" vs "no smell")
    
    Returns:
    dict: Results containing chi2, p-value, cramer's v, and contingency table
    """
    # Create condition based on input
    if negate:
        # For "Has any smell" test (smell_type != "")
        condition = df[condition_col] != condition_value
        label_true = "Has Smell"
        label_false = "No Smell"
    elif isinstance(condition_value, list):
        # For multiple possible values
        condition = df[condition_col].isin(condition_value)
        label_true = f"Has {condition_value}"
        label_false = f"No {condition_value}"
    else:
        # For specific smell type
        condition = df[condition_col] == condition_value
        label_true = f"Has {condition_value}" if condition_value else "No Smell"
        label_false = f"No {condition_value}" if condition_value else "Has Smell"
    
    # df_modified = create_valid_df(df, condition_col, condition_value, negate)

    # Create contingency table
    contingency = pd.crosstab(condition, df[score_col] == 0)
    
    # Rename for clarity
    contingency.index = [label_true, label_false]
    contingency.columns = ["Score = 1", "Score = 0"]
    
    # Calculate chi-squared test
    chi2, p, dof, expected = chi2_contingency(contingency)

    if condition_value == 'ambiguities':
        print(contingency)

    # Minimum dimension minus 1
    n = contingency.values.sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))

    return {
        "chi2": chi2,
        "p_value": p,
        "dof": dof,
        "cramers_v": cramers_v,
        "contingency": contingency
    }


def create_valid_df(df, condition_col, condition_value, negate = False):
    """
    Filter df for chi-squared testing.

    If negate=True:
        return df unchanged.

    If negate=False:
        keep only rows where
          - df[condition_col] == condition_value
            (the specific smell type you want to test)
        OR
          - df[condition_col] == "" 
            (non-smelly evaluations)
    """
    if negate:
        # test "any smell vs. no smell": use the full data
        return df.copy()

    # otherwise: only the target smell vs truly non-smelly
    mask_target   = df[condition_col] == condition_value
    mask_non_smell = df['smell_type'] == ""
    filtered_df = df[mask_target | mask_non_smell].copy()
    return filtered_df


def create_interpretable_table(results):
    """
    Create a table that interprets statistical results in plain language
    
    Parameters:
    results (list): Results from analyze_smells function
    
    Returns:
    DataFrame: Formatted table with statistical interpretations
    """
    interpretation_data = []
    
    for r in results:
        # Determine statistical significance
        if r["p_value"] < 0.01:
            significance = "Strong evidence"
            sig_symbol = "***"
        elif r["p_value"] < 0.05:
            significance = "Evidence"
            sig_symbol = "**"
        elif r["p_value"] < 0.1:
            significance = "Weak evidence"
            sig_symbol = "*"
        else:
            significance = "No evidence"
            sig_symbol = ""
        
        # Determine effect size based on Cramer's V
        if r["cramers_v"] < 0.1:
            effect = "Negligible"
        elif r["cramers_v"] < 0.3:
            effect = "Small"
        elif r["cramers_v"] < 0.5:
            effect = "Medium"
        else:
            effect = "Large"
        
        # Get percentages from contingency table
        # For rows where condition is True (Has Smell or Has specific smell type)
        try:
            total_with_condition = r["contingency"].iloc[0].sum()
            zeros_with_condition = r["contingency"].iloc[0, 1]
            percent_zeros_with = round((zeros_with_condition / total_with_condition) * 100, 1) if total_with_condition > 0 else 0
            
            total_without_condition = r["contingency"].iloc[1].sum()
            zeros_without_condition = r["contingency"].iloc[1, 1]
            percent_zeros_without = round((zeros_without_condition / total_without_condition) * 100, 1) if total_without_condition > 0 else 0
            
            # Calculate difference in percentages
            diff = percent_zeros_with - percent_zeros_without
        except:
            percent_zeros_with = "N/A"
            percent_zeros_without = "N/A"
            diff = "N/A"
        
        # Create interpretation
        if r["p_value"] < 0.05:
            # For smells, positive diff means more zeros (worse scores) which is the expected direction
            if diff > 0:
                direction = f"increases zeros by {abs(diff):.1f}% (negative impact)"
            else:
                direction = f"unexpectedly decreases zeros by {abs(diff):.1f}% (positive impact)"
            
            interpretation = f"{significance} ({sig_symbol}) that {r['test_name']} {direction}. Effect size: {effect} ({r['cramers_v']:.3f})"
        else:
            interpretation = f"No statistically significant effect on {r['metric']} scores (p={r['p_value']:.3f})"
        
        interpretation_data.append({
            "Test": r["test_name"],
            "Metric": r["metric"].capitalize(),
            "p-value": r["p_value"],
            "Significant": "Yes" if r["p_value"] < 0.05 else "No",
            "Cramer's V": r["cramers_v"],
            "Effect Size": effect,
            "% Zeros (With)": percent_zeros_with,
            "% Zeros (Without)": percent_zeros_without,
            "Difference": diff if isinstance(diff, (int, float)) else diff,
            "Impact": "Negative" if isinstance(diff, (int, float)) and diff > 0 else 
                    ("Positive" if isinstance(diff, (int, float)) and diff < 0 else "None"),
            "Interpretation": interpretation
        })
    
    # Create DataFrame
    df_interpretation = pd.DataFrame(interpretation_data)
    
    # Sort by test type and significance
    test_type_order = {"Has any smell": 0, "Smell type:": 1, "Subtype:": 2}
    df_interpretation["Type Order"] = df_interpretation["Test"].apply(
        lambda x: next((order for key, order in test_type_order.items() if x.startswith(key)), 3)
    )
    
    df_interpretation = df_interpretation.sort_values(
        by=["Type Order", "p-value"]
    ).drop(columns=["Type Order"])
    
    return df_interpretation