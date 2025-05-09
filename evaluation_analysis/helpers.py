import os
import csv
import math
import copy

def get_num_of_requirements(game_name):
    games = {
        'dice_game': 25,
        'arkanoid': 19,
        'snake': 14,
        'scopa': 16,
        'pong': 20
    }
    return games[game_name]


class Reason:
    def __init__(self):
        self.non_critical_completeness_reasons = ['1.3.1', '1.4.1', '1.5.1', '1.6', '1.6.1']
        self.non_critical_correctness_reasons = ['2.3.1', '2.3.2', '2.4.1', '2.5.1', '2.6.1']
        self.reason_counts = {
            'dice_game': {},
            'arkanoid': {},
            'snake': {},
            'scopa': {},
            'pong': {}
        }

        current_file_path = os.path.abspath(__file__)
        # Directory containing the file (./model-generation/evaluation_analysis)
        current_dir = os.path.dirname(current_file_path)
        # Path to your CSV file
        csv_file_path = os.path.join(current_dir, 'games_component_counts.csv')

        # Open the CSV file
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            # If the CSV has a header and you want to skip it
            header = next(reader, None)
            # Process each row
            game = None
            for row_number, row in enumerate(reader, start=1):
                for col_index, value in enumerate(row):
                    if row[0] in self.reason_counts:
                        game = row[0]
                        if col_index != 0:
                            reasons = header[col_index]
                            reasons_list = reasons.split(',')
                            for reason in reasons_list:
                                self.reason_counts[game][reason] = int(value)

    def calc_critical_and_non_critical_percentages(self, evals_info, games = [], variants = []):
        inner_dict = {
            'critical_reasons': {},
            'non_critical_reasons': {},
        }
        counts_dict = {
            'dice_game': copy.deepcopy(inner_dict),
            'arkanoid': copy.deepcopy(inner_dict),
            'snake': copy.deepcopy(inner_dict),
            'scopa': copy.deepcopy(inner_dict),
            'pong': copy.deepcopy(inner_dict)
        }
        reason_types = ['completeness_reasons', 'correctness_reasons']
        # Next 3 for loops goes through all evaluations
        # Finding counts of critical and non-critical reasons by games
        for game, results in evals_info.items():
            if games and game not in games:
                    continue
            for variant_name, variant_results in results.items():
                if variants and variant_name not in variants:
                    continue
                for req_id_str, req_result in variant_results.items():
                    for reason_type in reason_types:
                        if reason_type in req_result and req_result[reason_type]:
                            for reason in req_result[reason_type]:
                                if reason_type == 'completeness_reasons':
                                    non_critical_list = self.non_critical_completeness_reasons
                                else:
                                    non_critical_list = self.non_critical_correctness_reasons
                                if reason in non_critical_list:
                                    critical_type = 'non_critical_reasons'
                                else:
                                    critical_type = 'critical_reasons'
                                # 2.7 can be critical or non critical depending on the completeness reasons
                                if reason == '2.7' and 'correctness_reasons' in req_result and req_result['correctness_reasons'] == ['2.7']:
                                    if 'completeness_reasons' in req_result and req_result['completeness_reasons']:
                                        completeness_reasons_set = set(req_result['completeness_reasons'])
                                        if completeness_reasons_set.issubset(set(self.non_critical_completeness_reasons)):
                                            critical_type = 'non_critical_reasons'
                                counts_dict[game][critical_type][reason] = counts_dict[game][critical_type][reason] + 1 if reason in counts_dict[game][critical_type] else 1

        inner_percentage_dict = {
            'critical_reasons_percentage': 0,
            'non_critical_reasons_percentage': 0,
        }
        percentages_dict = {
            'dice_game': copy.deepcopy(inner_percentage_dict),
            'arkanoid': copy.deepcopy(inner_percentage_dict),
            'snake': copy.deepcopy(inner_percentage_dict),
            'scopa': copy.deepcopy(inner_percentage_dict),
            'pong': copy.deepcopy(inner_percentage_dict)
        }
        critical_types = ['critical_reasons', 'non_critical_reasons']
        
        # Normalization and summation of counts
        for game, percentages in percentages_dict.items():
            for critical_type in critical_types:
                reason_counts = counts_dict[game][critical_type]
                total_normalized_reasons_count = 0
                if reason_counts:
                    for reason, count in reason_counts.items():
                        reason_prefix = reason[0:3]
                        normalized_reason_count = count / self.reason_counts[game][reason_prefix]
                        total_normalized_reasons_count += normalized_reason_count
                percentages[f"{critical_type}_percentage"] = total_normalized_reasons_count
            
            sum = percentages_dict[game]['critical_reasons_percentage'] + percentages_dict[game]['non_critical_reasons_percentage']
            percentages_dict[game]['critical_reasons_percentage'] = (percentages_dict[game]['critical_reasons_percentage'] / sum) * 100
            percentages_dict[game]['non_critical_reasons_percentage'] = (percentages_dict[game]['non_critical_reasons_percentage'] / sum) * 100

        total_critical_reasons_percentage = 0
        total_non_critical_reasons_percentage = 0
        for game, percentages in percentages_dict.items():
            total_critical_reasons_percentage += percentages['critical_reasons_percentage']
            total_non_critical_reasons_percentage += percentages['non_critical_reasons_percentage']

        percentages_dict['average'] = {
            'critical_reasons_percentage': total_critical_reasons_percentage / 5,
            'non_critical_reasons_percentage': total_non_critical_reasons_percentage / 5,
        }

        return percentages_dict