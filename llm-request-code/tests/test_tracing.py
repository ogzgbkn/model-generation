import pytest

from req_smell_tracing.tracing import Trace, Tracing, TracingColumn, TracingComparator


# fmt: off
class TestTracing:
    def test_parse_lines_str(self):
        assert Trace.parse_lines_str("1,2,3") == [1, 2, 3]
        assert Trace.parse_lines_str("1,2,3,") == [1, 2, 3]
        assert Trace.parse_lines_str("1-5,25-30") == [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30]
        assert Trace.parse_lines_str("1-5,25-30,") == [1, 2, 3, 4, 5, 25, 26, 27, 28, 29, 30]
        assert Trace.parse_lines_str("1,2,3,5-10") == [1, 2, 3, 5, 6, 7, 8, 9, 10]
        assert Trace.parse_lines_str("1-10,5-15") == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    def test_parse_implemented(self):
        true_values = ["true", "yes", "1", "True", "Yes", "YES", "TRUE"]
        false_values = ["false", "no", "0", "False", "No", "NO", "FALSE"]

        for value in true_values:
            assert Trace.parse_implemented(value) is True
        
        for value in false_values:
            assert Trace.parse_implemented(value) is False

    def test_from_csv(self):
        csv_str = (
        "\"requirement_id\",\"implemented\",\"lines_of_code\"\n"
        "\"1\",\"yes\",\"1,5,39\"\n"
        "\"2\",\"no\",\"5-10,7-12\"\n"
        "\"3\",\"yes\",\"7,\"\n"
        )
        columns = {
            TracingColumn.REQUIREMENT_ID: "requirement_id",
            TracingColumn.IMPLEMENTED: "implemented",
            TracingColumn.LINES_OF_CODE: "lines_of_code",
        }
        tracing = Tracing.from_csv(csv_str, columns)

        assert tracing.traces == [
            Trace(1, True, [1, 5, 39]),
            Trace(2, False, [5, 6, 7, 8, 9, 10, 11, 12]),
            Trace(3, True, [7]),
        ]

    def test_compare_linewise_precision_recall(self):
        tracing = Tracing([
            Trace(1, True, [1, 2, 3, 4, 5]),
            Trace(2, True, [5, 10, 17]),
            Trace(3, False, []),
            Trace(4, True, [20, 21, 22]),
            Trace(5, False, []),
        ])
        ground_truth_tracing = Tracing([
            Trace(1, True, [1, 2, 3]),
            Trace(2, True, [10, 11, 17, 19, 20]),
            Trace(3, False, []),
            Trace(4, False, []),
            Trace(5, True, [5, 7, 10, 22, 23, 25, 29, 30]),
        ])
        
        tp = 8.0
        fp = 2.0
        fn = 7.0
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        comparison = tracing.compare(ground_truth_tracing, TracingComparator.LINEWISE_PRECISION_RECALL)

        assert comparison["precision"] == pytest.approx(precision)
        assert comparison["recall"] == pytest.approx(recall)
