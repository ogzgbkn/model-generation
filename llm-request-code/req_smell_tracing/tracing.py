import csv
import json
import logging
from dataclasses import dataclass
from enum import Enum
from io import StringIO

logger = logging.getLogger(__name__)

type ComparisonResult = dict[str, float] | dict[int, dict[str, float]]


class TracingComparator(Enum):
    """
    Enumeration for different tracing comparators. This is used to specify how tracings
    should be compared.
    """

    LINEWISE_PRECISION_RECALL = "linewise_precision_recall"
    IMPLEMENTED_PRECISION_RECALL = "implemented_precision_recall"


class TracingColumn(Enum):
    """
    Enumeration for different tracing columns. This is used to specify the columns in a
    tracing CSV file.
    """

    REQUIREMENT_ID = "requirement_id"
    IMPLEMENTED = "implemented"
    LINES_OF_CODE = "lines_of_code"


@dataclass
class Trace:
    """
    Represents a trace of a requirement.

    Attributes:
        requirement_id (int): The ID of the requirement.
        implemented (bool): Whether the requirement is implemented.
        lines_of_code (list[int]): The lines of code that implement the requirement.
    """

    requirement_id: int
    implemented: bool
    lines_of_code: list[int]

    def __lt__(self, other):
        return self.requirement_id < other.requirement_id

    @staticmethod
    def parse_lines_str(lines_str: str) -> list[int]:
        """
        Parses a string of lines and returns a list of integers. The string can contain a
        comma-separated list of line numbers and ranges of line numbers (e.g. "1,3-5,7").

        Args:
            lines_str (str): The string of lines.

        Returns:
            list[int]: The list of parsed lines.
        """
        parts = lines_str.strip().split(",")
        lines = []

        try:
            for part in parts:
                if not part:
                    continue

                if "-" in part:
                    start, end = part.split("-")
                    lines.extend(range(int(start), int(end) + 1))
                else:
                    lines.append(int(part))
        except ValueError as e:
            logger.error(f"Error parsing lines string {lines_str}: {e}")
            raise e

        # Remove duplicates
        lines = list(set(lines))

        return lines

    @staticmethod
    def parse_implemented(implemented: str) -> bool:
        """
        Parses a string representing whether a requirement is implemented. The string can
        be "true", "yes", "1" (for True) or "false", "no", "0" (for False).

        Args:
            implemented (str): The string representing whether the requirement is implemented.

        Returns:
            bool: True if the requirement is implemented, False otherwise.
        """
        true_values = ["true", "yes", "1"]
        false_values = ["false", "no", "0"]

        if implemented.strip().lower() in true_values:
            return True
        elif implemented.strip().lower() in false_values or not implemented.strip():
            return False
        else:
            raise ValueError(f"Unknown implemented value {implemented}")


class Tracing:
    """
    Represents a tracing of requirements to lines of code.

    Attributes:
        traces (list[Trace]): The traces of the requirements.
    """

    def __init__(self, traces: list[Trace]):
        self.traces = traces

    def get_trace_by_requirement_id(self, requirement_id: int) -> Trace | None:
        """
        Returns the trace with the specified requirement ID if it exists in the tracing.

        Args:
            requirement_id (int): The requirement ID.

        Returns:
            Trace | None: The trace with the specified requirement ID, or None if not found.
        """
        return next(
            filter(lambda t: t.requirement_id == requirement_id, self.traces), None
        )

    @staticmethod
    def from_json(
        json_str: str, list_key: str, keys: dict[TracingColumn, str]
    ) -> "Tracing":
        """
        Loads a tracing from a JSON string with the specified keys.

        Args:
            json_str (str): The JSON string.
            list_key (str): The key of the list of traces in the JSON string.
            keys (dict[TracingColumn, str]): The keys in the JSON string.
        """
        traces = []

        try:
            json_data = json.loads(json_str)

            for trace_data in json_data[list_key]:
                requirement_id = trace_data[keys[TracingColumn.REQUIREMENT_ID]]
                implemented = Trace.parse_implemented(
                    trace_data[keys[TracingColumn.IMPLEMENTED]]
                )
                lines_of_code = Trace.parse_lines_str(
                    trace_data[keys[TracingColumn.LINES_OF_CODE]]
                )

                traces.append(Trace(requirement_id, implemented, lines_of_code))
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing tracing JSON: {e}")

        return Tracing(traces)

    @staticmethod
    def from_csv(csv_str: str, columns: dict[TracingColumn, str]) -> "Tracing":
        """
        Loads a tracing from a CSV string with the specified columns.

        Args:
            csv_str (str): The CSV string.
            columns (dict[TracingColumn, str]): The columns in the CSV string.

        Returns:
            Tracing: The loaded tracing.
        """
        traces = []

        f = StringIO(csv_str)
        reader = csv.DictReader(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_ALL)

        for row in reader:
            try:
                requirement_id = int(row[columns[TracingColumn.REQUIREMENT_ID]])
                implemented = Trace.parse_implemented(
                    row[columns[TracingColumn.IMPLEMENTED]]
                )
                lines_of_code = Trace.parse_lines_str(
                    row[columns[TracingColumn.LINES_OF_CODE]]
                )

                traces.append(Trace(requirement_id, implemented, lines_of_code))
            except ValueError as e:
                logger.error(f"Error parsing tracing CSV row {row}: {e}")

        return Tracing(traces)

    def compare(
        self,
        ground_truth: "Tracing",
        comparator: TracingComparator,
    ) -> ComparisonResult:
        """
        Compares this tracing to a ground truth tracing using the specified comparator.

        Args:
            ground_truth (Tracing): The ground truth tracing to compare to.
            comparator (TracingComparator): The comparator to use.
            total_loc (int): The total number of lines of code in the source code
                             used for the tracings.

        Returns:
            ComparisonResult: The comparison results for the entire tracing or for each
                              requirement by ID, depending on the comparator.
        """
        match comparator:
            case TracingComparator.LINEWISE_PRECISION_RECALL:
                return self._compare_linewise_precision_recall(ground_truth)
            case TracingComparator.IMPLEMENTED_PRECISION_RECALL:
                return self._compare_implemented_precision_recall(ground_truth)
            case _:
                raise ValueError(f"Unknown comparator {comparator}")

    def _compare_implemented_precision_recall(
        self, ground_truth: "Tracing"
    ) -> dict[str, float]:
        """
        Compares this tracing to a ground truth tracing by comparing whether requirements are
        implemented both in this tracing and the ground trouth tracing. Computes precision
        and recall.

        Args:
            ground_truth (Tracing): The ground truth tracing to compare to.

        Returns:
            dict[str, float]: A dictionary containing the fields "precision" and "recall".
        """
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for ground_truth_trace in ground_truth.traces:
            trace = self.get_trace_by_requirement_id(ground_truth_trace.requirement_id)

            if not trace:
                logger.warning(
                    f"Trace not found for requirement {ground_truth_trace.requirement_id}"
                )
                continue

            if ground_truth_trace.implemented:
                if trace.implemented:
                    tp += 1
                else:
                    fn += 1
            else:
                if trace.implemented:
                    fp += 1
                else:
                    tn += 1

        return {
            "precision": tp / (tp + fp) if tp + fp > 0 else 0,
            "recall": tp / (tp + fn) if tp + fn > 0 else 0,
        }

    def _compare_linewise_precision_recall(
        self,
        ground_truth: "Tracing",
    ) -> dict[int, dict[str, float]]:
        """
        Compares this tracing to a ground truth tracing by comparing the lines of code that are
        predicted by the tracing. Computes precision and recall.

        Args:
            ground_truth (Tracing): The ground truth tracing to compare to.

        Returns:
            dict[str, float]: A dictionary containing the fields "precision" and "recall".
        """
        all_loc_predicted = list(
            set([line for trace in self.traces for line in trace.lines_of_code])
        )
        all_loc_ground_truth = list(
            set([line for trace in ground_truth.traces for line in trace.lines_of_code])
        )

        tp = 0
        fp = len(all_loc_predicted)
        fn = 0

        for line in all_loc_ground_truth:
            if line in all_loc_predicted:
                tp += 1
                fp -= 1
            else:
                fn += 1

        return {
            "precision": tp / (tp + fp) if tp + fp > 0 else 0,
            "recall": tp / (tp + fn) if tp + fn > 0 else 0,
        }

    def _lines_predicted(self) -> int:
        """
        Returns the total number of lines predicted by the tracing.

        Returns:
            int: The total number of lines predicted.
        """
        return sum([len(trace.lines_of_code) for trace in self.traces])
