"""Pure validation helpers for pandas DataFrames.

These functions perform no database or file I/O and are intended to be
unit-tested in isolation.
"""

from __future__ import annotations

import re

import pandas as pd

NullCheckResult = dict[str, bool | dict[str, int]]
TypeCheckResult = dict[str, bool | dict[str, dict[str, str | None]]]
DuplicateCheckResult = dict[str, bool | int]
DriftResult = dict[str, bool | dict[str, list[str] | dict[str, dict[str, str]]] | None]
DateFormatCheckResult = dict[
    str,
    bool | list[str] | dict[str, dict[str, int | list[str]]],
]

ISO8601_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ISO8601_DATETIME_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?$"
)
DATE_COLUMN_PARSE_THRESHOLD = 0.8
MAX_NON_ISO8601_EXAMPLES = 5

_STRING_DTYPES = frozenset({"large_string", "string", "str", "object"})
_FLOAT_DTYPES = frozenset({"double", "float64", "float32", "float"})
_INT_DTYPES = frozenset({"int64", "int32", "int"})
_BOOL_DTYPES = frozenset({"bool", "boolean"})


def normalize_dtype_name(dtype_str: str) -> str:
    """Map equivalent PyArrow/pandas dtype names to one canonical form."""
    normalized = dtype_str.strip().lower()
    if normalized in _STRING_DTYPES:
        return "string"
    if normalized in _FLOAT_DTYPES:
        return "float"
    if normalized in _INT_DTYPES:
        return "int"
    if normalized in _BOOL_DTYPES:
        return "bool"
    return normalized


def dataframe_column_schema(df: pd.DataFrame) -> dict[str, str]:
    """Return column dtypes using pandas ``df.dtypes`` string forms."""
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def check_nulls(df: pd.DataFrame, threshold: float = 0.0) -> NullCheckResult:
    """Return null counts per column and whether all columns meet the threshold."""
    report: dict[str, int] = {
        col: int(df[col].isnull().sum()) for col in df.columns
    }

    if df.empty:
        return {"passed": True, "report": report}

    passed = True
    row_count = len(df)
    for col in df.columns:
        null_ratio = report[col] / row_count
        if null_ratio > threshold:
            passed = False
            break

    return {"passed": passed, "report": report}


def check_types(
    df: pd.DataFrame,
    expected_schema: dict[str, str] | None,
) -> TypeCheckResult:
    """Compare actual column dtypes against an expected schema."""
    if expected_schema is None:
        return {"passed": True, "report": {}}

    report: dict[str, dict[str, str | None]] = {}
    for col, expected in expected_schema.items():
        if col not in df.columns:
            report[col] = {"expected": expected, "actual": None}
            continue

        actual = str(df[col].dtype)
        if normalize_dtype_name(actual) != normalize_dtype_name(expected):
            report[col] = {"expected": expected, "actual": actual}

    return {"passed": len(report) == 0, "report": report}


def check_duplicates(df: pd.DataFrame) -> DuplicateCheckResult:
    """Return whether the DataFrame contains duplicate rows."""
    duplicate_count = int(df.duplicated().sum())
    return {
        "passed": duplicate_count == 0,
        "duplicate_count": duplicate_count,
    }


def detect_schema_drift(
    current_columns: dict[str, str],
    previous_columns: dict[str, str] | None,
) -> DriftResult:
    """Detect added/removed columns and dtype changes between schema versions."""
    if previous_columns is None:
        return {"drift_detected": False, "report": None}

    current_names = set(current_columns)
    previous_names = set(previous_columns)

    added_columns = sorted(current_names - previous_names)
    removed_columns = sorted(previous_names - current_names)

    type_changes: dict[str, dict[str, str]] = {}
    for col in sorted(current_names & previous_names):
        current_dtype = current_columns[col]
        previous_dtype = previous_columns[col]
        if normalize_dtype_name(current_dtype) != normalize_dtype_name(previous_dtype):
            type_changes[col] = {
                "previous": previous_dtype,
                "current": current_dtype,
            }

    drift_detected = bool(added_columns or removed_columns or type_changes)
    report = {
        "added_columns": added_columns,
        "removed_columns": removed_columns,
        "type_changes": type_changes,
    }

    return {"drift_detected": drift_detected, "report": report}


def _is_iso8601(value: object) -> bool:
    text = str(value).strip()
    return bool(
        ISO8601_DATE_PATTERN.fullmatch(text)
        or ISO8601_DATETIME_PATTERN.fullmatch(text)
    )


def _parses_as_datetime(value: object) -> bool:
    try:
        pd.to_datetime(value, errors="raise")
        return True
    except (TypeError, ValueError):
        return False


def _is_string_like_column(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _is_date_column_candidate(series: pd.Series, sample_size: int) -> bool:
    if not _is_string_like_column(series):
        return False

    non_null_values = series.dropna()
    if non_null_values.empty:
        return False

    sample = non_null_values.head(sample_size)
    parseable_count = sum(_parses_as_datetime(value) for value in sample)
    return (parseable_count / len(sample)) >= DATE_COLUMN_PARSE_THRESHOLD


def check_date_format(df: pd.DataFrame, sample_size: int = 20) -> DateFormatCheckResult:
    """Detect string-like date columns and enforce strict ISO 8601 formatting."""
    detected_date_columns: list[str] = []
    report: dict[str, dict[str, int | list[str]]] = {}

    for col in df.columns:
        series = df[col]
        dtype_name = str(series.dtype)

        if dtype_name.startswith("datetime64") or dtype_name in {"bool", "boolean"}:
            continue
        if pd.api.types.is_numeric_dtype(series):
            continue
        if not _is_date_column_candidate(series, sample_size):
            continue

        detected_date_columns.append(col)

        non_iso8601_count = 0
        examples: list[str] = []
        for value in series.dropna():
            if _is_iso8601(value):
                continue

            non_iso8601_count += 1
            example = str(value)
            if example not in examples and len(examples) < MAX_NON_ISO8601_EXAMPLES:
                examples.append(example)

        if non_iso8601_count > 0:
            report[col] = {
                "non_iso8601_count": non_iso8601_count,
                "examples": examples,
            }

    passed = len(report) == 0
    return {
        "passed": passed,
        "detected_date_columns": detected_date_columns,
        "report": report,
    }


def compute_quality_score(
    null_result: NullCheckResult,
    type_result: TypeCheckResult,
    duplicate_result: DuplicateCheckResult,
    drift_result: DriftResult,
    date_format_result: DateFormatCheckResult,
    total_rows: int,
) -> float:
    """
    Compute a weighted data-quality score from 0 to 100.

    Scoring formula (defensible in review/interview):
    - Start at 100 points.
    - Null penalty (up to 40): 40 * (total_null_cells / total_cells). This scales
      with the overall sparsity of the dataset rather than punishing a single
      bad column disproportionately.
    - Type penalty (20 flat): subtract 20 if any column dtype does not match the
      expected schema. Type mismatches are binary correctness failures.
    - Duplicate penalty (up to 20): 20 * (duplicate_rows / total_rows). Duplicate
      rows reduce trust in row-level uniqueness without dominating the score.
    - Drift penalty (20 flat): subtract 20 if schema drift is detected (added,
      removed, or changed columns vs. the previous version). Drift is a structural
      change that may break downstream consumers even when row values look fine.
    - Date format penalty (up to 15): if ISO 8601 date-format validation fails,
      subtract 15 * (non_compliant_date_values / total_date_column_cells), where
      total_date_column_cells is total_rows multiplied by the number of detected
      date columns. This penalizes non-standard date strings (e.g. "15/01/2024")
      without treating free-text columns as dates.
    - Final score is floored at 0 and rounded to 2 decimal places.
    """
    score = 100.0

    null_report = null_result["report"]
    if isinstance(null_report, dict) and total_rows > 0 and null_report:
        total_cells = total_rows * len(null_report)
        total_nulls = sum(null_report.values())
        null_ratio = total_nulls / total_cells if total_cells else 0.0
        score -= 40.0 * null_ratio

    if not type_result["passed"]:
        score -= 20.0

    duplicate_count = duplicate_result["duplicate_count"]
    if isinstance(duplicate_count, int) and total_rows > 0:
        duplicate_ratio = duplicate_count / total_rows
        score -= 20.0 * duplicate_ratio

    if drift_result["drift_detected"]:
        score -= 20.0

    detected_date_columns = date_format_result.get("detected_date_columns", [])
    if (
        not date_format_result.get("passed", True)
        and isinstance(detected_date_columns, list)
        and detected_date_columns
        and total_rows > 0
    ):
        date_report = date_format_result.get("report", {})
        total_non_compliant = 0
        if isinstance(date_report, dict):
            for column_report in date_report.values():
                if isinstance(column_report, dict):
                    count = column_report.get("non_iso8601_count", 0)
                    if isinstance(count, int):
                        total_non_compliant += count

        total_date_cells = total_rows * len(detected_date_columns)
        date_format_ratio = total_non_compliant / total_date_cells
        score -= 15.0 * date_format_ratio

    return round(max(score, 0.0), 2)
