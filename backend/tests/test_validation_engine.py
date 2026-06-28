import pandas as pd
import pytest

from app.services.validation_engine import (
    check_date_format,
    check_duplicates,
    check_nulls,
    check_types,
    compute_quality_score,
    detect_schema_drift,
    normalize_dtype_name,
)


@pytest.fixture
def clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Carol"],
            "amount": [10.5, 20.0, 30.25],
        }
    )


@pytest.fixture
def nulls_fail_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", None, "Carol"],
        }
    )


@pytest.fixture
def types_pass_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})


@pytest.fixture
def types_fail_df() -> pd.DataFrame:
    return pd.DataFrame({"id": [1, 2], "label": ["a", "b"]})


@pytest.fixture
def duplicates_pass_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def duplicates_fail_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})


# --- check_nulls ---


def test_check_nulls_passes_when_no_nulls(clean_df: pd.DataFrame) -> None:
    result = check_nulls(clean_df)

    assert result["passed"] is True
    assert result["report"] == {"id": 0, "name": 0, "amount": 0}


def test_check_nulls_fails_when_nulls_exceed_threshold(nulls_fail_df: pd.DataFrame) -> None:
    result = check_nulls(nulls_fail_df, threshold=0.0)

    assert result["passed"] is False
    assert result["report"]["name"] == 1


def test_check_nulls_passes_with_nonzero_threshold(nulls_fail_df: pd.DataFrame) -> None:
    result = check_nulls(nulls_fail_df, threshold=0.5)

    assert result["passed"] is True


# --- check_types ---


def test_check_types_passes_when_schema_matches(types_pass_df: pd.DataFrame) -> None:
    result = check_types(types_pass_df, {"id": "int64", "label": "str"})

    assert result["passed"] is True
    assert result["report"] == {}


def test_check_types_passes_when_no_expected_schema(types_pass_df: pd.DataFrame) -> None:
    result = check_types(types_pass_df, None)

    assert result["passed"] is True
    assert result["report"] == {}


def test_check_types_fails_on_dtype_mismatch(types_fail_df: pd.DataFrame) -> None:
    result = check_types(types_fail_df, {"id": "float64", "label": "int64"})

    assert result["passed"] is False
    assert result["report"]["id"] == {"expected": "float64", "actual": "int64"}
    assert result["report"]["label"] == {"expected": "int64", "actual": "str"}


def test_check_types_fails_on_missing_column(types_pass_df: pd.DataFrame) -> None:
    result = check_types(types_pass_df, {"missing_col": "int64"})

    assert result["passed"] is False
    assert result["report"]["missing_col"] == {"expected": "int64", "actual": None}


def test_check_types_treats_pyarrow_and_pandas_string_dtypes_as_equivalent() -> None:
    df = pd.DataFrame({"label": ["a", "b"]})

    result = check_types(df, {"label": "large_string"})

    assert result["passed"] is True
    assert result["report"] == {}


def test_check_types_treats_pyarrow_and_pandas_float_dtypes_as_equivalent() -> None:
    df = pd.DataFrame({"amount": [1.0, 2.5]})

    result = check_types(df, {"amount": "double"})

    assert result["passed"] is True
    assert result["report"] == {}


def test_normalize_dtype_name_maps_equivalent_names() -> None:
    assert normalize_dtype_name("large_string") == "string"
    assert normalize_dtype_name("str") == "string"
    assert normalize_dtype_name("double") == "float"
    assert normalize_dtype_name("float64") == "float"
    assert normalize_dtype_name("int64") == "int"
    assert normalize_dtype_name("boolean") == "bool"
    assert normalize_dtype_name("datetime64[ns]") == "datetime64[ns]"


# --- check_duplicates ---


def test_check_duplicates_passes_when_rows_are_unique(duplicates_pass_df: pd.DataFrame) -> None:
    result = check_duplicates(duplicates_pass_df)

    assert result["passed"] is True
    assert result["duplicate_count"] == 0


def test_check_duplicates_fails_when_rows_repeat(duplicates_fail_df: pd.DataFrame) -> None:
    result = check_duplicates(duplicates_fail_df)

    assert result["passed"] is False
    assert result["duplicate_count"] == 1


# --- detect_schema_drift ---


def test_detect_schema_drift_passes_with_no_previous_schema() -> None:
    current = {"id": "int64", "name": "str"}

    result = detect_schema_drift(current, None)

    assert result["drift_detected"] is False
    assert result["report"] is None


def test_detect_schema_drift_passes_when_schemas_match() -> None:
    schema = {"id": "int64", "name": "str"}

    result = detect_schema_drift(schema, schema.copy())

    assert result["drift_detected"] is False
    assert result["report"]["added_columns"] == []
    assert result["report"]["removed_columns"] == []
    assert result["report"]["type_changes"] == {}


def test_detect_schema_drift_ignores_pyarrow_vs_pandas_dtype_naming() -> None:
    previous = {"name": "large_string", "amount": "double"}
    current = {"name": "str", "amount": "float64"}

    result = detect_schema_drift(current, previous)

    assert result["drift_detected"] is False
    assert result["report"]["type_changes"] == {}


def test_detect_schema_drift_fails_on_added_removed_and_type_changes() -> None:
    previous = {"id": "int64", "name": "str", "legacy": "float64"}
    current = {"id": "float64", "name": "str", "created_at": "str"}

    result = detect_schema_drift(current, previous)

    assert result["drift_detected"] is True
    assert result["report"]["added_columns"] == ["created_at"]
    assert result["report"]["removed_columns"] == ["legacy"]
    assert result["report"]["type_changes"] == {
        "id": {"previous": "int64", "current": "float64"},
    }


# --- check_date_format ---


def test_check_date_format_passes_for_iso8601_dates() -> None:
    df = pd.DataFrame(
        {
            "order_date": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-10T14:30:00Z",
            ],
            "name": ["Alice", "Bob", "Carol"],
        }
    )

    result = check_date_format(df)

    assert result["passed"] is True
    assert result["detected_date_columns"] == ["order_date"]
    assert result["report"] == {}


def test_check_date_format_fails_for_mixed_date_formats() -> None:
    df = pd.DataFrame(
        {
            "order_date": [
                "2024-01-15",
                "15/01/2024",
                "Jan 15, 2024",
                "2024-03-10",
            ],
        }
    )

    result = check_date_format(df)

    assert result["passed"] is False
    assert result["detected_date_columns"] == ["order_date"]
    assert result["report"]["order_date"]["non_iso8601_count"] == 2
    assert set(result["report"]["order_date"]["examples"]) == {
        "15/01/2024",
        "Jan 15, 2024",
    }


def test_check_date_format_ignores_non_date_text_columns() -> None:
    df = pd.DataFrame(
        {
            "customer_name": [
                "Alice Smith",
                "Bob Jones",
                "Carol Lee",
                "Dave Kim",
            ],
        }
    )

    result = check_date_format(df)

    assert result["passed"] is True
    assert result["detected_date_columns"] == []
    assert result["report"] == {}


# --- compute_quality_score ---


def _perfect_check_results() -> tuple[dict, dict, dict, dict, dict]:
    return (
        {"passed": True, "report": {"id": 0, "name": 0}},
        {"passed": True, "report": {}},
        {"passed": True, "duplicate_count": 0},
        {"drift_detected": False, "report": None},
        {"passed": True, "detected_date_columns": [], "report": {}},
    )


def test_compute_quality_score_returns_perfect_score_when_all_checks_pass() -> None:
    null_result, type_result, duplicate_result, drift_result, date_result = (
        _perfect_check_results()
    )

    score = compute_quality_score(
        null_result,
        type_result,
        duplicate_result,
        drift_result,
        date_result,
        total_rows=10,
    )

    assert score == 100.0


def test_compute_quality_score_applies_penalties_when_checks_fail() -> None:
    null_result = {"passed": False, "report": {"id": 5, "name": 5}}
    type_result = {"passed": False, "report": {"id": {"expected": "float64", "actual": "int64"}}}
    duplicate_result = {"passed": False, "duplicate_count": 2}
    drift_result = {
        "drift_detected": True,
        "report": {
            "added_columns": ["new_col"],
            "removed_columns": [],
            "type_changes": {},
        },
    }
    date_result = {
        "passed": False,
        "detected_date_columns": ["order_date"],
        "report": {
            "order_date": {
                "non_iso8601_count": 2,
                "examples": ["15/01/2024"],
            }
        },
    }

    score = compute_quality_score(
        null_result,
        type_result,
        duplicate_result,
        drift_result,
        date_result,
        total_rows=10,
    )

    # 100 - 40*(10/20) null - 20 type - 20*(2/10) dup - 20 drift - 15*(2/10) date = 33
    assert score == 33.0


def test_compute_quality_score_floors_at_zero() -> None:
    null_result = {"passed": False, "report": {"a": 10, "b": 10, "c": 10}}
    type_result = {"passed": False, "report": {"a": {"expected": "x", "actual": "y"}}}
    duplicate_result = {"passed": False, "duplicate_count": 10}
    drift_result = {"drift_detected": True, "report": {"added_columns": ["x"], "removed_columns": [], "type_changes": {}}}
    date_result = {
        "passed": False,
        "detected_date_columns": ["d"],
        "report": {"d": {"non_iso8601_count": 10, "examples": ["bad"]}},
    }

    score = compute_quality_score(
        null_result,
        type_result,
        duplicate_result,
        drift_result,
        date_result,
        total_rows=10,
    )

    assert score == 0.0
