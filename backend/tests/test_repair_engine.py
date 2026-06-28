import pandas as pd
import pytest

from app.services.repair_engine import (
    normalize_schema,
    repair_date_format,
    repair_duplicates,
    repair_nulls,
    repair_types,
    run_repair_pipeline,
)


# --- repair_nulls ---


def test_repair_nulls_fills_numeric_nulls_with_median() -> None:
    df = pd.DataFrame({"amount": [10.0, None, 30.0, None]})
    null_report = {"amount": 2}

    repaired, actions = repair_nulls(df, null_report)

    assert repaired["amount"].tolist() == [10.0, 20.0, 30.0, 20.0]
    assert len(actions) == 1
    assert actions[0]["action_type"] == "null_fill"
    assert actions[0]["target_column"] == "amount"
    assert actions[0]["rows_affected"] == 2
    assert actions[0]["success"] is True
    assert actions[0]["before_value_sample"] == [None, None]
    assert actions[0]["after_value_sample"] == [20.0, 20.0]


def test_repair_nulls_fills_string_nulls_with_unknown() -> None:
    df = pd.DataFrame({"city": ["London", None, "Paris", None, "Berlin"]})
    null_report = {"city": 2}

    repaired, actions = repair_nulls(df, null_report)

    assert repaired["city"].tolist() == [
        "London",
        "UNKNOWN",
        "Paris",
        "UNKNOWN",
        "Berlin",
    ]
    assert actions[0]["after_value_sample"] == ["UNKNOWN", "UNKNOWN"]


def test_repair_nulls_leaves_datetime_nulls_for_manual_review() -> None:
    df = pd.DataFrame({"event_date": pd.to_datetime(["2024-01-01", None, "2024-03-01"])})
    null_report = {"event_date": 1}

    repaired, actions = repair_nulls(df, null_report)

    assert pd.isna(repaired["event_date"].iloc[1])
    assert actions[0]["success"] is False
    assert actions[0]["reason"] == "requires_manual_review"
    assert actions[0]["rows_affected"] == 1


def test_repair_nulls_skips_columns_without_nulls_in_report() -> None:
    df = pd.DataFrame({"id": [1, 2], "note": ["a", None]})
    null_report = {"id": 0, "note": 1}

    repaired, actions = repair_nulls(df, null_report)

    assert repaired["id"].tolist() == [1, 2]
    assert repaired["note"].tolist() == ["a", "UNKNOWN"]
    assert len(actions) == 1
    assert actions[0]["target_column"] == "note"


def test_repair_nulls_logs_failure_for_missing_column() -> None:
    df = pd.DataFrame({"id": [1, 2]})

    repaired, actions = repair_nulls(df, {"missing_col": 2})

    assert repaired.equals(df)
    assert actions[0]["success"] is False
    assert actions[0]["reason"] == "column not found in DataFrame"


def test_repair_nulls_rejects_unsupported_strategy() -> None:
    df = pd.DataFrame({"amount": [1.0, None]})

    repaired, actions = repair_nulls(df, {"amount": 1}, strategy="drop_rows")

    assert pd.isna(repaired["amount"].iloc[1])
    assert actions[0]["success"] is False
    assert "unsupported null-fill strategy" in actions[0]["reason"]


# --- repair_duplicates ---


def test_repair_duplicates_removes_exact_duplicate_rows() -> None:
    df = pd.DataFrame(
        {
            "order_id": [101, 101, 102, 101],
            "status": ["new", "new", "shipped", "new"],
        }
    )

    repaired, action = repair_duplicates(df)

    assert len(repaired) == 2
    assert repaired["order_id"].tolist() == [101, 102]
    assert repaired["status"].tolist() == ["new", "shipped"]
    assert action["action_type"] == "duplicate_removal"
    assert action["target_column"] is None
    assert action["rows_affected"] == 2
    assert action["success"] is True


def test_repair_duplicates_keeps_first_occurrence() -> None:
    df = pd.DataFrame({"a": [1, 1], "b": ["same", "same"]})

    repaired, action = repair_duplicates(df)

    assert len(repaired) == 1
    assert repaired["b"].iloc[0] == "same"
    assert action["rows_affected"] == 1


def test_repair_duplicates_noop_when_rows_are_unique() -> None:
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})

    repaired, action = repair_duplicates(df)

    assert repaired.equals(df)
    assert action["rows_affected"] == 0
    assert action["success"] is True


def test_repair_duplicates_does_not_drop_rows_that_differ() -> None:
    df = pd.DataFrame({"a": [1, 1], "b": ["x", "y"]})

    repaired, action = repair_duplicates(df)

    assert len(repaired) == 2
    assert action["rows_affected"] == 0


# --- repair_types ---


def test_repair_types_converts_clean_numeric_strings() -> None:
    df = pd.DataFrame({"quantity": ["1", "2", "3"]})
    type_report = {"quantity": {"expected": "int64", "actual": "object"}}

    repaired, actions = repair_types(df, type_report)

    assert repaired["quantity"].tolist() == [1, 2, 3]
    assert actions[0]["success"] is True
    assert actions[0]["newly_nulled_count"] == 0
    assert "partial_success" not in actions[0]


def test_repair_types_flags_newly_introduced_nulls_from_failed_coercion() -> None:
    df = pd.DataFrame({"amount": ["10", "not-a-number", "30"]})
    type_report = {"amount": {"expected": "float64", "actual": "object"}}

    repaired, actions = repair_types(df, type_report)

    assert repaired["amount"].tolist()[0] == 10.0
    assert pd.isna(repaired["amount"].iloc[1])
    assert repaired["amount"].tolist()[2] == 30.0

    action = actions[0]
    assert action["partial_success"] is True
    assert action["success"] is False
    assert action["newly_nulled_count"] == 1
    assert action["reason"] == "1 value(s) could not be coerced and became null"
    assert action["rows_affected"] == 2
    assert pd.isna(repaired.loc[1, "amount"])


def test_repair_types_does_not_count_preexisting_nulls_as_newly_nulled() -> None:
    df = pd.DataFrame({"amount": ["10", None, "30"]})
    type_report = {"amount": {"expected": "float64", "actual": "object"}}

    repaired, actions = repair_types(df, type_report)

    assert repaired["amount"].tolist()[0] == 10.0
    assert pd.isna(repaired["amount"].iloc[1])
    assert repaired["amount"].tolist()[2] == 30.0
    assert actions[0]["newly_nulled_count"] == 0
    assert actions[0]["success"] is True


def test_repair_types_coerces_datetime_strings() -> None:
    df = pd.DataFrame({"created_at": ["2024-01-15", "2024-02-20"]})
    type_report = {"created_at": {"expected": "datetime64[ns]", "actual": "object"}}

    repaired, actions = repair_types(df, type_report)

    assert pd.api.types.is_datetime64_any_dtype(repaired["created_at"])
    assert actions[0]["success"] is True
    assert actions[0]["newly_nulled_count"] == 0


def test_repair_types_logs_failure_for_unsupported_expected_dtype() -> None:
    df = pd.DataFrame({"flag": ["yes", "no"]})
    type_report = {"flag": {"expected": "bool", "actual": "object"}}

    repaired, actions = repair_types(df, type_report)

    assert repaired.equals(df)
    assert actions[0]["success"] is False
    assert "no safe coercion available" in actions[0]["reason"]


# --- repair_date_format ---


def test_repair_date_format_normalizes_mixed_formats_to_iso() -> None:
    df = pd.DataFrame(
        {"order_date": ["2024-01-15", "15/01/2024", "2024-03-10"]}
    )
    date_format_report = {
        "order_date": {
            "non_iso8601_count": 1,
            "examples": ["15/01/2024"],
        }
    }

    repaired, actions = repair_date_format(
        df,
        date_format_report,
        ["order_date"],
    )

    assert repaired["order_date"].tolist() == [
        "2024-01-15",
        "2024-01-15",
        "2024-03-10",
    ]
    assert len(actions) == 1
    assert actions[0]["action_type"] == "date_format_normalization"
    assert actions[0]["target_column"] == "order_date"
    assert actions[0]["rows_affected"] == 1
    assert actions[0]["rows_unparseable"] == 0
    assert actions[0]["before_value_sample"] == ["15/01/2024"]
    assert actions[0]["after_value_sample"] == ["2024-01-15"]


def test_repair_date_format_leaves_unparseable_values_null() -> None:
    df = pd.DataFrame(
        {"order_date": ["2024-01-15", "not_a_date", "15/01/2024"]}
    )
    date_format_report = {
        "order_date": {
            "non_iso8601_count": 2,
            "examples": ["not_a_date", "15/01/2024"],
        }
    }

    repaired, actions = repair_date_format(
        df,
        date_format_report,
        ["order_date"],
    )

    assert repaired["order_date"].tolist()[0] == "2024-01-15"
    assert pd.isna(repaired["order_date"].iloc[1])
    assert repaired["order_date"].tolist()[2] == "2024-01-15"
    assert actions[0]["rows_unparseable"] == 1
    assert actions[0]["rows_affected"] == 2
    assert actions[0]["before_value_sample"] == ["not_a_date", "15/01/2024"]
    assert actions[0]["after_value_sample"] == [None, "2024-01-15"]


def test_repair_date_format_uses_datetime_iso_when_time_present() -> None:
    df = pd.DataFrame(
        {"created_at": ["2024-01-15 14:30:00", "2024-02-01"]}
    )
    date_format_report = {
        "created_at": {
            "non_iso8601_count": 1,
            "examples": ["2024-01-15 14:30:00"],
        }
    }

    repaired, actions = repair_date_format(
        df,
        date_format_report,
        ["created_at"],
    )

    assert repaired["created_at"].tolist() == [
        "2024-01-15T14:30:00",
        "2024-02-01T00:00:00",
    ]
    assert actions[0]["rows_affected"] == 2
    assert actions[0]["rows_unparseable"] == 0


def test_repair_date_format_noop_when_nothing_to_repair() -> None:
    df = pd.DataFrame({"order_date": ["2024-01-15"]})

    repaired, actions = repair_date_format(df, {}, ["order_date"])

    assert repaired.equals(df)
    assert actions == []


# --- normalize_schema ---


def test_normalize_schema_lowercases_and_snake_cases_column_names() -> None:
    df = pd.DataFrame({"Order ID": [1], "Customer Name": ["Alice"]})

    repaired, action = normalize_schema(df)

    assert list(repaired.columns) == ["order_id", "customer_name"]
    assert action["action_type"] == "schema_normalization"
    assert action["renames"] == {
        "Order ID": "order_id",
        "Customer Name": "customer_name",
    }
    assert action["rows_affected"] == 2
    assert action["success"] is True


def test_normalize_schema_noop_when_names_already_normalized() -> None:
    df = pd.DataFrame({"order_id": [1], "customer_name": ["Alice"]})

    repaired, action = normalize_schema(df)

    assert list(repaired.columns) == ["order_id", "customer_name"]
    assert action["renames"] == {}
    assert action["rows_affected"] == 0


def test_normalize_schema_strips_whitespace_and_replaces_spaces() -> None:
    df = pd.DataFrame({"  Sign Up Date ": ["2024-01-01"]})

    repaired, action = normalize_schema(df)

    assert list(repaired.columns) == ["sign_up_date"]
    assert action["renames"] == {"  Sign Up Date ": "sign_up_date"}


# --- run_repair_pipeline ---


@pytest.fixture
def multi_issue_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Order ID": [1, 1, None],
            "Amount": ["10", "10", "bad"],
        }
    )


def test_run_repair_pipeline_runs_only_failed_checks() -> None:
    df = pd.DataFrame({"a": [1, 1, None], "b": ["x", "x", "y"]})
    validation_result = {
        "null_check_passed": False,
        "null_report": {"a": 1, "b": 0},
        "type_check_passed": True,
        "type_report": {},
        "duplicate_check_passed": False,
        "duplicate_count": 1,
        "schema_drift_detected": False,
    }

    repaired, actions = run_repair_pipeline(df, validation_result)

    assert len(repaired) == 2
    assert repaired["a"].tolist() == [1.0, 1.0]
    assert [action["action_type"] for action in actions] == [
        "null_fill",
        "duplicate_removal",
    ]


def test_run_repair_pipeline_normalizes_schema_when_drift_detected(
    multi_issue_df: pd.DataFrame,
) -> None:
    validation_result = {
        "null_check_passed": True,
        "null_report": {"Order ID": 1, "Amount": 0},
        "type_check_passed": False,
        "type_report": {"Amount": {"expected": "float64", "actual": "object"}},
        "duplicate_check_passed": True,
        "duplicate_count": 0,
        "schema_drift_detected": True,
    }

    repaired, actions = run_repair_pipeline(multi_issue_df, validation_result)

    assert list(repaired.columns) == ["order_id", "amount"]
    assert actions[0]["action_type"] == "schema_normalization"
    assert actions[1]["action_type"] == "type_conversion"
    assert actions[1]["target_column"] == "amount"


def test_run_repair_pipeline_runs_date_format_repair_when_check_failed() -> None:
    df = pd.DataFrame({"signup_date": ["2024-01-15", "15/01/2024", "2024-03-10"]})
    validation_result = {
        "null_check_passed": True,
        "null_report": {"signup_date": 0},
        "type_check_passed": True,
        "type_report": {},
        "duplicate_check_passed": True,
        "duplicate_count": 0,
        "schema_drift_detected": False,
        "date_format_passed": False,
        "date_format_report": {
            "signup_date": {
                "non_iso8601_count": 1,
                "examples": ["15/01/2024"],
            }
        },
        "detected_date_columns": ["signup_date"],
    }

    repaired, actions = run_repair_pipeline(df, validation_result)

    assert repaired["signup_date"].tolist()[1] == "2024-01-15"
    assert [action["action_type"] for action in actions] == [
        "date_format_normalization",
    ]


def test_run_repair_pipeline_skips_repairs_when_all_checks_passed() -> None:
    df = pd.DataFrame({"id": [1, 2], "name": ["Alice", "Bob"]})
    validation_result = {
        "null_check_passed": True,
        "null_report": {"id": 0, "name": 0},
        "type_check_passed": True,
        "type_report": {},
        "duplicate_check_passed": True,
        "duplicate_count": 0,
        "schema_drift_detected": False,
    }

    repaired, actions = run_repair_pipeline(df, validation_result)

    assert repaired.equals(df)
    assert actions == []
