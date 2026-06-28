"""Pure repair helpers for pandas DataFrames.

These functions perform no database or file I/O and are intended to be
unit-tested in isolation.
"""

from __future__ import annotations

import re

import pandas as pd

from app.services.validation_engine import _is_iso8601

MAX_VALUE_SAMPLE = 3
UNKNOWN_FILL_VALUE = "UNKNOWN"


def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def _is_datetime_series(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def _is_string_like_series(series: pd.Series) -> bool:
    return series.dtype == object or pd.api.types.is_string_dtype(series)


def _is_expected_numeric(expected: str) -> bool:
    lowered = expected.lower()
    return any(
        token in lowered
        for token in ("int", "float", "double", "numeric", "decimal", "number")
    )


def _is_expected_datetime(expected: str) -> bool:
    lowered = expected.lower()
    return "datetime" in lowered or lowered.startswith("timestamp")


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def _sample_filled_values(
    before: pd.Series,
    after: pd.Series,
    limit: int = MAX_VALUE_SAMPLE,
) -> tuple[list[object], list[object]]:
    filled_mask = before.isnull() & after.notnull()
    if not filled_mask.any():
        return [], []

    sample_indices = before.index[filled_mask][:limit]
    before_samples: list[object] = [None] * len(sample_indices)
    after_samples = [after.loc[idx] for idx in sample_indices]
    return before_samples, after_samples


def _failure_action(
    action_type: str,
    *,
    target_column: str | None = None,
    reason: str,
    rows_affected: int = 0,
) -> dict:
    return {
        "action_type": action_type,
        "target_column": target_column,
        "rows_affected": rows_affected,
        "success": False,
        "reason": reason,
    }


def repair_nulls(
    df: pd.DataFrame,
    null_report: dict,
    strategy: str = "safe_default",
) -> tuple[pd.DataFrame, list[dict]]:
    """Fill null values using a conservative default strategy per dtype."""
    if strategy != "safe_default":
        return _safe_copy(df), [
            _failure_action(
                "null_fill",
                reason=f"unsupported null-fill strategy: {strategy}",
            )
        ]

    repaired = _safe_copy(df)
    actions: list[dict] = []

    if not isinstance(null_report, dict):
        return repaired, actions

    for column, null_count in null_report.items():
        if not isinstance(null_count, int) or null_count <= 0:
            continue
        if column not in repaired.columns:
            actions.append(
                _failure_action(
                    "null_fill",
                    target_column=column,
                    reason="column not found in DataFrame",
                    rows_affected=null_count,
                )
            )
            continue

        try:
            series = repaired[column]
            before = series.copy()

            if _is_datetime_series(series):
                actions.append(
                    {
                        "action_type": "null_fill",
                        "target_column": column,
                        "rows_affected": null_count,
                        "before_value_sample": [None] * min(null_count, MAX_VALUE_SAMPLE),
                        "after_value_sample": [None] * min(null_count, MAX_VALUE_SAMPLE),
                        "success": False,
                        "reason": "requires_manual_review",
                    }
                )
                continue

            if pd.api.types.is_numeric_dtype(series):
                fill_value = series.median()
                if pd.isna(fill_value):
                    actions.append(
                        _failure_action(
                            "null_fill",
                            target_column=column,
                            reason="cannot compute median for all-null numeric column",
                            rows_affected=null_count,
                        )
                    )
                    continue
                repaired[column] = series.fillna(fill_value)
            elif _is_string_like_series(series):
                repaired[column] = series.fillna(UNKNOWN_FILL_VALUE)
            else:
                actions.append(
                    _failure_action(
                        "null_fill",
                        target_column=column,
                        reason=f"unsupported dtype for automatic null fill: {series.dtype}",
                        rows_affected=null_count,
                    )
                )
                continue

            before_samples, after_samples = _sample_filled_values(
                before,
                repaired[column],
            )
            actions.append(
                {
                    "action_type": "null_fill",
                    "target_column": column,
                    "rows_affected": null_count,
                    "before_value_sample": before_samples,
                    "after_value_sample": after_samples,
                    "success": True,
                }
            )
        except Exception as exc:
            actions.append(
                _failure_action(
                    "null_fill",
                    target_column=column,
                    reason=str(exc),
                    rows_affected=null_count,
                )
            )

    return repaired, actions


def repair_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows, keeping the first occurrence."""
    try:
        duplicate_mask = df.duplicated()
        rows_dropped = int(duplicate_mask.sum())
        if rows_dropped == 0:
            return _safe_copy(df), {
                "action_type": "duplicate_removal",
                "target_column": None,
                "rows_affected": 0,
                "success": True,
            }

        repaired = df.loc[~duplicate_mask].copy(deep=True)
        return repaired, {
            "action_type": "duplicate_removal",
            "target_column": None,
            "rows_affected": rows_dropped,
            "success": True,
        }
    except Exception as exc:
        return _safe_copy(df), _failure_action(
            "duplicate_removal",
            reason=str(exc),
        )


def repair_types(
    df: pd.DataFrame,
    type_report: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Attempt safe dtype coercion for columns flagged in the type report."""
    repaired = _safe_copy(df)
    actions: list[dict] = []

    if not isinstance(type_report, dict):
        return repaired, actions

    for column, mismatch in type_report.items():
        if not isinstance(mismatch, dict):
            continue

        expected = mismatch.get("expected")
        if not isinstance(expected, str):
            actions.append(
                _failure_action(
                    "type_conversion",
                    target_column=column,
                    reason="missing expected dtype in type report",
                )
            )
            continue

        if column not in repaired.columns:
            actions.append(
                _failure_action(
                    "type_conversion",
                    target_column=column,
                    reason="column not found in DataFrame",
                )
            )
            continue

        try:
            before = repaired[column].copy()
            before_notnull = before.notnull()
            before_null_count = int(before_notnull.sum())

            if _is_expected_numeric(expected):
                coerced = pd.to_numeric(before, errors="coerce")
            elif _is_expected_datetime(expected):
                coerced = pd.to_datetime(before, errors="coerce")
            else:
                actions.append(
                    _failure_action(
                        "type_conversion",
                        target_column=column,
                        reason=f"no safe coercion available for expected dtype: {expected}",
                    )
                )
                continue

            coerced_notnull = coerced.notnull()
            newly_nulled = int((before_notnull & ~coerced_notnull).sum())
            rows_affected = int((before_notnull & coerced_notnull).sum())

            repaired[column] = coerced

            changed_mask = before_notnull & coerced_notnull & (
                before.astype(str) != coerced.astype(str)
            )
            sample_indices = before.index[changed_mask][:MAX_VALUE_SAMPLE]
            before_samples = [before.loc[idx] for idx in sample_indices]
            after_samples = [coerced.loc[idx] for idx in sample_indices]

            action: dict = {
                "action_type": "type_conversion",
                "target_column": column,
                "rows_affected": rows_affected,
                "before_value_sample": before_samples,
                "after_value_sample": after_samples,
                "expected_dtype": expected,
                "actual_dtype": str(before.dtype),
                "newly_nulled_count": newly_nulled,
                "success": newly_nulled == 0,
            }
            if newly_nulled > 0:
                action["partial_success"] = True
                action["reason"] = (
                    f"{newly_nulled} value(s) could not be coerced and became null"
                )

            actions.append(action)
        except Exception as exc:
            actions.append(
                _failure_action(
                    "type_conversion",
                    target_column=column,
                    reason=str(exc),
                )
            )

    return repaired, actions


def normalize_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Normalize column names to lowercase snake_case."""
    try:
        renames: dict[str, str] = {}
        reserved = set(df.columns)

        for column in df.columns:
            new_name = _normalize_column_name(str(column))
            if new_name == column:
                continue

            final_name = new_name
            suffix = 1
            while final_name in reserved:
                final_name = f"{new_name}_{suffix}"
                suffix += 1

            renames[column] = final_name
            reserved.add(final_name)

        if not renames:
            return _safe_copy(df), {
                "action_type": "schema_normalization",
                "target_column": None,
                "rows_affected": 0,
                "renames": {},
                "success": True,
            }

        repaired = df.rename(columns=renames).copy(deep=True)
        return repaired, {
            "action_type": "schema_normalization",
            "target_column": None,
            "rows_affected": len(renames),
            "renames": renames,
            "success": True,
        }
    except Exception as exc:
        return _safe_copy(df), _failure_action(
            "schema_normalization",
            reason=str(exc),
        )


def _parse_single_datetime(value: object) -> pd.Timestamp:
    if value is None or pd.isna(value):
        return pd.NaT

    text = str(value).strip()
    if "/" in text and not _is_iso8601(text):
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if pd.notna(parsed):
            return parsed

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return parsed

    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    return series.apply(_parse_single_datetime)


def _value_suggests_time_component(value: object) -> bool:
    text = str(value).strip()
    if re.search(r"T\d{2}:\d{2}", text):
        return True
    return bool(re.search(r"\d{1,2}:\d{2}(:\d{2})?", text))


def _parsed_has_non_midnight(timestamp: pd.Timestamp) -> bool:
    return (
        timestamp.hour != 0
        or timestamp.minute != 0
        or timestamp.second != 0
        or timestamp.microsecond != 0
    )


def _should_use_datetime_iso_format(
    before: pd.Series,
    parsed: pd.Series,
    report_entry: dict,
) -> bool:
    for timestamp in parsed.dropna():
        if _parsed_has_non_midnight(timestamp):
            return True

    examples = report_entry.get("examples", [])
    if isinstance(examples, list):
        for example in examples:
            if _value_suggests_time_component(example):
                return True

    for value in before.dropna():
        if not _value_suggests_time_component(value):
            continue
        parsed_value = _parse_single_datetime(value)
        if pd.notna(parsed_value) and _parsed_has_non_midnight(parsed_value):
            return True

    return False


def _normalized_cell_value(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value).strip()


def _has_date_format_report_entry(report_entry: object) -> bool:
    if not isinstance(report_entry, dict):
        return False
    count = report_entry.get("non_iso8601_count", 0)
    return isinstance(count, int) and count > 0


def repair_date_format(
    df: pd.DataFrame,
    date_format_report: dict,
    detected_date_columns: list[str],
) -> tuple[pd.DataFrame, list[dict]]:
    """Normalize non-ISO8601 date strings to consistent ISO 8601 formatting."""
    repaired = _safe_copy(df)
    actions: list[dict] = []

    if not detected_date_columns or not isinstance(date_format_report, dict):
        return repaired, actions

    columns_to_repair = [
        column
        for column in detected_date_columns
        if _has_date_format_report_entry(date_format_report.get(column))
    ]
    if not columns_to_repair:
        return repaired, actions

    for column in columns_to_repair:
        report_entry = date_format_report[column]
        if column not in repaired.columns:
            actions.append(
                _failure_action(
                    "date_format_normalization",
                    target_column=column,
                    reason="column not found in DataFrame",
                )
            )
            continue

        try:
            before = repaired[column].copy()
            parsed = _parse_datetime_series(before)
            use_datetime_format = _should_use_datetime_iso_format(
                before,
                parsed,
                report_entry,
            )

            after = pd.Series(index=before.index, dtype=object)
            after.loc[before.isna()] = None

            parsed_mask = parsed.notna()
            if use_datetime_format:
                after.loc[parsed_mask] = parsed.loc[parsed_mask].dt.strftime(
                    "%Y-%m-%dT%H:%M:%S"
                )
            else:
                after.loc[parsed_mask] = parsed.loc[parsed_mask].dt.strftime("%Y-%m-%d")

            unparsed_mask = before.notna() & parsed.isna()
            after.loc[unparsed_mask] = None

            rows_unparseable = int(unparsed_mask.sum())
            rows_affected = int(
                sum(
                    _normalized_cell_value(before.loc[idx])
                    != _normalized_cell_value(after.loc[idx])
                    for idx in before.index
                )
            )

            before_samples: list[object] = []
            after_samples: list[object] = []
            for idx in before.index:
                original = before.loc[idx]
                if pd.isna(original) or _is_iso8601(original):
                    continue
                before_samples.append(str(original))
                repaired_value = after.loc[idx]
                after_samples.append(
                    None if pd.isna(repaired_value) else str(repaired_value)
                )
                if len(before_samples) >= MAX_VALUE_SAMPLE:
                    break

            repaired[column] = after
            actions.append(
                {
                    "action_type": "date_format_normalization",
                    "target_column": column,
                    "rows_affected": rows_affected,
                    "before_value_sample": before_samples,
                    "after_value_sample": after_samples,
                    "rows_unparseable": rows_unparseable,
                }
            )
        except Exception as exc:
            actions.append(
                _failure_action(
                    "date_format_normalization",
                    target_column=column,
                    reason=str(exc),
                )
            )

    return repaired, actions


def _remap_report_keys(report: dict, renames: dict) -> dict:
    if not renames:
        return report

    remapped: dict = {}
    for key, value in report.items():
        remapped[renames.get(key, key)] = value
    return remapped


def run_repair_pipeline(
    df: pd.DataFrame,
    validation_result: dict,
) -> tuple[pd.DataFrame, list[dict]]:
    """Run only the repair steps required by a validation result."""
    repaired = _safe_copy(df)
    all_actions: list[dict] = []
    pipeline_state = dict(validation_result)

    try:
        if pipeline_state.get("schema_drift_detected"):
            repaired, schema_action = normalize_schema(repaired)
            all_actions.append(schema_action)
            renames = schema_action.get("renames", {})
            if isinstance(renames, dict) and renames:
                detected_date_columns = pipeline_state.get("detected_date_columns", [])
                remapped_date_columns = [
                    renames.get(column, column)
                    for column in detected_date_columns
                    if isinstance(column, str)
                ]
                pipeline_state = {
                    **pipeline_state,
                    "null_report": _remap_report_keys(
                        pipeline_state.get("null_report", {}),
                        renames,
                    ),
                    "type_report": _remap_report_keys(
                        pipeline_state.get("type_report", {}),
                        renames,
                    ),
                    "date_format_report": _remap_report_keys(
                        pipeline_state.get("date_format_report", {}),
                        renames,
                    ),
                    "detected_date_columns": remapped_date_columns,
                }

        if not pipeline_state.get("type_check_passed", True):
            type_report = pipeline_state.get("type_report", {})
            if isinstance(type_report, dict) and type_report:
                repaired, type_actions = repair_types(repaired, type_report)
                all_actions.extend(type_actions)

        if not pipeline_state.get("date_format_passed", True):
            date_format_report = pipeline_state.get("date_format_report", {})
            detected_date_columns = pipeline_state.get("detected_date_columns", [])
            if isinstance(date_format_report, dict) and date_format_report:
                if not detected_date_columns:
                    detected_date_columns = [
                        column
                        for column in date_format_report
                        if _has_date_format_report_entry(date_format_report[column])
                    ]
                repaired, date_actions = repair_date_format(
                    repaired,
                    date_format_report,
                    detected_date_columns,
                )
                all_actions.extend(date_actions)

        if not pipeline_state.get("null_check_passed", True):
            null_report = pipeline_state.get("null_report", {})
            if isinstance(null_report, dict) and null_report:
                repaired, null_actions = repair_nulls(repaired, null_report)
                all_actions.extend(null_actions)

        if not pipeline_state.get("duplicate_check_passed", True):
            duplicate_count = pipeline_state.get("duplicate_count", 0)
            if isinstance(duplicate_count, int) and duplicate_count > 0:
                repaired, duplicate_action = repair_duplicates(repaired)
                all_actions.append(duplicate_action)

        return repaired, all_actions
    except Exception as exc:
        all_actions.append(
            _failure_action(
                "schema_normalization",
                reason=f"repair pipeline aborted: {exc}",
            )
        )
        return _safe_copy(df), all_actions
