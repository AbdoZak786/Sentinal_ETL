"""Orchestration layer for dataset version validation."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import (
    DatasetVersion,
    DatasetVersionStatus,
    PipelineRun,
    PipelineRunStatus,
    PipelineStage,
    ValidationResult,
)
from app.services.validation_engine import (
    check_date_format,
    check_duplicates,
    check_nulls,
    check_types,
    compute_quality_score,
    dataframe_column_schema,
    detect_schema_drift,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


class DatasetVersionNotFoundError(Exception):
    """Raised when a dataset version cannot be found for validation."""


def _parquet_path(bronze_path: str) -> Path:
    return REPO_ROOT / bronze_path


def _read_parquet_dataframe(bronze_path: str) -> pd.DataFrame:
    parquet_file = _parquet_path(bronze_path)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found: {bronze_path}")
    return pd.read_parquet(parquet_file)


def _read_parquet_column_schema(bronze_path: str) -> dict[str, str]:
    """Load column dtypes via pandas, matching in-memory DataFrame schema capture."""
    return dataframe_column_schema(_read_parquet_dataframe(bronze_path))


def _current_column_schema(df: pd.DataFrame) -> dict[str, str]:
    return dataframe_column_schema(df)


def _all_checks_passed(
    null_result: dict,
    type_result: dict,
    duplicate_result: dict,
    drift_result: dict,
    date_format_result: dict,
) -> bool:
    return (
        bool(null_result["passed"])
        and bool(type_result["passed"])
        and bool(duplicate_result["passed"])
        and not bool(drift_result["drift_detected"])
        and bool(date_format_result["passed"])
    )


def _build_issues_summary(
    null_result: dict,
    type_result: dict,
    duplicate_result: dict,
    drift_result: dict,
    date_format_result: dict,
) -> list[str]:
    issues: list[str] = []

    null_report = null_result.get("report", {})
    if isinstance(null_report, dict):
        null_columns = [col for col, count in null_report.items() if count]
        if null_columns:
            issues.append(
                f"{len(null_columns)} column{'s' if len(null_columns) != 1 else ''} "
                "contain null values"
            )

    type_report = type_result.get("report", {})
    if isinstance(type_report, dict) and type_report:
        issues.append(
            f"{len(type_report)} column{'s' if len(type_report) != 1 else ''} "
            "have dtype mismatches"
        )

    duplicate_count = duplicate_result.get("duplicate_count", 0)
    if isinstance(duplicate_count, int) and duplicate_count > 0:
        issues.append(
            f"{duplicate_count} duplicate row{'s' if duplicate_count != 1 else ''} found"
        )

    if drift_result.get("drift_detected"):
        drift_report = drift_result.get("report", {})
        if isinstance(drift_report, dict):
            added = drift_report.get("added_columns", [])
            removed = drift_report.get("removed_columns", [])
            type_changes = drift_report.get("type_changes", {})
            if added:
                issues.append(
                    f"{len(added)} column{'s' if len(added) != 1 else ''} added in schema drift"
                )
            if removed:
                issues.append(
                    f"{len(removed)} column{'s' if len(removed) != 1 else ''} "
                    "removed in schema drift"
                )
            if isinstance(type_changes, dict) and type_changes:
                issues.append(
                    f"{len(type_changes)} column{'s' if len(type_changes) != 1 else ''} "
                    "changed dtype in schema drift"
                )

    date_report = date_format_result.get("report", {})
    detected_date_columns = date_format_result.get("detected_date_columns", [])
    if not date_format_result.get("passed", True) and isinstance(date_report, dict):
        affected_columns = [
            col
            for col in detected_date_columns
            if isinstance(date_report.get(col), dict)
            and date_report[col].get("non_iso8601_count", 0) > 0
        ]
        if affected_columns:
            count = len(affected_columns)
            issues.append(
                f"{count} column{'s' if count != 1 else ''} "
                "contain non-ISO8601 date values"
            )

    return issues


def _validation_result_to_dict(
    validation_result: ValidationResult,
    issues: list[str],
) -> dict:
    return {
        "id": str(validation_result.id),
        "dataset_version_id": str(validation_result.dataset_version_id),
        "null_check_passed": validation_result.null_check_passed,
        "null_report": validation_result.null_report,
        "type_check_passed": validation_result.type_check_passed,
        "type_report": validation_result.type_report,
        "duplicate_check_passed": validation_result.duplicate_check_passed,
        "duplicate_count": validation_result.duplicate_count,
        "schema_drift_detected": validation_result.schema_drift_detected,
        "schema_drift_report": validation_result.schema_drift_report,
        "date_format_passed": validation_result.date_format_passed,
        "date_format_report": validation_result.date_format_report,
        "quality_score": validation_result.quality_score,
        "created_at": validation_result.created_at.isoformat(),
        "issues": issues,
    }


def validation_has_failing_checks(validation_response: dict) -> bool:
    """Return True when any validation check failed."""
    return not (
        validation_response.get("null_check_passed", True)
        and validation_response.get("type_check_passed", True)
        and validation_response.get("duplicate_check_passed", True)
        and not validation_response.get("schema_drift_detected", False)
        and validation_response.get("date_format_passed", True)
    )


async def run_validation(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Validate a dataset version and persist ValidationResult + PipelineRun rows."""
    started_at = datetime.now(timezone.utc)

    result = await db.execute(
        select(DatasetVersion).where(
            DatasetVersion.id == version_id,
            DatasetVersion.dataset_id == dataset_id,
        )
    )
    dataset_version = result.scalar_one_or_none()
    if dataset_version is None:
        raise DatasetVersionNotFoundError(
            f"Dataset version {version_id} not found for dataset {dataset_id}"
        )

    try:
        df = _read_parquet_dataframe(dataset_version.bronze_path)
        current_columns = _current_column_schema(df)

        previous_columns: dict[str, str] | None = None
        expected_schema: dict[str, str] | None = None

        if dataset_version.version_number > 1:
            previous_result = await db.execute(
                select(DatasetVersion).where(
                    DatasetVersion.dataset_id == dataset_version.dataset_id,
                    DatasetVersion.version_number == dataset_version.version_number - 1,
                    DatasetVersion.status != DatasetVersionStatus.failed,
                )
            )
            previous_version = previous_result.scalar_one_or_none()
            if previous_version is not None:
                previous_columns = _read_parquet_column_schema(previous_version.bronze_path)
                expected_schema = previous_columns

        null_result = check_nulls(df)
        type_result = check_types(df, expected_schema)
        duplicate_result = check_duplicates(df)
        drift_result = detect_schema_drift(current_columns, previous_columns)
        date_format_result = check_date_format(df)

        total_rows = len(df)
        quality_score = compute_quality_score(
            null_result,
            type_result,
            duplicate_result,
            drift_result,
            date_format_result,
            total_rows,
        )

        all_passed = _all_checks_passed(
            null_result,
            type_result,
            duplicate_result,
            drift_result,
            date_format_result,
        )
        issues = _build_issues_summary(
            null_result,
            type_result,
            duplicate_result,
            drift_result,
            date_format_result,
        )

        validation_result = ValidationResult(
            dataset_version_id=dataset_version.id,
            null_check_passed=bool(null_result["passed"]),
            null_report=null_result["report"],
            type_check_passed=bool(type_result["passed"]),
            type_report=type_result["report"],
            duplicate_check_passed=bool(duplicate_result["passed"]),
            duplicate_count=int(duplicate_result["duplicate_count"]),
            schema_drift_detected=bool(drift_result["drift_detected"]),
            schema_drift_report=drift_result["report"],
            date_format_passed=bool(date_format_result["passed"]),
            date_format_report=date_format_result["report"],
            quality_score=quality_score,
        )
        db.add(validation_result)

        if all_passed:
            dataset_version.status = DatasetVersionStatus.validated

        finished_at = datetime.now(timezone.utc)
        pipeline_run = PipelineRun(
            dataset_version_id=dataset_version.id,
            stage=PipelineStage.validate,
            status=PipelineRunStatus.success,
            started_at=started_at,
            finished_at=finished_at,
        )
        db.add(pipeline_run)

        await db.commit()
        await db.refresh(validation_result)

        return _validation_result_to_dict(validation_result, issues)

    except Exception as exc:
        await db.rollback()
        finished_at = datetime.now(timezone.utc)

        failed_run = PipelineRun(
            dataset_version_id=dataset_version.id,
            stage=PipelineStage.validate,
            status=PipelineRunStatus.failed,
            started_at=started_at,
            finished_at=finished_at,
            error_message=str(exc),
        )
        db.add(failed_run)
        await db.commit()
        raise
