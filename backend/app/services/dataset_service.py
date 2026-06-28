"""Read-only dataset and version queries."""

from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import Dataset, DatasetVersion, ValidationResult
from app.services.repair_service import ValidationResultNotFoundError
from app.services.validation_service import (
    DatasetVersionNotFoundError,
    _build_issues_summary,
    _validation_result_to_dict,
)


def _dataset_version_to_dict(dataset_version: DatasetVersion) -> dict:
    return {
        "id": str(dataset_version.id),
        "dataset_id": str(dataset_version.dataset_id),
        "version_number": dataset_version.version_number,
        "bronze_path": dataset_version.bronze_path,
        "silver_path": dataset_version.silver_path,
        "gold_path": dataset_version.gold_path,
        "status": dataset_version.status.value,
        "row_count": dataset_version.row_count,
        "promoted_to_silver_at": (
            dataset_version.promoted_to_silver_at.isoformat()
            if dataset_version.promoted_to_silver_at
            else None
        ),
        "promoted_to_gold_at": (
            dataset_version.promoted_to_gold_at.isoformat()
            if dataset_version.promoted_to_gold_at
            else None
        ),
        "created_at": dataset_version.created_at.isoformat(),
    }


async def _load_dataset_version(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> DatasetVersion:
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
    return dataset_version


async def _resolve_latest_version(
    db: AsyncSession,
    dataset: Dataset,
) -> DatasetVersion | None:
    if dataset.current_version_id is not None:
        result = await db.execute(
            select(DatasetVersion).where(DatasetVersion.id == dataset.current_version_id)
        )
        current_version = result.scalar_one_or_none()
        if current_version is not None:
            return current_version

    result = await db.execute(
        select(DatasetVersion)
        .where(DatasetVersion.dataset_id == dataset.id)
        .order_by(DatasetVersion.version_number.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _latest_validation_result(
    db: AsyncSession,
    version_id: uuid.UUID,
) -> ValidationResult | None:
    result = await db.execute(
        select(ValidationResult)
        .where(ValidationResult.dataset_version_id == version_id)
        .order_by(ValidationResult.created_at.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


def _issues_from_validation_result(validation_result: ValidationResult) -> list[str]:
    null_result = {
        "passed": validation_result.null_check_passed,
        "report": validation_result.null_report,
    }
    type_result = {
        "passed": validation_result.type_check_passed,
        "report": validation_result.type_report,
    }
    duplicate_result = {
        "passed": validation_result.duplicate_check_passed,
        "duplicate_count": validation_result.duplicate_count,
    }
    drift_result = {
        "drift_detected": validation_result.schema_drift_detected,
        "report": validation_result.schema_drift_report,
    }
    date_format_result = {
        "passed": validation_result.date_format_passed,
        "report": validation_result.date_format_report,
        "detected_date_columns": list((validation_result.date_format_report or {}).keys()),
    }
    return _build_issues_summary(
        null_result,
        type_result,
        duplicate_result,
        drift_result,
        date_format_result,
    )


async def list_datasets(db: AsyncSession) -> list[dict]:
    """Return all datasets with their current/latest version summary."""
    datasets_result = await db.execute(
        select(Dataset).order_by(Dataset.created_at.desc())
    )
    datasets = datasets_result.scalars().all()

    items: list[dict] = []
    for dataset in datasets:
        latest_version = await _resolve_latest_version(db, dataset)
        validation_result = (
            await _latest_validation_result(db, latest_version.id)
            if latest_version is not None
            else None
        )

        items.append(
            {
                "id": str(dataset.id),
                "name": dataset.name,
                "created_at": dataset.created_at.isoformat(),
                "version_id": str(latest_version.id) if latest_version else None,
                "version_number": (
                    latest_version.version_number if latest_version else None
                ),
                "status": latest_version.status.value if latest_version else None,
                "quality_score": (
                    validation_result.quality_score if validation_result else None
                ),
                "row_count": latest_version.row_count if latest_version else None,
            }
        )

    return items


async def get_dataset_version(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Return a single dataset version record."""
    dataset_version = await _load_dataset_version(db, dataset_id, version_id)

    dataset_result = await db.execute(
        select(Dataset).where(Dataset.id == dataset_id)
    )
    dataset = dataset_result.scalar_one_or_none()

    payload = _dataset_version_to_dict(dataset_version)
    payload["dataset_name"] = dataset.name if dataset is not None else None
    return payload


async def get_validation_result(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Return the latest validation result for a dataset version."""
    await _load_dataset_version(db, dataset_id, version_id)

    validation_result = await _latest_validation_result(db, version_id)
    if validation_result is None:
        raise ValidationResultNotFoundError(
            f"No validation result found for dataset version {version_id}"
        )

    issues = _issues_from_validation_result(validation_result)
    return _validation_result_to_dict(validation_result, issues)
