"""Orchestration layer for dataset version silver/gold promotion."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import AuditLogEntry, DatasetVersion, DatasetVersionStatus
from app.services.duckdb_engine import (
    DuckDBPromotionError,
    promote_to_gold,
    promote_to_silver,
    suggest_aggregation_spec,
)
from app.services.validation_service import DatasetVersionNotFoundError

REPO_ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ACTOR = "system"


class PromotionNotAllowedError(Exception):
    """Raised when a dataset version is not eligible for promotion."""


def _parquet_path(relative_path: str) -> Path:
    return REPO_ROOT / relative_path


def _repaired_parquet_path(dataset_id: uuid.UUID, version_number: int) -> str:
    return f"data/bronze/{dataset_id}/v{version_number}/repaired.parquet"


def _silver_parquet_path(dataset_id: uuid.UUID, version_number: int) -> str:
    return f"data/silver/{dataset_id}/v{version_number}/silver.parquet"


def _gold_parquet_path(dataset_id: uuid.UUID, version_number: int) -> str:
    return f"data/gold/{dataset_id}/v{version_number}/gold.parquet"


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


def _resolve_silver_source_path(
    dataset_id: uuid.UUID,
    dataset_version: DatasetVersion,
) -> str:
    repaired_path = _repaired_parquet_path(dataset_id, dataset_version.version_number)
    if _parquet_path(repaired_path).exists():
        return repaired_path
    return dataset_version.bronze_path


async def promote_version_to_silver(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Promote a repaired or validated dataset version to silver."""
    dataset_version = await _load_dataset_version(db, dataset_id, version_id)

    if dataset_version.status == DatasetVersionStatus.failed:
        raise PromotionNotAllowedError(
            "Cannot promote a failed dataset version to silver"
        )

    if dataset_version.status not in {
        DatasetVersionStatus.repaired,
        DatasetVersionStatus.validated,
    }:
        raise PromotionNotAllowedError(
            "Dataset version must have status 'repaired' or 'validated' "
            f"before silver promotion (current status: {dataset_version.status.value})"
        )

    source_path = _resolve_silver_source_path(dataset_id, dataset_version)
    if not _parquet_path(source_path).exists():
        raise PromotionNotAllowedError(
            f"Source Parquet file not found for promotion: {source_path}"
        )

    output_path = _silver_parquet_path(dataset_id, dataset_version.version_number)

    try:
        promotion_result = promote_to_silver(source_path, output_path)
    except DuckDBPromotionError as exc:
        raise PromotionNotAllowedError(str(exc)) from exc

    promoted_at = datetime.now(timezone.utc)
    dataset_version.silver_path = output_path
    dataset_version.promoted_to_silver_at = promoted_at
    dataset_version.status = DatasetVersionStatus.promoted_silver
    dataset_version.row_count = promotion_result["row_count"]

    audit_entry = AuditLogEntry(
        dataset_version_id=dataset_version.id,
        event_type="promotion",
        actor=SYSTEM_ACTOR,
        details={
            "tier": "silver",
            "source_path": source_path,
            "output_path": output_path,
            "row_count": promotion_result["row_count"],
            "column_count": promotion_result["column_count"],
        },
    )
    db.add(audit_entry)
    await db.commit()

    return promotion_result


async def promote_version_to_gold(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    aggregation_spec: dict,
) -> dict:
    """Promote a silver dataset version to gold using DuckDB aggregation."""
    dataset_version = await _load_dataset_version(db, dataset_id, version_id)

    if dataset_version.status != DatasetVersionStatus.promoted_silver:
        raise PromotionNotAllowedError(
            "Dataset version must have status 'promoted_silver' before gold promotion "
            f"(current status: {dataset_version.status.value})"
        )

    if not dataset_version.silver_path:
        raise PromotionNotAllowedError(
            "Dataset version has no silver_path; promote to silver first"
        )

    if not _parquet_path(dataset_version.silver_path).exists():
        raise PromotionNotAllowedError(
            f"Silver Parquet file not found: {dataset_version.silver_path}"
        )

    output_path = _gold_parquet_path(dataset_id, dataset_version.version_number)

    try:
        promotion_result = promote_to_gold(
            dataset_version.silver_path,
            output_path,
            aggregation_spec,
        )
    except DuckDBPromotionError as exc:
        raise PromotionNotAllowedError(str(exc)) from exc

    promoted_at = datetime.now(timezone.utc)
    dataset_version.gold_path = output_path
    dataset_version.promoted_to_gold_at = promoted_at
    dataset_version.status = DatasetVersionStatus.promoted_gold
    dataset_version.row_count = promotion_result["row_count"]

    audit_entry = AuditLogEntry(
        dataset_version_id=dataset_version.id,
        event_type="promotion",
        actor=SYSTEM_ACTOR,
        details={
            "tier": "gold",
            "source_path": dataset_version.silver_path,
            "output_path": output_path,
            "row_count": promotion_result["row_count"],
            "query_used": promotion_result["query_used"],
            "aggregation_spec": aggregation_spec,
        },
    )
    db.add(audit_entry)
    await db.commit()

    return promotion_result


async def suggest_gold_aggregation(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Suggest a default gold aggregation_spec from the version's silver Parquet."""
    dataset_version = await _load_dataset_version(db, dataset_id, version_id)

    if not dataset_version.silver_path:
        raise PromotionNotAllowedError(
            "Dataset version has no silver_path; promote to silver first"
        )

    if not _parquet_path(dataset_version.silver_path).exists():
        raise PromotionNotAllowedError(
            f"Silver Parquet file not found: {dataset_version.silver_path}"
        )

    try:
        aggregation_spec = suggest_aggregation_spec(dataset_version.silver_path)
    except DuckDBPromotionError as exc:
        raise PromotionNotAllowedError(str(exc)) from exc

    return {
        "aggregation_spec": aggregation_spec,
        "silver_path": dataset_version.silver_path,
    }


def _stage_entry(
    stage: str,
    path: str | None,
    timestamp: datetime | str | None,
) -> dict | None:
    if path is None:
        return None
    return {
        "stage": stage,
        "path": path,
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
    }


async def get_lineage(
    db: AsyncSession,
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
) -> dict:
    """Return the promotion lineage chain for a dataset version."""
    dataset_version = await _load_dataset_version(db, dataset_id, version_id)

    audit_result = await db.execute(
        select(AuditLogEntry)
        .where(AuditLogEntry.dataset_version_id == version_id)
        .order_by(AuditLogEntry.created_at.asc())
    )
    audit_entries = audit_result.scalars().all()

    repaired_path = _repaired_parquet_path(dataset_id, dataset_version.version_number)
    repaired_timestamp: str | None = None

    for entry in audit_entries:
        if entry.event_type not in {"repair_succeeded", "repair_failed"}:
            continue
        details = entry.details if isinstance(entry.details, dict) else {}
        logged_repaired_path = details.get("repaired_path")
        if isinstance(logged_repaired_path, str):
            repaired_path = logged_repaired_path
        repaired_timestamp = entry.created_at.isoformat()
        break

    silver_audit_timestamp: str | None = None
    gold_audit_timestamp: str | None = None
    for entry in audit_entries:
        if entry.event_type != "promotion":
            continue
        details = entry.details if isinstance(entry.details, dict) else {}
        tier = details.get("tier")
        if tier == "silver":
            silver_audit_timestamp = entry.created_at.isoformat()
        elif tier == "gold":
            gold_audit_timestamp = entry.created_at.isoformat()

    stages: list[dict] = []

    bronze_stage = _stage_entry(
        "bronze",
        dataset_version.bronze_path,
        dataset_version.created_at,
    )
    if bronze_stage is not None:
        stages.append(bronze_stage)

    if _parquet_path(repaired_path).exists() or repaired_timestamp is not None:
        repaired_stage = _stage_entry(
            "repaired",
            repaired_path if _parquet_path(repaired_path).exists() else None,
            repaired_timestamp,
        )
        if repaired_stage is not None:
            stages.append(repaired_stage)

    if dataset_version.silver_path:
        silver_timestamp = dataset_version.promoted_to_silver_at or silver_audit_timestamp
        silver_stage = _stage_entry(
            "silver",
            dataset_version.silver_path,
            silver_timestamp,
        )
        if silver_stage is not None:
            stages.append(silver_stage)

    if dataset_version.gold_path:
        gold_timestamp = dataset_version.promoted_to_gold_at or gold_audit_timestamp
        gold_stage = _stage_entry(
            "gold",
            dataset_version.gold_path,
            gold_timestamp,
        )
        if gold_stage is not None:
            stages.append(gold_stage)

    return {
        "dataset_id": str(dataset_id),
        "version_id": str(version_id),
        "version_number": dataset_version.version_number,
        "status": dataset_version.status.value,
        "stages": stages,
    }
