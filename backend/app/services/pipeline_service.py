"""Pipeline run queries for dashboard and monitoring."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import DatasetVersion, PipelineRun, PipelineRunStatus

STORAGE_TIER_STATUSES = (
    "uploaded",
    "validated",
    "repaired",
    "promoted_silver",
    "promoted_gold",
    "failed",
)


def _run_to_dict(run: PipelineRun) -> dict:
    return {
        "id": str(run.id),
        "dataset_version_id": str(run.dataset_version_id),
        "stage": run.stage.value,
        "status": run.status.value,
        "started_at": run.started_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
        "error_message": run.error_message,
    }


async def get_pipeline_runs_summary(
    db: AsyncSession,
    *,
    period_hours: int = 24,
) -> dict:
    """Return pipeline run counts grouped by status for the recent period."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=period_hours)

    result = await db.execute(
        select(PipelineRun.status, func.count(PipelineRun.id))
        .where(PipelineRun.started_at >= cutoff)
        .group_by(PipelineRun.status)
    )
    rows = result.all()

    counts = {status.value: 0 for status in PipelineRunStatus}
    for status, count in rows:
        counts[status.value] = int(count)

    return {
        "period_hours": period_hours,
        "counts": {
            "success": counts.get("success", 0),
            "failed": counts.get("failed", 0),
            "running": counts.get("running", 0),
        },
        "total": sum(counts.values()),
    }


async def get_recent_pipeline_runs(
    db: AsyncSession,
    *,
    limit: int = 5,
) -> list[dict]:
    """Return the most recent pipeline runs."""
    bounded_limit = max(1, min(limit, 50))

    result = await db.execute(
        select(PipelineRun)
        .order_by(PipelineRun.started_at.desc())
        .limit(bounded_limit)
    )
    runs = result.scalars().all()
    return [_run_to_dict(run) for run in runs]


async def get_storage_tier_counts(db: AsyncSession) -> dict:
    """Return dataset version counts grouped by medallion-relevant status."""
    result = await db.execute(
        select(DatasetVersion.status, func.count(DatasetVersion.id)).group_by(
            DatasetVersion.status
        )
    )
    rows = result.all()

    raw_counts = {status.value: int(count) for status, count in rows}
    counts = {status: raw_counts.get(status, 0) for status in STORAGE_TIER_STATUSES}

    return {
        "counts": counts,
        "total_versions": sum(raw_counts.values()),
    }
