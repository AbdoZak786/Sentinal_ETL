from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.pipeline_service import (
    get_pipeline_runs_summary,
    get_recent_pipeline_runs,
    get_storage_tier_counts,
)

router = APIRouter()


@router.get("/summary")
async def pipeline_runs_summary(
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await get_pipeline_runs_summary(db)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load pipeline run summary: {exc}",
        ) from exc


@router.get("/recent")
async def recent_pipeline_runs(
    limit: int = Query(default=5, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        runs = await get_recent_pipeline_runs(db, limit=limit)
        return {"runs": runs}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load recent pipeline runs: {exc}",
        ) from exc


@router.get("/storage-tiers")
async def storage_tier_summary(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        return await get_storage_tier_counts(db)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load storage tier counts: {exc}",
        ) from exc
