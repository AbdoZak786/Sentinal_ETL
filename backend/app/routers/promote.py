import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.promotion_service import (
    PromotionNotAllowedError,
    get_lineage,
    promote_version_to_gold,
    promote_version_to_silver,
    suggest_gold_aggregation,
)
from app.services.validation_service import DatasetVersionNotFoundError

router = APIRouter()


class GoldPromotionRequest(BaseModel):
    aggregation_spec: dict = Field(
        ...,
        description="Aggregation spec with group_by and metrics for promote_to_gold",
    )


@router.post("/{dataset_id}/versions/{version_id}/promote/silver")
async def promote_dataset_version_to_silver(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await promote_version_to_silver(db, dataset_id, version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromotionNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Silver promotion failed: {exc}",
        ) from exc


@router.post("/{dataset_id}/versions/{version_id}/promote/gold")
async def promote_dataset_version_to_gold(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    body: GoldPromotionRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await promote_version_to_gold(
            db,
            dataset_id,
            version_id,
            body.aggregation_spec,
        )
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromotionNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Gold promotion failed: {exc}",
        ) from exc


@router.get("/{dataset_id}/versions/{version_id}/suggest-aggregation")
async def suggest_dataset_version_aggregation(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await suggest_gold_aggregation(db, dataset_id, version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PromotionNotAllowedError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to suggest aggregation: {exc}",
        ) from exc


@router.get("/{dataset_id}/versions/{version_id}/lineage")
async def get_dataset_version_lineage(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await get_lineage(db, dataset_id, version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load lineage: {exc}",
        ) from exc
