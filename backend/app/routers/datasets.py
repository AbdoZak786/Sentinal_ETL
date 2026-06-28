import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.dataset_service import (
    get_dataset_version,
    get_validation_result,
    list_datasets,
)
from app.services.repair_service import ValidationResultNotFoundError
from app.services.validation_service import DatasetVersionNotFoundError

router = APIRouter()


@router.get("")
async def list_all_datasets(db: AsyncSession = Depends(get_db)) -> dict:
    try:
        datasets = await list_datasets(db)
        return {"datasets": datasets}
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list datasets: {exc}",
        ) from exc


@router.get("/{dataset_id}/versions/{version_id}")
async def get_dataset_version_detail(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await get_dataset_version(db, dataset_id, version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load dataset version: {exc}",
        ) from exc


@router.get("/{dataset_id}/versions/{version_id}/validation")
async def get_dataset_version_validation(
    dataset_id: uuid.UUID,
    version_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> dict:
    try:
        return await get_validation_result(db, dataset_id, version_id)
    except DatasetVersionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValidationResultNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load validation result: {exc}",
        ) from exc
