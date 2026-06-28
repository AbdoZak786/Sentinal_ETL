from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import settings
from app.routers import datasets, pipeline_runs, promote, repair, upload, validate

app = FastAPI(title="Sentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(datasets.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(pipeline_runs.router, prefix="/api/pipeline-runs", tags=["pipeline-runs"])
app.include_router(validate.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(repair.router, prefix="/api/datasets", tags=["datasets"])
app.include_router(promote.router, prefix="/api/datasets", tags=["datasets"])


@app.get("/")
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "sentinel-api"}
