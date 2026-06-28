import enum
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Integer, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


class DatasetVersionStatus(str, enum.Enum):
    uploaded = "uploaded"
    validating = "validating"
    validated = "validated"
    repairing = "repairing"
    repaired = "repaired"
    promoted = "promoted"
    promoted_silver = "promoted_silver"
    promoted_gold = "promoted_gold"
    failed = "failed"


class PipelineStage(str, enum.Enum):
    ingest = "ingest"
    validate = "validate"
    repair = "repair"
    transform = "transform"


class PipelineRunStatus(str, enum.Enum):
    running = "running"
    success = "success"
    failed = "failed"


class RepairActionType(str, enum.Enum):
    type_conversion = "type_conversion"
    duplicate_removal = "duplicate_removal"
    null_fill = "null_fill"
    column_mapping = "column_mapping"
    schema_normalization = "schema_normalization"
    date_format_normalization = "date_format_normalization"


class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    name: Mapped[str] = mapped_column()
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
    current_version_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id", use_alter=True),
        nullable=True,
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("datasets.id"),
    )
    version_number: Mapped[int] = mapped_column(Integer)
    bronze_path: Mapped[str] = mapped_column()
    silver_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    gold_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    promoted_to_silver_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    promoted_to_gold_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    status: Mapped[DatasetVersionStatus] = mapped_column(
        Enum(
            DatasetVersionStatus,
            name="dataset_version_status",
            native_enum=False,
            length=30,
        ),
    )
    row_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class PipelineRun(Base):
    __tablename__ = "pipeline_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id"),
    )
    stage: Mapped[PipelineStage] = mapped_column(
        Enum(
            PipelineStage,
            name="pipeline_stage",
            native_enum=False,
        ),
    )
    status: Mapped[PipelineRunStatus] = mapped_column(
        Enum(
            PipelineRunStatus,
            name="pipeline_run_status",
            native_enum=False,
        ),
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)


class ValidationResult(Base):
    __tablename__ = "validation_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id"),
    )
    null_check_passed: Mapped[bool] = mapped_column(Boolean)
    null_report: Mapped[dict] = mapped_column(JSON)
    type_check_passed: Mapped[bool] = mapped_column(Boolean)
    type_report: Mapped[dict] = mapped_column(JSON)
    duplicate_check_passed: Mapped[bool] = mapped_column(Boolean)
    duplicate_count: Mapped[int] = mapped_column(Integer)
    schema_drift_detected: Mapped[bool] = mapped_column(Boolean)
    schema_drift_report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    date_format_passed: Mapped[bool] = mapped_column(Boolean)
    date_format_report: Mapped[dict] = mapped_column(JSON)
    quality_score: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class RepairAction(Base):
    __tablename__ = "repair_actions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id"),
    )
    action_type: Mapped[RepairActionType] = mapped_column(String(50))
    target_column: Mapped[str | None] = mapped_column(Text, nullable=True)
    before_value_sample: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    after_value_sample: Mapped[dict | list | None] = mapped_column(JSON, nullable=True)
    rows_affected: Mapped[int] = mapped_column(Integer)
    success: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )


class AuditLogEntry(Base):
    __tablename__ = "audit_log_entries"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=text("gen_random_uuid()"),
    )
    dataset_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dataset_versions.id"),
    )
    event_type: Mapped[str] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )
