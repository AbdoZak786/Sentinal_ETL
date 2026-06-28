"""DuckDB-powered Parquet promotion helpers.

Uses an in-memory DuckDB connection per call. In-memory is a good fit for
short-lived API request handlers: no temp files to clean up, no cross-request
state, and promotion work completes within a single function scope. A persistent
on-disk DuckDB file would matter more for long-running background jobs that
reuse cached views or intermediate tables across many steps.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb

VALID_AGGREGATIONS = frozenset({"sum", "avg", "count", "min", "max"})
IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_STRING_TYPE_TOKENS = ("CHAR", "VARCHAR", "TEXT", "STRING", "ENUM", "UUID")
_NUMERIC_TYPE_TOKENS = (
    "INT",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "HUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "NUMERIC",
    "REAL",
)
MAX_SUGGESTED_GROUP_BY_COLUMNS = 2
MAX_SUGGESTED_NUMERIC_METRICS = 3
LOW_CARDINALITY_DISTINCT_RATIO = 0.5


class DuckDBPromotionError(Exception):
    """Raised when promotion inputs or generated SQL are invalid."""


REPO_ROOT = Path(__file__).resolve().parents[3]


def _connect() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _filesystem_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / path).resolve()


def _parquet_path(path: str) -> str:
    return str(_filesystem_path(path)).replace("\\", "/")


def _normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    return re.sub(r"\s+", "_", normalized)


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_string_type(column_type: str) -> bool:
    upper = column_type.upper()
    return any(token in upper for token in _STRING_TYPE_TOKENS)


def _is_numeric_type(column_type: str) -> bool:
    upper = column_type.upper()
    return any(token in upper for token in _NUMERIC_TYPE_TOKENS)


def _validate_identifier(name: str, allowed: set[str], field_name: str) -> str:
    if name not in allowed:
        raise DuckDBPromotionError(
            f"{field_name} references unknown column '{name}'"
        )
    return name


def _validate_alias(alias: str) -> str:
    if not IDENTIFIER_PATTERN.match(alias):
        raise DuckDBPromotionError(
            f"Invalid metric alias '{alias}': must match {IDENTIFIER_PATTERN.pattern}"
        )
    return alias


def _validate_aggregation(agg: str) -> str:
    lowered = agg.lower()
    if lowered not in VALID_AGGREGATIONS:
        raise DuckDBPromotionError(
            f"Unsupported aggregation '{agg}'. Allowed: {sorted(VALID_AGGREGATIONS)}"
        )
    return lowered


def _read_parquet_schema(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
) -> list[tuple[str, str]]:
    rows = conn.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [parquet_path],
    ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def _resolve_output_column_names(
    schema: list[tuple[str, str]],
    column_mapping: dict[str, str] | None,
) -> dict[str, str]:
    """Map physical Parquet column names to unique snake_case output names."""
    mapping = column_mapping or {}
    output_names: dict[str, str] = {}
    reserved: set[str] = set()

    for source_name, _ in schema:
        if source_name in mapping:
            logical_name = mapping[source_name]
        else:
            logical_name = source_name

        candidate = _normalize_column_name(logical_name)
        if not candidate:
            raise DuckDBPromotionError(
                f"Column '{source_name}' normalizes to an empty name"
            )

        final_name = candidate
        suffix = 1
        while final_name in reserved:
            final_name = f"{candidate}_{suffix}"
            suffix += 1

        output_names[source_name] = final_name
        reserved.add(final_name)

    return output_names


def _build_silver_select_sql(
    schema: list[tuple[str, str]],
    output_names: dict[str, str],
    source_parquet_path: str,
) -> str:
    select_parts: list[str] = []
    for source_name, column_type in schema:
        source_sql = _quote_identifier(source_name)
        if _is_string_type(column_type):
            expression = f"trim({source_sql})"
        else:
            expression = source_sql

        output_sql = _quote_identifier(output_names[source_name])
        select_parts.append(f"{expression} AS {output_sql}")

    columns_sql = ", ".join(select_parts)
    return (
        f"SELECT {columns_sql} "
        f"FROM read_parquet('{source_parquet_path}')"
    )


def _build_gold_aggregation_sql(
    schema: list[tuple[str, str]],
    aggregation_spec: dict,
    source_parquet_path: str,
) -> str:
    allowed_columns = {name for name, _ in schema}

    group_by = aggregation_spec.get("group_by")
    metrics = aggregation_spec.get("metrics")

    if not isinstance(group_by, list) or not group_by:
        raise DuckDBPromotionError("aggregation_spec.group_by must be a non-empty list")
    if not isinstance(metrics, list) or not metrics:
        raise DuckDBPromotionError("aggregation_spec.metrics must be a non-empty list")

    validated_group_by = [
        _validate_identifier(column, allowed_columns, "group_by")
        for column in group_by
    ]

    metric_selects: list[str] = []
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict):
            raise DuckDBPromotionError(f"metrics[{index}] must be an object")

        column = metric.get("column")
        agg = metric.get("agg")
        alias = metric.get("alias")

        if not isinstance(column, str) or not isinstance(agg, str) or not isinstance(alias, str):
            raise DuckDBPromotionError(
                f"metrics[{index}] requires string fields: column, agg, alias"
            )

        validated_column = _validate_identifier(column, allowed_columns, f"metrics[{index}].column")
        validated_agg = _validate_aggregation(agg)
        validated_alias = _validate_alias(alias)

        column_sql = _quote_identifier(validated_column)
        alias_sql = _quote_identifier(validated_alias)
        agg_sql = validated_agg.upper()

        metric_selects.append(f"{agg_sql}({column_sql}) AS {alias_sql}")

    group_by_sql = ", ".join(_quote_identifier(column) for column in validated_group_by)
    select_sql = ", ".join([group_by_sql, *metric_selects])

    return (
        f"SELECT {select_sql} "
        f"FROM read_parquet('{source_parquet_path}') "
        f"GROUP BY {group_by_sql}"
    )


def _write_query_to_parquet(
    conn: duckdb.DuckDBPyConnection,
    select_sql: str,
    output_path: str,
) -> None:
    destination = _filesystem_path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination_sql = _parquet_path(output_path)

    conn.execute(
        f"COPY ({select_sql}) TO '{destination_sql}' (FORMAT PARQUET)"
    )


def _count_rows(conn: duckdb.DuckDBPyConnection, parquet_path: str) -> int:
    result = conn.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [parquet_path],
    ).fetchone()
    if result is None:
        return 0
    return int(result[0])


def _count_distinct(
    conn: duckdb.DuckDBPyConnection,
    parquet_path: str,
    column_name: str,
) -> int:
    column_sql = _quote_identifier(column_name)
    result = conn.execute(
        f"SELECT COUNT(DISTINCT {column_sql}) FROM read_parquet(?)",
        [parquet_path],
    ).fetchone()
    if result is None:
        return 0
    return int(result[0])


def suggest_aggregation_spec(silver_parquet_path: str) -> dict:
    """Propose a default gold aggregation_spec from a silver Parquet schema.

    Heuristic (intentionally simple):
    - Compute distinct counts for every column.
    - group_by: up to 2 string columns with distinct_count / row_count <= 0.5,
      preferring the lowest cardinality first (good categorical dimensions).
    - metrics: up to 3 numeric columns (excluding group_by), each with sum and avg.
    """
    source_path = _parquet_path(silver_parquet_path)
    source_file = Path(source_path)
    if not source_file.exists():
        raise DuckDBPromotionError(f"Source Parquet file not found: {silver_parquet_path}")

    conn = _connect()
    try:
        schema = _read_parquet_schema(conn, source_path)
        if not schema:
            raise DuckDBPromotionError("Source Parquet file has no columns")

        row_count = _count_rows(conn, source_path)
        if row_count == 0:
            raise DuckDBPromotionError("Source Parquet file has no rows")

        column_stats: list[dict] = []
        for column_name, column_type in schema:
            distinct_count = _count_distinct(conn, source_path, column_name)
            distinct_ratio = distinct_count / row_count
            column_stats.append(
                {
                    "name": column_name,
                    "type": column_type,
                    "distinct_count": distinct_count,
                    "distinct_ratio": distinct_ratio,
                    "is_string": _is_string_type(column_type),
                    "is_numeric": _is_numeric_type(column_type),
                }
            )

        string_low_cardinality = [
            column
            for column in column_stats
            if column["is_string"]
            and column["distinct_count"] >= 1
            and column["distinct_ratio"] <= LOW_CARDINALITY_DISTINCT_RATIO
        ]
        string_low_cardinality.sort(
            key=lambda column: (
                column["distinct_ratio"],
                column["distinct_count"],
                column["name"],
            )
        )
        group_by = [
            column["name"]
            for column in string_low_cardinality[:MAX_SUGGESTED_GROUP_BY_COLUMNS]
        ]

        if not group_by:
            any_low_cardinality = [
                column
                for column in column_stats
                if column["distinct_count"] >= 1
                and column["distinct_ratio"] <= LOW_CARDINALITY_DISTINCT_RATIO
            ]
            any_low_cardinality.sort(
                key=lambda column: (
                    column["distinct_ratio"],
                    column["distinct_count"],
                    column["name"],
                )
            )
            group_by = [
                column["name"]
                for column in any_low_cardinality[:MAX_SUGGESTED_GROUP_BY_COLUMNS]
            ]

        if not group_by:
            non_unique_columns = [
                column for column in column_stats if column["distinct_ratio"] < 1.0
            ]
            if non_unique_columns:
                non_unique_columns.sort(
                    key=lambda column: (
                        column["distinct_ratio"],
                        column["distinct_count"],
                        column["name"],
                    )
                )
                group_by = [non_unique_columns[0]["name"]]
            else:
                group_by = [column_stats[0]["name"]]

        group_by_names = set(group_by)
        numeric_columns = [
            column["name"]
            for column in column_stats
            if column["is_numeric"] and column["name"] not in group_by_names
        ][:MAX_SUGGESTED_NUMERIC_METRICS]

        metrics: list[dict] = []
        for column_name in numeric_columns:
            metrics.append(
                {"column": column_name, "agg": "sum", "alias": f"{column_name}_sum"}
            )
            metrics.append(
                {"column": column_name, "agg": "avg", "alias": f"{column_name}_avg"}
            )

        if not metrics:
            metrics.append(
                {
                    "column": group_by[0],
                    "agg": "count",
                    "alias": "row_count",
                }
            )

        return {
            "group_by": group_by,
            "metrics": metrics,
        }
    finally:
        conn.close()


def promote_to_silver(
    repaired_parquet_path: str,
    output_path: str,
    column_mapping: dict[str, str] | None = None,
) -> dict:
    """Promote a repaired bronze Parquet file to silver using DuckDB SQL transforms."""
    source_path = _parquet_path(repaired_parquet_path)
    source_file = Path(source_path)
    if not source_file.exists():
        raise DuckDBPromotionError(f"Source Parquet file not found: {repaired_parquet_path}")

    if column_mapping is not None and not isinstance(column_mapping, dict):
        raise DuckDBPromotionError("column_mapping must be a dict when provided")

    conn = _connect()
    try:
        schema = _read_parquet_schema(conn, source_path)
        if not schema:
            raise DuckDBPromotionError("Source Parquet file has no columns")

        if column_mapping:
            unknown_mapping_keys = set(column_mapping) - {name for name, _ in schema}
            if unknown_mapping_keys:
                unknown = ", ".join(sorted(unknown_mapping_keys))
                raise DuckDBPromotionError(
                    f"column_mapping references unknown columns: {unknown}"
                )

        output_names = _resolve_output_column_names(schema, column_mapping)
        select_sql = _build_silver_select_sql(schema, output_names, source_path)
        _write_query_to_parquet(conn, select_sql, output_path)

        output_file_path = _parquet_path(output_path)
        row_count = _count_rows(conn, output_file_path)

        return {
            "row_count": row_count,
            "column_count": len(schema),
            "output_path": output_path,
        }
    finally:
        conn.close()


def promote_to_gold(
    silver_parquet_path: str,
    output_path: str,
    aggregation_spec: dict,
) -> dict:
    """Aggregate a silver Parquet file to gold using a validated DuckDB query."""
    source_path = _parquet_path(silver_parquet_path)
    source_file = Path(source_path)
    if not source_file.exists():
        raise DuckDBPromotionError(f"Source Parquet file not found: {silver_parquet_path}")

    if not isinstance(aggregation_spec, dict):
        raise DuckDBPromotionError("aggregation_spec must be a dict")

    conn = _connect()
    try:
        schema = _read_parquet_schema(conn, source_path)
        if not schema:
            raise DuckDBPromotionError("Source Parquet file has no columns")

        query_used = _build_gold_aggregation_sql(schema, aggregation_spec, source_path)
        _write_query_to_parquet(conn, query_used, output_path)

        output_file_path = _parquet_path(output_path)
        row_count = _count_rows(conn, output_file_path)

        return {
            "row_count": row_count,
            "output_path": output_path,
            "query_used": query_used,
        }
    finally:
        conn.close()
