from pathlib import Path

import duckdb
import pytest

from app.services.duckdb_engine import (
    DuckDBPromotionError,
    promote_to_gold,
    promote_to_silver,
    suggest_aggregation_spec,
)


def _write_sample_repaired_parquet(path: Path) -> None:
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        COPY (
            SELECT
                1001 AS "Order ID",
                '  Alice  ' AS customer_name,
                '  US  ' AS country,
                10 AS amount
            UNION ALL
            SELECT 1002, 'Bob', 'UK', 20
            UNION ALL
            SELECT 1003, '  Alice  ', 'US', 15
            UNION ALL
            SELECT 1004, 'Carol', 'CA', 5
        ) TO ? (FORMAT PARQUET)
        """,
        [str(path)],
    )
    conn.close()


def _parquet_row_count(path: Path) -> int:
    conn = duckdb.connect(":memory:")
    result = conn.execute(
        "SELECT COUNT(*) FROM read_parquet(?)",
        [str(path)],
    ).fetchone()
    conn.close()
    assert result is not None
    return int(result[0])


def _parquet_column_count(path: Path) -> int:
    conn = duckdb.connect(":memory:")
    rows = conn.execute(
        "DESCRIBE SELECT * FROM read_parquet(?)",
        [str(path)],
    ).fetchall()
    conn.close()
    return len(rows)


@pytest.fixture
def sample_repaired_parquet(tmp_path: Path) -> Path:
    parquet_path = tmp_path / "repaired.parquet"
    _write_sample_repaired_parquet(parquet_path)
    assert parquet_path.exists()
    assert _parquet_row_count(parquet_path) == 4
    assert _parquet_column_count(parquet_path) == 4
    return parquet_path


def test_promote_to_silver_writes_conformed_parquet(
    tmp_path: Path,
    sample_repaired_parquet: Path,
) -> None:
    silver_path = tmp_path / "silver" / "orders.parquet"

    result = promote_to_silver(
        str(sample_repaired_parquet),
        str(silver_path),
        column_mapping={"Order ID": "order id"},
    )

    assert silver_path.exists()
    assert result["output_path"] == str(silver_path)
    assert result["row_count"] == 4
    assert result["column_count"] == 4
    assert _parquet_row_count(silver_path) == 4
    assert _parquet_column_count(silver_path) == 4

    conn = duckdb.connect(":memory:")
    rows = conn.execute(
        """
        SELECT order_id, customer_name, country
        FROM read_parquet(?)
        WHERE order_id = 1001
        """,
        [str(silver_path)],
    ).fetchone()
    conn.close()

    assert rows == (1001, "Alice", "US")


def test_promote_to_gold_aggregates_silver_parquet(
    tmp_path: Path,
    sample_repaired_parquet: Path,
) -> None:
    silver_path = tmp_path / "silver.parquet"
    gold_path = tmp_path / "gold" / "orders_by_customer.parquet"

    promote_to_silver(str(sample_repaired_parquet), str(silver_path))

    result = promote_to_gold(
        str(silver_path),
        str(gold_path),
        {
            "group_by": ["customer_name"],
            "metrics": [
                {"column": "amount", "agg": "sum", "alias": "total_amount"},
                {"column": "order_id", "agg": "count", "alias": "order_count"},
            ],
        },
    )

    assert gold_path.exists()
    assert result["output_path"] == str(gold_path)
    assert result["row_count"] == 3
    assert _parquet_row_count(gold_path) == 3
    assert _parquet_column_count(gold_path) == 3
    assert "GROUP BY" in result["query_used"]
    assert "SUM" in result["query_used"]

    conn = duckdb.connect(":memory:")
    alice = conn.execute(
        """
        SELECT total_amount, order_count
        FROM read_parquet(?)
        WHERE customer_name = 'Alice'
        """,
        [str(gold_path)],
    ).fetchone()
    conn.close()

    assert alice == (25, 2)


def test_promote_to_silver_rejects_unknown_column_mapping(
    tmp_path: Path,
    sample_repaired_parquet: Path,
) -> None:
    with pytest.raises(DuckDBPromotionError, match="unknown columns"):
        promote_to_silver(
            str(sample_repaired_parquet),
            str(tmp_path / "silver.parquet"),
            column_mapping={"missing_col": "new_name"},
        )


def test_promote_to_gold_rejects_unknown_metric_column(
    tmp_path: Path,
    sample_repaired_parquet: Path,
) -> None:
    silver_path = tmp_path / "silver.parquet"
    promote_to_silver(str(sample_repaired_parquet), str(silver_path))

    with pytest.raises(DuckDBPromotionError, match="unknown column"):
        promote_to_gold(
            str(silver_path),
            str(tmp_path / "gold.parquet"),
            {
                "group_by": ["customer_name"],
                "metrics": [
                    {"column": "not_a_column", "agg": "sum", "alias": "total"},
                ],
            },
        )


def test_suggest_aggregation_spec_picks_low_cardinality_group_by_and_numeric_metrics(
    tmp_path: Path,
    sample_repaired_parquet: Path,
) -> None:
    silver_path = tmp_path / "silver.parquet"
    promote_to_silver(str(sample_repaired_parquet), str(silver_path))

    spec = suggest_aggregation_spec(str(silver_path))

    assert spec["group_by"] == ["country"]
    assert {"column": "amount", "agg": "sum", "alias": "amount_sum"} in spec["metrics"]
    assert {"column": "amount", "agg": "avg", "alias": "amount_avg"} in spec["metrics"]
    assert len(spec["metrics"]) >= 2


def test_suggest_aggregation_spec_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DuckDBPromotionError, match="Source Parquet file not found"):
        suggest_aggregation_spec(str(tmp_path / "missing.parquet"))
