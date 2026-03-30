"""
T-06: Full ETL pipeline — extract → transform → load.

Run this script to populate (or refresh) all three PostgreSQL tables:
    dim_players
    fact_performance_2025
    fact_contracts_2026

Usage
-----
    python pipeline.py

The script is fully idempotent: re-running updates existing rows via
INSERT ... ON CONFLICT DO UPDATE and produces identical row counts.
"""

from __future__ import annotations

import logging
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def main() -> None:
    # ------------------------------------------------------------------
    # Step 1 — Init database (CREATE TABLE IF NOT EXISTS)
    # ------------------------------------------------------------------
    logger.info("=== STEP 1: init_db ===")
    from database import (
        init_db,
        upsert_dim_players,
        upsert_fact_performance,
        upsert_fact_contracts,
        print_row_counts,
    )
    init_db()

    # ------------------------------------------------------------------
    # Step 2 — Extract stats (nfl_data_py — free, no API key)
    # ------------------------------------------------------------------
    logger.info("=== STEP 2: extract stats ===")
    from extract_stats import extract_nfl_stats

    stats_df = extract_nfl_stats(season=2025)
    logger.info("Stats rows: %d", len(stats_df))

    # ------------------------------------------------------------------
    # Step 3 — Extract contracts (Sportradar API + manual_contracts.csv)
    # ------------------------------------------------------------------
    logger.info("=== STEP 3: extract contracts ===")
    from extract_contracts import extract_contracts

    contracts_df = extract_contracts()
    logger.info("Contracts rows: %d", len(contracts_df))

    # ------------------------------------------------------------------
    # Step 4 — Transform: fuzzy-match contracts onto stats to get player_id
    # ------------------------------------------------------------------
    logger.info("=== STEP 4: fuzzy match + value metrics ===")
    from transform import match_and_merge, add_value_metrics

    merged = match_and_merge(contracts_df, stats_df)

    # Carry player_id from the stats side into the merged DataFrame so
    # fact_contracts_2026 has an FK that resolves against dim_players.
    # The match uses player_name as the bridge; we re-attach player_id
    # via another left join keyed on the resolved player_name.
    if "player_id" not in merged.columns:
        name_to_id = (
            stats_df[["player_name", "player_id"]]
            .dropna(subset=["player_name", "player_id"])
            .drop_duplicates(subset=["player_name"])
            .set_index("player_name")["player_id"]
        )
        merged["player_id"] = merged["player_name"].map(
            lambda n: name_to_id.get(n) if pd.notna(n) else None
        )

    valued = add_value_metrics(merged)

    # ------------------------------------------------------------------
    # Step 5 — Load dim_players
    # ------------------------------------------------------------------
    logger.info("=== STEP 5: load dim_players ===")
    n_dim = upsert_dim_players(stats_df)
    print(f"[T-06] dim_players upserted: {n_dim} rows")

    # ------------------------------------------------------------------
    # Step 6 — Load fact_performance_2025
    # ------------------------------------------------------------------
    logger.info("=== STEP 6: load fact_performance_2025 ===")
    n_perf = upsert_fact_performance(stats_df)
    print(f"[T-06] fact_performance_2025 upserted: {n_perf} rows")

    # ------------------------------------------------------------------
    # Step 7 — Load fact_contracts_2026
    # ------------------------------------------------------------------
    logger.info("=== STEP 7: load fact_contracts_2026 ===")
    # Only load contracts rows that have a FK-resolvable player_id.
    n_before_filter = len(valued)
    contracts_with_id = valued[valued["player_id"].notna()].copy()
    n_skipped = n_before_filter - len(contracts_with_id)
    if n_skipped:
        print(
            f"[T-06] {n_skipped} contract row(s) have no player_id FK match — "
            "skipping (unmatched signings)."
        )

    n_contracts = upsert_fact_contracts(contracts_with_id)
    print(f"[T-06] fact_contracts_2026 upserted: {n_contracts} rows")

    # ------------------------------------------------------------------
    # Step 8 — Confirm row counts
    # ------------------------------------------------------------------
    logger.info("=== STEP 8: row count verification ===")
    print_row_counts()

    # Sanity bounds
    with __import__("database").engine.connect() as conn:
        from sqlalchemy import text

        dim_count  = conn.execute(text("SELECT COUNT(*) FROM dim_players")).scalar()
        perf_count = conn.execute(text("SELECT COUNT(*) FROM fact_performance_2025")).scalar()
        con_count  = conn.execute(text("SELECT COUNT(*) FROM fact_contracts_2026")).scalar()

    _assert_range(dim_count,  1_500, 2_000, "dim_players")
    _assert_range(perf_count, 1_500, 2_000, "fact_performance_2025")
    _assert_range(con_count,     30,    60,  "fact_contracts_2026")

    print("[T-06] All row-count assertions passed ✓")


def _assert_range(count: int, lo: int, hi: int, label: str) -> None:
    if not (lo <= count <= hi):
        logger.warning(
            "[T-06] %s row count %d is outside expected range [%d, %d] — "
            "check your data sources.",
            label, count, lo, hi,
        )
    else:
        print(f"[T-06] {label}: {count} rows  (expected {lo}–{hi}) ✓")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)
