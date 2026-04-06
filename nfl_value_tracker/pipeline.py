"""
Full ETL pipeline — extract -> transform -> load -> validate -> report.

Run this script to populate (or refresh) all four PostgreSQL tables:
    dim_players
    fact_performance_2025
    fact_contracts_2026
    dim_free_agent_status   <- NEW: unsigned FA tracker (DAZN scrape)

Usage
-----
    python pipeline.py
    python pipeline.py --fresh   # drop all ORM tables first, then full rebuild

The script is fully idempotent: re-running updates existing rows via
INSERT ... ON CONFLICT DO UPDATE and produces identical row counts.
"""

from __future__ import annotations

import argparse
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


def main(*, fresh: bool = False) -> None:
    # ------------------------------------------------------------------
    # Step 0 — Optional clean slate
    # ------------------------------------------------------------------
    from database import (
        drop_all_tables,
        engine,
        init_db,
        upsert_dim_players,
        upsert_fact_performance,
        upsert_free_agent_status,
        print_row_counts,
    )

    if fresh:
        logger.info("=== STEP 0: drop_all_tables (clean database) ===")
        drop_all_tables()

    # ------------------------------------------------------------------
    # Step 1 — Init database (CREATE TABLE IF NOT EXISTS)
    # ------------------------------------------------------------------
    logger.info("=== STEP 1: init_db ===")
    init_db()

    # ------------------------------------------------------------------
    # Step 2 — Extract stats (nfl_data_py — free, no API key)
    # ------------------------------------------------------------------
    logger.info("=== STEP 2: extract stats ===")
    from extract_stats import extract_nfl_stats

    stats_df = extract_nfl_stats(season=2025)
    logger.info("Stats rows: %d", len(stats_df))

    # ------------------------------------------------------------------
    # Step 3 — Sync seasonal rosters
    # ------------------------------------------------------------------
    logger.info("=== STEP 3: fetch nflverse seasonal rosters ===")
    from extract_rosters import extract_rosters
    from config import FA_START_DATE

    roster_df = extract_rosters(season=2026)
    logger.info("Active nflverse players: %d", len(roster_df))


    # ------------------------------------------------------------------
    # Step 4 — Transform: add performance index
    # ------------------------------------------------------------------
    logger.info("=== STEP 4: performance index ===")
    from transform import add_performance_index

    stats_df = add_performance_index(stats_df)

    # ------------------------------------------------------------------
    # Step 5 — Load dim_players
    # ------------------------------------------------------------------
    logger.info("=== STEP 5: load dim_players ===")
    n_dim = upsert_dim_players(stats_df)
    print(f"dim_players upserted: {n_dim} rows")

    # ------------------------------------------------------------------
    # Step 6 — Load fact_performance_2025
    # ------------------------------------------------------------------
    logger.info("=== STEP 6: load fact_performance_2025 ===")
    n_perf = upsert_fact_performance(stats_df)
    print(f"fact_performance_2025 upserted: {n_perf} rows")

    # ------------------------------------------------------------------
    # Step 7.5 — Build dim_free_agent_status (unsigned FA tracker)
    # ------------------------------------------------------------------
    logger.info("=== STEP 7.5: upsert dim_free_agent_status ===")
    n_unsigned, n_signed = upsert_free_agent_status(
        stats_df, roster_df, fa_start_date=FA_START_DATE
    )
    print(
        f"dim_free_agent_status: {n_signed} signed, {n_unsigned} unsigned players."
    )

    # ------------------------------------------------------------------
    # Step 8 — Confirm row counts
    # ------------------------------------------------------------------
    logger.info("=== STEP 8: row count verification ===")
    print_row_counts()

    # Row-count guidance (original design target vs nfl_data_py weekly scope)
    with engine.connect() as conn:
        from sqlalchemy import text

        dim_count = conn.execute(text("SELECT COUNT(*) FROM dim_players")).scalar()
        perf_count = conn.execute(text("SELECT COUNT(*) FROM fact_performance_2025")).scalar()

    _log_row_count_guidance(dim_count, perf_count)

    # ------------------------------------------------------------------
    # Step 9 — Export Power BI Reports
    # ------------------------------------------------------------------
    logger.info("=== STEP 9: export power bi metrics ===")
    from export_powerbi import export_powerbi

    export_powerbi()
    print("\nPipeline complete.")


def _log_row_count_guidance(dim: int, perf: int) -> None:
    """Log actual counts; warn if far from design targets or from each other."""
    print(
        f"Row counts — dim_players: {dim}, "
        f"fact_performance_2025: {perf}"
    )
    design_dim_perf = (1_500, 2_000)
    if not (design_dim_perf[0] <= dim <= design_dim_perf[1]):
        logger.warning(
            "dim_players=%d outside typical design band %s (nfl_data_py weekly "
            "scope often yields ~600–900 unique players).",
            dim,
            design_dim_perf,
        )
    else:
        print(f"dim_players in design band {design_dim_perf} ✓")
    if not (design_dim_perf[0] <= perf <= design_dim_perf[1]):
        logger.warning(
            "fact_performance_2025=%d outside typical design band %s.",
            perf,
            design_dim_perf,
        )
    else:
        print(f"fact_performance_2025 in design band {design_dim_perf} ✓")
    if dim != perf:
        logger.warning(
            "dim_players (%d) and fact_performance_2025 (%d) differ — "
            "expected 1:1 after load.",
            dim,
            perf,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL FA tracker ETL pipeline.")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Drop all ORM tables before running (clean PostgreSQL rebuild).",
    )
    args = parser.parse_args()
    try:
        main(fresh=args.fresh)
    except Exception:
        logger.exception("Pipeline failed.")
        sys.exit(1)
