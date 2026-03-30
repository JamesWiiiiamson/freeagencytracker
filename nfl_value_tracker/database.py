"""
T-06: SQLAlchemy schema + PostgreSQL loader.

Tables
------
dim_players          – one row per player (PK: player_id from nfl_data_py)
fact_performance_2025 – one row per player with 2025 EPA/fantasy stats
fact_contracts_2026   – one row per player with 2026 free-agency contract data

Upsert strategy
---------------
All three loaders use INSERT … ON CONFLICT (player_id) DO UPDATE so that
re-running the pipeline updates existing rows rather than duplicating them.

PostgreSQL note — ROUND() and ::numeric
---------------------------------------
PostgreSQL's ROUND() requires a numeric argument; passing a float8 column
directly raises "function round(double precision, integer) does not exist".
Fix: cast first, e.g.  ROUND(epa_per_play::numeric, 4).
This file uses Python-side rounding (round()) before inserting so the cast
is unnecessary here, but keep this note for any future raw-SQL queries.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Column,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Session

from config import DB_URL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Engine — one shared instance for the whole application.
# ---------------------------------------------------------------------------
engine = create_engine(DB_URL, pool_pre_ping=True, future=True)


# ---------------------------------------------------------------------------
# ORM Base
# ---------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# dim_players
# ---------------------------------------------------------------------------
class DimPlayer(Base):
    """
    One row per unique player.
    player_id comes from nfl_data_py (e.g. '00-0023459') and is stable
    across seasons, making it the natural primary key.
    """

    __tablename__ = "dim_players"

    player_id = Column(Text, primary_key=True, comment="nfl_data_py GSIS player ID")
    full_name  = Column(Text,         nullable=True)
    position   = Column(String(10),   nullable=True)
    age        = Column(SmallInteger, nullable=True)
    team_2025  = Column(String(10),   nullable=True, comment="Most recent 2025 team alias")


# ---------------------------------------------------------------------------
# fact_performance_2025
# ---------------------------------------------------------------------------
class FactPerformance2025(Base):
    """
    2025 season performance stats — one row per player.
    A UNIQUE constraint on player_id drives the ON CONFLICT upsert.
    """

    __tablename__ = "fact_performance_2025"
    __table_args__ = (
        UniqueConstraint("player_id", name="uq_fact_perf_player"),
    )

    id           = Column(Integer, primary_key=True, autoincrement=True)
    player_id    = Column(Text, ForeignKey("dim_players.player_id"), nullable=False)
    season       = Column(SmallInteger, nullable=True)
    plays        = Column(Integer, nullable=True, comment="Pass/rush play count from PBP")
    epa_total    = Column(Float, nullable=True)
    epa_per_play = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True, comment="Fraction of plays where EPA > 0")
    touchdowns   = Column(Integer, nullable=True, comment="Passing + rushing + receiving TDs")
    fantasy_pts  = Column(Float, nullable=True, comment="Half-PPR fantasy points")


# ---------------------------------------------------------------------------
# fact_contracts_2026
# ---------------------------------------------------------------------------
class FactContract2026(Base):
    """
    2026 free-agency contract data — one row per player.
    A UNIQUE constraint on player_id drives the ON CONFLICT upsert.
    """

    __tablename__ = "fact_contracts_2026"
    __table_args__ = (
        UniqueConstraint("player_id", name="uq_fact_contract_player"),
    )

    id             = Column(Integer, primary_key=True, autoincrement=True)
    player_id      = Column(Text, ForeignKey("dim_players.player_id"), nullable=False)
    new_team       = Column(String(10), nullable=True)
    contract_years = Column(SmallInteger, nullable=True)
    total_value    = Column(Float, nullable=True, comment="Total contractvalue in $M")
    aav            = Column(Float, nullable=True, comment="Average annual value in $M")
    value_metric   = Column(Float, nullable=True, comment="epa_per_play / aav_m")
    value_tier     = Column(String(20), nullable=True, comment="elite|solid|fair|overpaid")
    signing_month  = Column(SmallInteger, nullable=True, comment="Calendar month of signing (e.g. 3 = March)")


# ---------------------------------------------------------------------------
# Indexes on FK columns in fact tables (created after table DDL)
# ---------------------------------------------------------------------------
_idx_perf_player     = Index("ix_fact_perf_player_id",     FactPerformance2025.player_id)
_idx_contract_player = Index("ix_fact_contract_player_id", FactContract2026.player_id)


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------
def init_db() -> None:
    """
    Create all tables (and indexes) if they don't already exist.
    Safe to call on every pipeline run — CREATE TABLE IF NOT EXISTS semantics.
    """
    Base.metadata.create_all(engine)
    logger.info("[T-06] init_db complete — all tables present.")

    # Verify tables are visible in pg_catalog.
    with engine.connect() as conn:
        result = conn.execute(
            text(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname = 'public' "
                "AND tablename IN ('dim_players','fact_performance_2025','fact_contracts_2026') "
                "ORDER BY tablename;"
            )
        )
        found = [row[0] for row in result]

    expected = {"dim_players", "fact_performance_2025", "fact_contracts_2026"}
    missing  = expected - set(found)
    if missing:
        raise RuntimeError(f"[T-06] init_db: tables not found after create_all: {missing}")

    print(f"[T-06] Tables confirmed in PostgreSQL: {sorted(found)}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_int(val) -> Optional[int]:
    """Convert a value to int, returning None for NaN/None."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return int(val) if val is not None else None


def _safe_float(val, ndigits: int = 6) -> Optional[float]:
    """Convert a value to float (rounded), returning None for NaN/None."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return round(float(val), ndigits) if val is not None else None


def _safe_str(val) -> Optional[str]:
    """Convert a value to str, returning None for NaN/None/empty."""
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s if s else None


# ---------------------------------------------------------------------------
# upsert_dim_players
# ---------------------------------------------------------------------------
def upsert_dim_players(stats_df: pd.DataFrame) -> int:
    """
    Load (or update) dim_players from the weekly stats DataFrame.

    Conflict resolution: ON CONFLICT (player_id) DO UPDATE — all columns
    are refreshed so the dimension stays current on every run.

    Returns the number of rows upserted.
    """
    required = {"player_id", "player_name"}
    missing  = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"upsert_dim_players: missing columns {missing}")

    # Deduplicate on player_id — players who changed teams mid-season appear
    # multiple times in the weekly groupby.  Keep the row with the most plays
    # (or first row if plays is absent) so dim_players has one stable entry.
    deduped = stats_df.copy()
    if "plays" in deduped.columns:
        deduped = (
            deduped.sort_values("plays", ascending=False)
            .drop_duplicates(subset=["player_id"], keep="first")
        )
    else:
        deduped = deduped.drop_duplicates(subset=["player_id"], keep="first")

    rows: list[dict] = []
    for _, r in deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid:
            continue
        rows.append(
            {
                "player_id": pid,
                "full_name":  _safe_str(r.get("player_name")),
                "position":   _safe_str(r.get("position")),
                "age":        _safe_int(r.get("age")),
                "team_2025":  _safe_str(r.get("recent_team")),
            }
        )

    if not rows:
        logger.warning("[T-06] upsert_dim_players: no valid rows to insert.")
        return 0

    stmt = pg_insert(DimPlayer).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id"],
        set_={
            "full_name":  stmt.excluded.full_name,
            "position":   stmt.excluded.position,
            "age":        stmt.excluded.age,
            "team_2025":  stmt.excluded.team_2025,
        },
    )

    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

    logger.info("[T-06] upsert_dim_players: %d rows upserted.", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# upsert_fact_performance
# ---------------------------------------------------------------------------
def upsert_fact_performance(stats_df: pd.DataFrame) -> int:
    """
    Load (or update) fact_performance_2025.

    Conflict resolution: ON CONFLICT (player_id) DO UPDATE.
    One row per player — the unique constraint on player_id is the conflict target.

    Returns the number of rows upserted.
    """
    required = {"player_id"}
    missing  = required - set(stats_df.columns)
    if missing:
        raise ValueError(f"upsert_fact_performance: missing columns {missing}")

    # Deduplicate on player_id — keep the row with the most plays so we
    # capture the player's primary role in 2025 (e.g. QB over emergency RB).
    deduped = stats_df.copy()
    if "plays" in deduped.columns:
        deduped = (
            deduped.sort_values("plays", ascending=False)
            .drop_duplicates(subset=["player_id"], keep="first")
        )
    else:
        deduped = deduped.drop_duplicates(subset=["player_id"], keep="first")

    rows: list[dict] = []
    for _, r in deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid:
            continue

        # Touchdowns: sum whichever TD columns are present.
        td_cols = [c for c in stats_df.columns if "touchdown" in c.lower()]
        total_tds: Optional[int] = None
        if td_cols:
            td_vals = [r[c] for c in td_cols if pd.notna(r[c])]
            total_tds = int(sum(td_vals)) if td_vals else None

        # Fantasy points: prefer half-PPR column if it exists.
        fp_col = next(
            (c for c in stats_df.columns if "fantasy_points_ppr" in c.lower() or "fantasy_points" in c.lower()),
            None,
        )
        fantasy_pts = _safe_float(r[fp_col], ndigits=2) if fp_col else None

        rows.append(
            {
                "player_id":    pid,
                "season":       _safe_int(r.get("season")),
                "plays":        _safe_int(r.get("plays")),
                "epa_total":    _safe_float(r.get("epa_total")),
                "epa_per_play": _safe_float(r.get("epa_per_play")),
                "success_rate": _safe_float(r.get("success_rate")),
                "touchdowns":   total_tds,
                "fantasy_pts":  fantasy_pts,
            }
        )

    if not rows:
        logger.warning("[T-06] upsert_fact_performance: no valid rows to insert.")
        return 0

    stmt = pg_insert(FactPerformance2025).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_fact_perf_player",
        set_={
            "season":       stmt.excluded.season,
            "plays":        stmt.excluded.plays,
            "epa_total":    stmt.excluded.epa_total,
            "epa_per_play": stmt.excluded.epa_per_play,
            "success_rate": stmt.excluded.success_rate,
            "touchdowns":   stmt.excluded.touchdowns,
            "fantasy_pts":  stmt.excluded.fantasy_pts,
        },
    )

    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

    logger.info("[T-06] upsert_fact_performance: %d rows upserted.", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# upsert_fact_contracts
# ---------------------------------------------------------------------------
def upsert_fact_contracts(contracts_df: pd.DataFrame) -> int:
    """
    Load (or update) fact_contracts_2026 from the merged contracts DataFrame
    (output of transform.match_and_merge + transform.add_value_metrics).

    Requires a 'player_id' column — call this after attaching player_ids
    from the stats merge.  Rows without a player_id are logged and skipped.

    Conflict resolution: ON CONFLICT (player_id) DO UPDATE.

    Returns the number of rows upserted.
    """
    if "player_id" not in contracts_df.columns:
        raise ValueError(
            "upsert_fact_contracts: 'player_id' column is required. "
            "Ensure contracts_df was joined with stats_df before loading."
        )

    # Deduplicate on player_id to prevent cardinality violations during upsert.
    deduped = contracts_df.drop_duplicates(subset=["player_id"], keep="first")

    rows: list[dict] = []
    skipped = 0
    for _, r in deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid:
            skipped += 1
            continue

        # Extract signing month from the signed_date column if present.
        signing_month: Optional[int] = None
        raw_date = r.get("signed_date")
        if raw_date and not pd.isna(raw_date):
            try:
                signing_month = pd.to_datetime(raw_date).month
            except Exception:
                pass

        rows.append(
            {
                "player_id":      pid,
                "new_team":       _safe_str(r.get("new_team")),
                "contract_years": _safe_int(r.get("contract_years")),
                "total_value":    _safe_float(r.get("total_value_m")),
                "aav":            _safe_float(r.get("aav_m")),
                "value_metric":   _safe_float(r.get("value_metric")),
                "value_tier":     _safe_str(r.get("value_tier")),
                "signing_month":  signing_month,
            }
        )

    if skipped:
        logger.warning(
            "[T-06] upsert_fact_contracts: skipped %d rows with no player_id.", skipped
        )

    if not rows:
        logger.warning("[T-06] upsert_fact_contracts: no valid rows to insert.")
        return 0

    stmt = pg_insert(FactContract2026).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_fact_contract_player",
        set_={
            "new_team":       stmt.excluded.new_team,
            "contract_years": stmt.excluded.contract_years,
            "total_value":    stmt.excluded.total_value,
            "aav":            stmt.excluded.aav,
            "value_metric":   stmt.excluded.value_metric,
            "value_tier":     stmt.excluded.value_tier,
            "signing_month":  stmt.excluded.signing_month,
        },
    )

    with Session(engine) as session:
        session.execute(stmt)
        session.commit()

    logger.info("[T-06] upsert_fact_contracts: %d rows upserted.", len(rows))
    return len(rows)


# ---------------------------------------------------------------------------
# Convenience: confirm row counts
# ---------------------------------------------------------------------------
def print_row_counts() -> None:
    """Print current row counts for all three tables."""
    tables = ["dim_players", "fact_performance_2025", "fact_contracts_2026"]
    print("\n" + "=" * 50)
    print("T-06 ROW COUNTS")
    print("=" * 50)
    with engine.connect() as conn:
        for tbl in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"  {tbl:<30}: {count:>6} rows")
    print("=" * 50 + "\n")
