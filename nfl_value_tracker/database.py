"""
T-06: SQLAlchemy schema + PostgreSQL loader.

Tables
------
dim_players          – one row per player (PK: player_id from nfl_data_py)
fact_performance_2025 – one row per player with 2025 EPA/fantasy stats
dim_free_agent_status - tracking FA signings

Upsert strategy
---------------
Loaders use INSERT … ON CONFLICT (player_id) DO UPDATE so that
re-running the pipeline updates existing rows rather than duplicating them.
"""

from __future__ import annotations
import logging
from typing import Optional

import pandas as pd
from sqlalchemy import (
    Boolean,
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
    case,
    DateTime
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Session
from config import DB_URL

logger = logging.getLogger(__name__)

engine = create_engine(DB_URL, echo=False)

class Base(DeclarativeBase):
    pass

class DimPlayer(Base):
    __tablename__ = "dim_players"
    player_id = Column(Text, primary_key=True)
    full_name = Column(String(100), nullable=False)
    position  = Column(String(10), nullable=True)
    age       = Column(SmallInteger, nullable=True)
    team_2025 = Column(String(10), nullable=True)

class FactPerformance2025(Base):
    __tablename__ = "fact_performance_2025"
    __table_args__ = (UniqueConstraint("player_id", name="uq_fact_perf_player"),)

    id           = Column(Integer, primary_key=True, autoincrement=True)
    player_id    = Column(Text, ForeignKey("dim_players.player_id"), nullable=False)
    season       = Column(SmallInteger, nullable=True)
    plays        = Column(Integer, nullable=True)
    epa_total    = Column(Float, nullable=True)
    epa_per_play = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    touchdowns   = Column(Integer, nullable=True)
    fantasy_pts  = Column(Float, nullable=True)
    performance_index = Column(Float, nullable=True)
    performance_grade = Column(String(10), nullable=True)

class DimFreeAgentStatus(Base):
    __tablename__ = "dim_free_agent_status"
    __table_args__ = (UniqueConstraint("player_id", name="uq_fa_status_player"),)

    id               = Column(Integer, primary_key=True, autoincrement=True)
    player_id        = Column(Text, ForeignKey("dim_players.player_id"), nullable=False)
    is_signed        = Column(Boolean, nullable=False, default=False)
    signed_team      = Column(String(10), nullable=True)
    signed_date      = Column(String(10), nullable=True)
    aav_m            = Column(Float, nullable=True)
    total_value_m    = Column(Float, nullable=True)
    contract_years   = Column(SmallInteger, nullable=True)
    transaction_type = Column(String(50), nullable=True)
    days_unsigned    = Column(Integer, nullable=True)
    last_updated     = Column(DateTime(timezone=True), nullable=True)

_idx_perf_player = Index("ix_fact_perf_player_id", FactPerformance2025.player_id)
_idx_fa_status   = Index("ix_dim_fa_status_player_id", DimFreeAgentStatus.player_id)

def drop_all_tables() -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS fact_contracts_2026 CASCADE;"))
    Base.metadata.drop_all(bind=engine)
    logger.info("drop_all_tables: all ORM tables dropped.")

def init_db() -> None:
    Base.metadata.create_all(engine)
    _ensure_pg_schema_extras()
    logger.info("init_db complete")

def _ensure_pg_schema_extras() -> None:
    stmts = [
        "ALTER TABLE fact_performance_2025 ADD COLUMN IF NOT EXISTS performance_index DOUBLE PRECISION",
        "ALTER TABLE fact_performance_2025 ADD COLUMN IF NOT EXISTS performance_grade VARCHAR(10)",
    ]
    with engine.begin() as conn:
        for sql in stmts:
            conn.execute(text(sql))

def _safe_int(val) -> Optional[int]:
    try:
        if pd.isna(val): return None
    except: pass
    return int(val) if val is not None else None

def _safe_float(val, ndigits=6) -> Optional[float]:
    try:
        if pd.isna(val): return None
    except: pass
    return round(float(val), ndigits) if val is not None else None

def _safe_str(val) -> Optional[str]:
    try:
        if pd.isna(val): return None
    except: pass
    s = str(val).strip()
    return s if s else None

def upsert_dim_players(stats_df: pd.DataFrame) -> int:
    deduped = stats_df.copy()
    if "plays" in deduped.columns:
        deduped = deduped.sort_values("plays", ascending=False).drop_duplicates(subset=["player_id"], keep="first")
    else:
        deduped = deduped.drop_duplicates(subset=["player_id"], keep="first")

    rows = []
    for _, r in deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid: continue
        rows.append(
            {
                "player_id": pid,
                "full_name": _safe_str(r.get("player_name")),
                "position":  _safe_str(r.get("position")),
                "age":       _safe_int(r.get("age")),
                "team_2025": _safe_str(r.get("recent_team")),
            }
        )
    if not rows: return 0

    stmt = pg_insert(DimPlayer).values(rows)
    stmt = stmt.on_conflict_do_update(
        index_elements=["player_id"],
        set_={
            "full_name": stmt.excluded.full_name,
            "position":  stmt.excluded.position,
            "age":       stmt.excluded.age,
            "team_2025": stmt.excluded.team_2025,
        },
    )
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()
    return len(rows)

def upsert_fact_performance(stats_df: pd.DataFrame) -> int:
    deduped = stats_df.copy()
    if "season" not in deduped.columns: deduped["season"] = 2025
    deduped = deduped.drop_duplicates(subset=["player_id"], keep="first")

    rows = []
    for _, r in deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid: continue

        tds = r.get("touchdowns")
        if pd.isna(tds):
            keys = ["passing_tds", "rushing_tds", "receiving_tds"]
            tds = sum([_safe_int(r.get(k)) or 0 for k in keys if k in r])

        rows.append(
            {
                "player_id": pid,
                "season": _safe_int(r.get("season")),
                "plays": _safe_int(r.get("plays")),
                "epa_total": _safe_float(r.get("epa_total")),
                "epa_per_play": _safe_float(r.get("epa_per_play")),
                "success_rate": _safe_float(r.get("success_rate")),
                "touchdowns": _safe_int(tds),
                "fantasy_pts": _safe_float(r.get("fantasy_points_ppr")),
                "performance_index": _safe_float(r.get("performance_index")),
                "performance_grade": _safe_str(r.get("performance_grade")),
            }
        )

    if not rows: return 0
    stmt = pg_insert(FactPerformance2025).values(rows)
    stmt = stmt.on_conflict_do_update(
        constraint="uq_fact_perf_player",
        set_={
            "plays":        stmt.excluded.plays,
            "epa_total":    stmt.excluded.epa_total,
            "epa_per_play": stmt.excluded.epa_per_play,
            "success_rate": stmt.excluded.success_rate,
            "touchdowns":   stmt.excluded.touchdowns,
            "fantasy_pts":  stmt.excluded.fantasy_pts,
            "performance_index": stmt.excluded.performance_index,
            "performance_grade": stmt.excluded.performance_grade,
        },
    )
    with Session(engine) as session:
        session.execute(stmt)
        session.commit()
    return len(rows)

def upsert_free_agent_status(stats_df: pd.DataFrame, roster_df: pd.DataFrame, fa_start_date: str = "2026-03-12") -> tuple[int, int]:
    from datetime import date
    import datetime as _dt
    today = date.today()
    try: fa_start = date.fromisoformat(fa_start_date)
    except: fa_start = date(2026, 3, 12)
    days_since_fa = (today - fa_start).days

    signed_pids = set(roster_df["player_id"].dropna())
    team_map = roster_df.set_index("player_id")["team"].to_dict()
    status_map = roster_df.set_index("player_id")["status"].to_dict() if "status" in roster_df.columns else {}

    rows_unsigned, rows_signed = [], []
    stats_deduped = stats_df[["player_id", "player_name"]].dropna(subset=["player_id"]).drop_duplicates(subset=["player_id"])

    now_utc = _dt.datetime.now(_dt.timezone.utc)
    today_str = today.isoformat()
    fa_statuses = {'UFA', 'FREE', 'CUT', 'RFA'}

    for _, r in stats_deduped.iterrows():
        pid = _safe_str(r.get("player_id"))
        if not pid: continue

        # A player is unsigned if they aren't on the roster OR their status explicitly says they are available
        is_unsigned = (pid not in signed_pids) or (str(status_map.get(pid)).upper() in fa_statuses)

        if not is_unsigned:
            rows_signed.append({
                "player_id": pid, "is_signed": True, "signed_team": _safe_str(team_map.get(pid)),
                "signed_date": today_str, "aav_m": None, "total_value_m": None, "contract_years": None,
                "transaction_type": "roster_sync", "days_unsigned": 0, "last_updated": now_utc
            })
        else:
            rows_unsigned.append({
                "player_id": pid, "is_signed": False, "signed_team": None, "signed_date": None,
                "aav_m": None, "total_value_m": None, "contract_years": None, "transaction_type": None,
                "days_unsigned": max(days_since_fa, 0), "last_updated": now_utc
            })

    def _upsert_batch(batch):
        if not batch: return 0
        stmt = pg_insert(DimFreeAgentStatus).values(batch)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_fa_status_player",
            set_={
                "is_signed": stmt.excluded.is_signed,
                "signed_team": stmt.excluded.signed_team,
                "signed_date": case((DimFreeAgentStatus.is_signed.is_(False) & stmt.excluded.is_signed.is_(True), stmt.excluded.signed_date), else_=DimFreeAgentStatus.signed_date),
                "aav_m": stmt.excluded.aav_m,
                "total_value_m": stmt.excluded.total_value_m,
                "contract_years": stmt.excluded.contract_years,
                "transaction_type": stmt.excluded.transaction_type,
                "days_unsigned": stmt.excluded.days_unsigned,
                "last_updated": stmt.excluded.last_updated,
            },
        )
        with Session(engine) as session:
            session.execute(stmt)
            session.commit()
        return len(batch)

    n_signed = _upsert_batch(rows_signed)
    n_unsigned = _upsert_batch(rows_unsigned)
    return n_unsigned, n_signed

def print_row_counts() -> None:
    tables = ["dim_players", "fact_performance_2025", "dim_free_agent_status"]
    print("\\n==================================================")
    print("ROW COUNTS")
    print("==================================================")
    with engine.connect() as conn:
        for tbl in tables:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
            print(f"  {tbl:<30}: {count:>6} rows")
    print("==================================================\\n")
