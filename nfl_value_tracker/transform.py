from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd
from thefuzz import fuzz, process as fuzz_process

from config import FUZZY_THRESHOLD, MIN_PLAYS

logger = logging.getLogger(__name__)




# ---------------------------------------------------------------------------
# Performance index (on-field only — no contract ROI)
# ---------------------------------------------------------------------------

_GRADE_LABELS = ["A+", "A", "B+", "B", "C", "D", "F"]


def _stats_position_key(df: pd.DataFrame) -> pd.Series:
    """Use nflverse position from the stats merge (handles position_x / position_y)."""
    if "position_y" in df.columns:
        s = df["position_y"]
    elif "position" in df.columns:
        s = df["position"]
    else:
        s = pd.Series("__UNK__", index=df.index)
    return (
        s.fillna("__UNK__")
        .astype(str)
        .str.strip()
        .replace("", "__UNK__")
    )


def _touchdown_series(df: pd.DataFrame) -> pd.Series:
    parts: list[pd.Series] = []
    for c in ("passing_tds", "rushing_tds", "receiving_tds"):
        if c in df.columns:
            parts.append(pd.to_numeric(df[c], errors="coerce").fillna(0))
    if parts:
        t = parts[0]
        for p in parts[1:]:
            t = t + p
        return t
    td_cols = [c for c in df.columns if "touchdown" in c.lower()]
    if td_cols:
        return df[td_cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return pd.Series(0.0, index=df.index, dtype=float)


def _group_z(s: pd.Series) -> pd.Series:
    m = float(s.mean())
    st = float(s.std(ddof=0))
    if st != st or st < 1e-9:
        return pd.Series(0.0, index=s.index)
    return (s - m) / st


def _z_to_grade(z: float) -> str | None:
    if z is None or (isinstance(z, float) and z != z):
        return None
    if z >= 1.5:
        return "A+"
    if z >= 1.0:
        return "A"
    if z >= 0.5:
        return "B+"
    if z >= 0.0:
        return "B"
    if z >= -0.5:
        return "C"
    if z >= -1.0:
        return "D"
    return "F"


def add_performance_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``performance_index`` and ``performance_grade`` from prior-season stats.

    Grades are driven by **position-relative EPA per play** (z-score within
    each position among rows in this frame — typically March FA signings).

    ``performance_index`` blends, with position-relative z-scores:
      ~ efficiency: epa_per_play
      ~ volume: log1p(plays), epa_total
      ~ impact: success_rate, log1p(touchdowns)

    Rows without matched stats (no EPA / plays) get null performance fields.

    Note
    ----
    Stats are from the **prior** nflverse season (e.g. 2025). They proxy
    “how good the player was recently,” not literal production for the **new**
    team until you load post-signing game data.
    """
    required = {"epa_per_play", "plays"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"add_performance_index: missing columns: {missing}")

    out = df.copy()
    out["performance_index"] = np.nan
    out["performance_grade"] = pd.NA

    epa_pp = pd.to_numeric(out["epa_per_play"], errors="coerce")
    plays = pd.to_numeric(out["plays"], errors="coerce")
    epa_tot = (
        pd.to_numeric(out["epa_total"], errors="coerce")
        if "epa_total" in out.columns
        else epa_pp * plays
    )
    succ = (
        pd.to_numeric(out["success_rate"], errors="coerce")
        if "success_rate" in out.columns
        else pd.Series(np.nan, index=out.index)
    )
    tds = _touchdown_series(out)

    valid = epa_pp.notna() & plays.notna() & (plays >= MIN_PLAYS) & succ.notna()
    if not valid.any():
        logger.warning(f"add_performance_index: no rows with plays >= {MIN_PLAYS}, EPA, and Success Rate.")
        return out

    w = out.loc[valid].copy()
    idx = w.index
    w["_pk"] = _stats_position_key(w)
    w["_epa_pp"] = epa_pp.loc[idx]
    w["_succ"] = succ.loc[idx]

    w["_perf_idx"] = (0.7 * w["_epa_pp"]) + (0.3 * w["_succ"])

    # We still assign grades based on positional z-score of the new performance index
    gkey = "_pk"
    w["_z_perf"] = w.groupby(gkey, group_keys=False)["_perf_idx"].transform(_group_z)

    grades = w["_z_perf"].apply(
        lambda x: _z_to_grade(float(x)) if pd.notna(x) else pd.NA
    )
    out.loc[w.index, "performance_index"] = w["_perf_idx"]
    out.loc[w.index, "performance_grade"] = grades

    print("\n" + "=" * 60)
    print("PERFORMANCE INDEX (on-field, no contract ROI)")
    print("=" * 60)
    gcounts = (
        out["performance_grade"].value_counts().reindex(_GRADE_LABELS, fill_value=0)
    )
    for g, c in gcounts.items():
        print(f"  {g:<4}: {c}")
    print(f"  null index: {out['performance_index'].isna().sum()} / {len(out)}")
    print("=" * 60)

    logger.info(
        "performance_index computed | grade distribution: %s",
        gcounts.to_dict(),
    )
    return out


def run_pattern_tests() -> None:

    print("\n" + "=" * 60)
    print("PATTERN TESTS")
    print("=" * 60)

    # Synthetic 'stats' side.
    stats = pd.DataFrame({
        "player_name": [
            "Calvin Austin",
            "CJ Gardner-Johnson",
            "Mitchell Trubisky",
            "Aaron Rodgers",
        ],
        "passing_yards": [0, 800, 3100, 4200],
    })

    # Synthetic 'contracts' side (names as they appear in the Sportradar feed).
    contracts = pd.DataFrame({
        "player_name": [
            "Calvin Austin III",          # suffix difference
            "C.J. Gardner-Johnson",       # punctuation difference
            "Mitch Trubisky",             # nickname difference
            "Completely Made Up Player",  # true non-match
        ],
        "new_team": ["NYG", "BUF", "PIT", "???"],
        "aav_m": [4.0, 12.0, 3.0, 0.0],
    })

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    result = match_and_merge(contracts, stats, threshold=FUZZY_THRESHOLD)

    for _, row in result.iterrows():
        is_matched = not pd.isna(row["match_method"])
        status = row["match_method"] if is_matched else "UNMATCHED"
        score = row["match_score"]
        score_str = f"(score={int(score)})" if pd.notna(score) else ""
        stats_val = row.get("passing_yards")
        tag = "[OK]" if is_matched else "[!!]"
        print(
            f"  {tag}  "
            f"{row['player_name']:<30} -> {status:<6} {score_str:<12} "
            f"passing_yards={stats_val}"
        )

    # Assertions — use pd.isna() because pandas stores None as NaN in
    # mixed-type object columns after merging.
    assert not pd.isna(result.loc[0, "match_method"]), "Suffix test FAILED"
    assert not pd.isna(result.loc[1, "match_method"]), "Punctuation test FAILED"
    assert not pd.isna(result.loc[2, "match_method"]), "Nickname test FAILED"
    assert pd.isna(result.loc[3, "match_method"]),     "Non-match test FAILED"
    print("\nAll 4 pattern tests passed [OK]")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    import sys

    # --- Pattern tests first ---
    run_pattern_tests()

    # --- Real-data smoke test ---
    print("Loading real data …")
    try:
        from extract_stats import extract_nfl_stats
        from extract_contracts import extract_contracts

        stats_df = extract_nfl_stats(season=2025)
        contracts_df = extract_contracts()
    except Exception as exc:
        print(f"Could not load real data: {exc}")
        sys.exit(1)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s | %(message)s",
    )

    merged = match_and_merge(contracts_df, stats_df)

    # Print borderline matches (85–92) for manual verification.
    borderline = merged[
        merged["match_score"].notna()
        & (merged["match_score"] >= 85)
        & (merged["match_score"] <= 92)
    ][["player_name", "match_score", "match_method", "new_team", "aav_m"]].copy()

    if borderline.empty:
        print("\nNo borderline matches in 85–92 band.")
    else:
        print(f"\n{'='*60}")
        print(f"Borderline matches (85–92) — manual review required")
        print(f"{'='*60}")
        print(borderline.to_string(index=False))

    print(f"\nFinal shape: {merged.shape}")
    print(f"Rows with stats data: {merged['match_method'].notna().sum()}")
    print(f"Rows without stats (unmatched): {merged['match_method'].isna().sum()}")
    assert len(merged) == len(contracts_df), (
        f"Row count mismatch! contracts={len(contracts_df)}, merged={len(merged)}"
    )
    print("\nRow-count assertion passed ✓")

    # --- performance index (on-field) ---
    valued = add_performance_index(merged)
    print("\nSample performance (top 10 by performance_index):")
    cols = [
        "player_name",
        "new_team",
        "epa_per_play",
        "plays",
        "performance_index",
        "performance_grade",
    ]
    available_cols = [c for c in cols if c in valued.columns]
    print(
        valued.sort_values("performance_index", ascending=False, na_position="last")
        .head(10)[available_cols]
        .to_string(index=False)
    )
