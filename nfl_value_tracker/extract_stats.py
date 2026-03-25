"""
T-02: Pull 2025 season stats using nfl_data_py.
Completely free, no API key needed, no rate limits.

Two data sources are combined:
  1. Weekly player stats  - totals for TDs, yards, fantasy points, etc.
  2. Play-by-play data    - EPA per play and success rate per player.

These are merged on player_id to produce one row per player
with all metrics combined
"""

import nfl_data_py as nfl
import pandas as pd
from urllib.error import HTTPError


def extract_weekly_stats(season: int = 2025) -> pd.DataFrame:
    """
    Pull weekly player stats and aggregate to full season totals.
    Returns one row per player with summed stats across all weeks.
    """
    print(f"[T-02] Fetching weekly stats for {season}...")
    weekly = None
    weekly_source_season = season
    for candidate_season in range(season, 1998, -1):
        try:
            weekly = nfl.import_weekly_data([candidate_season])
            weekly_source_season = candidate_season
            if candidate_season != season:
                print(
                    "[T-02] Weekly 2025 data unavailable; "
                    f"using {candidate_season} as fallback."
                )
            break
        except HTTPError as exc:
            if exc.code != 404:
                raise

    if weekly is None:
        raise RuntimeError(
            "Unable to load weekly player stats for requested season or prior seasons."
        )

    # Identity columns — keep these as-is (don't sum them).
    # player_display_name holds the full name (e.g. 'Tua Tagovailoa') used for
    # cross-dataset matching; player_name is the abbreviated form ('T.Tagovailoa').
    id_cols = ["player_id", "player_name", "player_display_name", "position", "recent_team", "season"]

    # Only include columns that actually exist (older seasons may lack display_name).
    id_cols = [c for c in id_cols if c in weekly.columns]

    # Sum all other numeric columns across the season.
    numeric_cols = [
        col for col in weekly.select_dtypes("number").columns if col not in id_cols
    ]

    season_stats = (
        weekly.groupby(id_cols).agg({col: "sum" for col in numeric_cols}).reset_index()
    )
    season_stats["weekly_source_season"] = weekly_source_season

    print(f"[T-02] Weekly stats loaded for {len(season_stats)} players.")
    return season_stats


def extract_pbp_epa(season: int = 2025) -> pd.DataFrame:
    """
    Pull play-by-play data and compute EPA per play and success rate by player.
    Success rate is the percentage of plays where EPA > 0.
    """
    print(f"[T-02] Fetching play-by-play data for {season}...")
    print("[T-02] This takes about 60 seconds on first run...")
    pbp = nfl.import_pbp_data([season])

    # Filter to passing and rushing plays only.
    plays = pbp[pbp["play_type"].isin(["pass", "run"])].copy()

    def compute_epa(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
        """Compute EPA metrics for one player role."""
        return (
            df[df[id_col].notna()]
            .groupby(id_col)
            .agg(
                epa_total=("epa", "sum"),
                plays=("epa", "count"),
                success_count=("epa", lambda x: (x > 0).sum()),
            )
            .assign(
                epa_per_play=lambda d: d["epa_total"] / d["plays"],
                success_rate=lambda d: d["success_count"] / d["plays"],
            )
            .reset_index()
            .rename(columns={id_col: "player_id"})
        )

    passer_epa = compute_epa(plays, "passer_player_id")
    rusher_epa = compute_epa(plays, "rusher_player_id")
    receiver_epa = compute_epa(plays, "receiver_player_id")

    # Keep the role with the highest play count for players appearing in multiple roles.
    combined = pd.concat([passer_epa, rusher_epa, receiver_epa], ignore_index=True)
    epa_df = (
        combined.sort_values("plays", ascending=False)
        .drop_duplicates(subset="player_id", keep="first")
        .reset_index(drop=True)
    )

    print(f"[T-02] EPA computed for {len(epa_df)} players.")
    return epa_df


def extract_nfl_stats(season: int = 2025) -> pd.DataFrame:
    """
    Orchestrate both pulls and merge into one DataFrame.
    This is the entrypoint intended for pipeline usage.
    """
    weekly_df = extract_weekly_stats(season)
    epa_df = extract_pbp_epa(season)

    # Left join keeps all weekly players even if no pass/run events exist in PBP.
    merged = weekly_df.merge(
        epa_df[["player_id", "plays", "epa_total", "epa_per_play", "success_rate"]],
        on="player_id",
        how="left",
    )

    print(f"[T-02] Final stats DataFrame: {len(merged)} players.")

    # Expose the full player name as 'player_name' for cross-dataset joining.
    # nfl_data_py's 'player_name' is abbreviated ('A.Rodgers'); 'player_display_name'
    # is the full name ('Aaron Rodgers') which matches Sportradar's format.
    if "player_display_name" in merged.columns:
        merged = merged.rename(columns={"player_name": "player_name_abbr",
                                        "player_display_name": "player_name"})
    return merged


def find_all_null_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are fully null (systemic nulls)."""
    return [col for col in df.columns if df[col].isna().all()]
