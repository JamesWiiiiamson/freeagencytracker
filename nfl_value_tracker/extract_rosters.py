import pandas as pd
import nfl_data_py as nfl

def extract_rosters(season: int = 2026) -> pd.DataFrame:
    """
    Fetch the 2026 NFL roster via nfl_data_py.
    Returns a DataFrame of players with their current teams.
    """
    print(f"Fetching official rosters for {season}...")
    try:
        rosters = nfl.import_seasonal_rosters([season])
    except Exception as exc:
        print(f"Failed to fetch rosters: {exc}")
        return pd.DataFrame()

    if "player_id" not in rosters.columns or "team" not in rosters.columns:
        raise KeyError("Expected 'player_id' and 'team' in roster columns")

    # Keep only players with an assigned team, and include their free agency status
    df = rosters[["player_id", "team", "status"]].dropna(subset=["team", "player_id"])
    return df
