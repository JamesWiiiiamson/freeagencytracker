"""
T-03: Pull free-agency transactions for March 2026.

Strategy
--------
Sportradar's trial key does NOT populate contract financial fields.
We therefore pull *who signed where* from the Daily Transactions endpoint
and supplement financial data (contract_years, total_value_m, aav_m) from
manual_contracts.csv, which is populated from Spotrac / OverTheCap.

Data flow
---------
1. Iterate over every day in FREE_AGENCY_YEAR / FREE_AGENCY_MONTH.
2. For each day fetch /league/{year}/{month}/{day}/transactions.json.
3. Keep only signing transaction codes: SGN (signed), RSGN (re-signed).
4. Normalise to a flat DataFrame with columns:
       player_name | position | old_team | new_team | signed_date
5. Left-join the manual_contracts.csv on player_name to attach financials.
6. Return the merged DataFrame.

The entrypoint for pipeline use is extract_contracts().
"""

import calendar
import pandas as pd
from pathlib import Path

from config import FREE_AGENCY_YEAR, FREE_AGENCY_MONTH
from sportradar_client import SportradarClient

# Transaction codes we treat as "a signing occurred".
_SIGNING_CODES = {"SGN", "RSGN"}

# Path to the manually maintained CSV file with financial data.
_MANUAL_CSV = Path(__file__).with_name("manual_contracts.csv")

# Final column order expected downstream.
_FINAL_COLS = [
    "player_name",
    "position",
    "old_team",
    "new_team",
    "signed_date",
    "contract_years",
    "total_value_m",
    "aav_m",
]


def _fetch_day_transactions(
    client: SportradarClient, year: int, month: int, day: int
) -> list[dict]:
    """
    Fetch the raw transactions list for one calendar day.
    Returns an empty list on 404 (no data published for that day yet).
    """
    path = f"league/{year}/{month:02d}/{day:02d}/transactions.json"
    try:
        data = client.get(path)
        return data.get("players", [])
    except Exception as exc:
        # 404 on future / empty days is expected – swallow silently.
        if "404" in str(exc):
            return []
        raise


def _parse_players(players: list[dict], signed_date: str) -> list[dict]:
    """
    Extract signing rows from one day's player-transaction list.

    Each player entry looks like:
        {
          "name": "Rachaad White",
          "position": "RB",
          "transactions": [
            {
              "transaction_code": "SGN",
              "from_team": {"alias": "TB"},   # may be absent
              "to_team":   {"alias": "WAS"},
              ...
            }
          ]
        }
    """
    rows = []
    for player in players:
        for txn in player.get("transactions", []):
            if txn.get("transaction_code") not in _SIGNING_CODES:
                continue
            rows.append(
                {
                    "player_name": player.get("name", ""),
                    "position": player.get("position", ""),
                    "old_team": txn.get("from_team", {}).get("alias", ""),
                    "new_team": txn.get("to_team", {}).get("alias", ""),
                    "signed_date": signed_date,
                }
            )
    return rows


def extract_signings(
    year: int = FREE_AGENCY_YEAR,
    month: int = FREE_AGENCY_MONTH,
    client: SportradarClient | None = None,
) -> pd.DataFrame:
    """
    Pull all signing transactions for the given year/month.

    Returns a DataFrame with columns:
        player_name | position | old_team | new_team | signed_date
    """
    if client is None:
        client = SportradarClient()

    days_in_month = calendar.monthrange(year, month)[1]
    all_rows: list[dict] = []

    for day in range(1, days_in_month + 1):
        date_str = f"{year}-{month:02d}-{day:02d}"
        print(f"[T-03] Fetching transactions for {date_str} …")
        players = _fetch_day_transactions(client, year, month, day)
        all_rows.extend(_parse_players(players, date_str))

    if not all_rows:
        print("[T-03] No signing transactions found.")
        return pd.DataFrame(columns=["player_name", "position", "old_team", "new_team", "signed_date"])

    df = pd.DataFrame(all_rows)
    # Drop exact duplicates (same player signed on multiple records same day).
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"[T-03] Found {len(df)} unique signing events.")
    return df


def _load_manual_financials() -> pd.DataFrame:
    """
    Load manual_contracts.csv and normalise the player_name column for joining.
    Returns an empty DataFrame (with correct schema) if the file is missing.
    """
    if not _MANUAL_CSV.exists():
        print("[T-03] manual_contracts.csv not found – financial columns will be null.")
        return pd.DataFrame(columns=["player_name", "contract_years", "total_value_m", "aav_m"])

    manual = pd.read_csv(_MANUAL_CSV)
    required = {"player_name", "contract_years", "total_value_m", "aav_m"}
    missing = required - set(manual.columns)
    if missing:
        raise ValueError(f"manual_contracts.csv missing columns: {missing}")

    # Normalise names: lowercase + strip whitespace for fuzzy-safe join.
    manual["_join_key"] = manual["player_name"].str.lower().str.strip()
    return manual[["_join_key", "contract_years", "total_value_m", "aav_m"]]


def extract_contracts(
    year: int = FREE_AGENCY_YEAR,
    month: int = FREE_AGENCY_MONTH,
    client: SportradarClient | None = None,
) -> pd.DataFrame:
    """
    Orchestrates pulling signings from the API and merging manual financial data.

    Returns a DataFrame with the final shape:
        player_name | position | old_team | new_team | signed_date |
        contract_years | total_value_m | aav_m

    Financial columns will be NaN for players not present in manual_contracts.csv.
    """
    signings_df = extract_signings(year, month, client)

    if signings_df.empty:
        for col in ["contract_years", "total_value_m", "aav_m"]:
            signings_df[col] = pd.NA
        return signings_df[_FINAL_COLS]

    manual_df = _load_manual_financials()

    # Build a join key on the signings side.
    signings_df["_join_key"] = signings_df["player_name"].str.lower().str.strip()
    merged = signings_df.merge(manual_df, on="_join_key", how="left")
    merged = merged.drop(columns=["_join_key"])

    # Ensure all expected columns are present (they may be absent if manual is empty).
    for col in ["contract_years", "total_value_m", "aav_m"]:
        if col not in merged.columns:
            merged[col] = pd.NA

    result = merged[_FINAL_COLS].copy()
    n_with_financials = result["aav_m"].notna().sum()
    print(
        f"[T-03] Final contracts DataFrame: {len(result)} rows, "
        f"{n_with_financials} with financial data."
    )
    return result


if __name__ == "__main__":
    # Quick smoke-test: print the first 10 rows.
    df = extract_contracts()
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}")
    print(f"Null financials: {df['aav_m'].isna().sum()} / {len(df)}")
