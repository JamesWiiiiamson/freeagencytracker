"""
T-07: Bargain Query + Output Report

Run this script to query the database and generate local reporting for:
1. Top n Bargain Signings
2. Top n Overpaid Signings
3. Positional AAV/EPA efficiency averages
4. Top n Recent Signings Pending Financial Updates
"""

import pandas as pd
from sqlalchemy import text
from database import engine

def get_top_bargains(eng, n=10) -> pd.DataFrame:
    """
    Find the top N bargain signings using the value_metric.
    Includes the ::numeric cast required by PostgreSQL for ROUND().
    """
    query = text("""
        SELECT 
            dp.full_name,
            dp.position,
            fc.new_team,
            fc.contract_years,
            fc.total_value,
            fc.aav,
            fp.plays,
            ROUND(fp.epa_per_play::numeric, 4) AS epa_per_play,
            ROUND(fc.value_metric::numeric, 4) AS value_metric,
            fc.value_tier
        FROM dim_players dp
        JOIN fact_performance_2025 fp ON dp.player_id = fp.player_id
        JOIN fact_contracts_2026 fc ON dp.player_id = fc.player_id
        WHERE fp.plays >= 100
          AND fp.epa_per_play IS NOT NULL
          AND fc.aav > 0
          AND fc.value_metric IS NOT NULL
        ORDER BY fc.value_metric DESC
        LIMIT :limit_n;
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit_n": n})
    return df

def get_most_overpaid(eng, n=10) -> pd.DataFrame:
    """
    Find the top N most overpaid signings (reverse order of bargain).
    """
    query = text("""
        SELECT 
            dp.full_name,
            dp.position,
            fc.new_team,
            fc.contract_years,
            fc.total_value,
            fc.aav,
            fp.plays,
            ROUND(fp.epa_per_play::numeric, 4) AS epa_per_play,
            ROUND(fc.value_metric::numeric, 4) AS value_metric,
            fc.value_tier
        FROM dim_players dp
        JOIN fact_performance_2025 fp ON dp.player_id = fp.player_id
        JOIN fact_contracts_2026 fc ON dp.player_id = fc.player_id
        WHERE fp.plays >= 100
          AND fp.epa_per_play IS NOT NULL
          AND fc.aav > 0
          AND fc.value_metric IS NOT NULL
        ORDER BY fc.value_metric ASC
        LIMIT :limit_n;
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn, params={"limit_n": n})
    return df

def get_value_by_position(eng) -> pd.DataFrame:
    """
    Group by position to see which positions yielded the highest average value metric.
    """
    query = text("""
        SELECT 
            dp.position,
            COUNT(*) as distinct_players,
            ROUND(AVG(fc.aav)::numeric, 2) AS avg_aav,
            ROUND(AVG(fp.epa_per_play)::numeric, 4) AS avg_epa_per_play,
            ROUND(AVG(fc.value_metric)::numeric, 4) AS avg_value_metric
        FROM dim_players dp
        JOIN fact_performance_2025 fp ON dp.player_id = fp.player_id
        JOIN fact_contracts_2026 fc ON dp.player_id = fc.player_id
        WHERE fp.plays >= 100
          AND fp.epa_per_play IS NOT NULL
          AND fc.aav > 0
          AND fc.value_metric IS NOT NULL
        GROUP BY dp.position
        ORDER BY avg_value_metric DESC;
    """)
    with eng.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def export_report(eng) -> None:
    """
    Combines queries into terminal outputs and CSV exports.
    """
    print("=" * 80)
    print("Top Bargain Signings")
    print("=" * 80)
    bargains_df = get_top_bargains(eng, n=10)
    assert not bargains_df.isnull().values.any(), "Unexpected Null values in Bargains output."
    print(bargains_df.to_string(index=False) if not bargains_df.empty else "No matching bargain data.")

    print("\n" + "=" * 80)
    print("Exporting to bargain_signings_march2026.csv...")
    bargains_df.to_csv("bargain_signings_march2026.csv", index=False)
    print("Export complete.")

    print("\n" + "=" * 80)
    print("Top Overpaid Signings")
    print("=" * 80)
    overpaid_df = get_most_overpaid(eng, n=10)
    assert not overpaid_df.isnull().values.any(), "Unexpected Null values in Overpaid output."
    print(overpaid_df.to_string(index=False) if not overpaid_df.empty else "No matching overpaid data.")

    print("\n" + "=" * 80)
    print("Average Value Metric by Position")
    print("=" * 80)
    pos_df = get_value_by_position(eng)
    print(pos_df.to_string(index=False) if not pos_df.empty else "No matching positional data.")
    print("=" * 80)

if __name__ == "__main__":
    export_report(engine)
