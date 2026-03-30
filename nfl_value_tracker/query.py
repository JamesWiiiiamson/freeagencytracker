

import pandas as pd
from sqlalchemy import text
from database import engine

def get_top_unsigned_players() -> pd.DataFrame:
    """
    Find the top 10 most valuable players who haven't been signed yet.
    Based on epa_per_play where no contract exists in fact_contracts_2026.
    """
    query = text("""
        SELECT 
            dp.full_name,
            dp.position,
            dp.team_2025 AS previous_team,
            fp.plays,
            ROUND(fp.epa_per_play::numeric, 4) AS epa_per_play
        FROM dim_players dp
        JOIN fact_performance_2025 fp ON dp.player_id = fp.player_id
        LEFT JOIN fact_contracts_2026 fc ON dp.player_id = fc.player_id
        WHERE fc.player_id IS NULL
          AND fp.plays >= 100
          AND fp.epa_per_play IS NOT NULL
        ORDER BY fp.epa_per_play DESC
        LIMIT 10;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_overpaid_signings() -> pd.DataFrame:
    
    #Find the top 10 most overpaid signings (reverse order of bargain).
    
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
        LIMIT 10;
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_position_averages() -> pd.DataFrame:
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
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def main():
    print("=" * 70)
    print("Top 10 Most Valuable Unsigned Players")
    print("=" * 70)
    unsigned_df = get_top_unsigned_players()
    
    # Assertions
    assert not unsigned_df.isnull().values.any(), "Unexpected Null values present in Unsigned query output"
    
    print(unsigned_df.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("Exporting to top_unsigned_players_march2026.csv...")
    unsigned_df.to_csv("top_unsigned_players_march2026.csv", index=False)
    print("Export complete.")

    print("\n" + "=" * 70)
    print("Top 10 Overpaid Signings")
    print("=" * 70)
    overpaid_df = get_overpaid_signings()
    
    # Assertions
    assert not overpaid_df.isnull().values.any(), "Unexpected Null values present in Overpaid query output"
    
    print(overpaid_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("Average Value Metric by Position")
    print("=" * 70)
    pos_df = get_position_averages()
    
    print(pos_df.to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()
