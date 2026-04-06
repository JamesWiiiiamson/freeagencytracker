import os
import pandas as pd
from datetime import date, timedelta
from sqlalchemy import text
from database import engine

def export_powerbi():
    os.makedirs('outputs', exist_ok=True)
    
    query = """
    SELECT 
        dp.player_id,
        dp.full_name,
        dp.position,
        f.plays,
        f.epa_per_play,
        f.success_rate,
        f.performance_index,
        f.performance_grade,
        fa.is_signed,
        fa.signed_team,
        fa.signed_date
    FROM dim_players dp
    JOIN fact_performance_2025 f ON dp.player_id = f.player_id
    JOIN dim_free_agent_status fa ON dp.player_id = fa.player_id
    WHERE f.performance_index IS NOT NULL
    """
    
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
        
    print(f"Exporting {len(df)} total player records to PowerBI formats...")
    
    # 1. Top 10 Unsigned Tracker
    unsigned = df[df['is_signed'] == False].copy()
    
    top_10 = unsigned.sort_values(by='performance_index', ascending=False).head(10)
    top_10.to_csv('outputs/unsigned_tracker.csv', index=False)
    print(f"Created outputs/unsigned_tracker.csv with {len(top_10)} rows.")
    
    # 2. Market Efficiency Matrix (all unsigned with enough plays)
    # The >= 100 plays constraint is already handled in transform.py which sets performance_index to NULL if not met
    unsigned.to_csv('outputs/market_efficiency_matrix.csv', index=False)
    print(f"Created outputs/market_efficiency_matrix.csv with {len(unsigned)} rows.")
    
    # 3. Positional Z-Scores vs Unsigned Peers
    def calc_z(s):
        m = s.mean()
        st = s.std()
        if pd.isna(st) or st < 1e-9:
            return pd.Series(0.0, index=s.index)
        return (s - m) / st
        
    unsigned['perf_z_score_unsigned'] = unsigned.groupby('position')['performance_index'].transform(calc_z)
    z_scores_df = unsigned[['player_id', 'full_name', 'position', 'performance_index', 'perf_z_score_unsigned']].copy()
    z_scores_df = z_scores_df.sort_values(by=['position', 'perf_z_score_unsigned'], ascending=[True, False])
    z_scores_df.to_csv('outputs/positional_z_scores.csv', index=False)
    print(f"Created outputs/positional_z_scores.csv with {len(z_scores_df)} rows.")
    
    # 4. Recent Signings Log
    yesterday = date.today() - timedelta(days=1)
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    recently_signed = df[(df['is_signed'] == True) & (df['signed_date'].fillna('') >= yesterday_str)].copy()
    recently_signed.to_csv('outputs/recent_signings_log.csv', index=False)
    print(f"Created outputs/recent_signings_log.csv with {len(recently_signed)} rows.")
    print("PowerBI Data Export Complete.")

if __name__ == '__main__':
    export_powerbi()
