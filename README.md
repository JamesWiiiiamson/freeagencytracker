# NFL Free Agency Value Tracker

Automated ETL pipeline for tracking the best value in NFL Free Agency. This project extracts player performance data, pulls new contract signing data, matching players based on their EPA (Expected Points Added), and surfaces actionable insights for a Power BI dashboard.

## Features

- **Automated Data Extraction**: 
  - Player statistics and metrics from `nfl_data_py`.
  - Seasonal roster tracking & unsigned FA status from `nflverse`.
  - Live contract and transaction updates from the Sportradar Transactions API.
- **Robust Transformation Engine**: Applies advanced fuzzy matching logic to join contract data to performance data, categorizing players into performance tiers based on their value index.
- **Relational PostgreSQL Database**: Ensures data consistency using local PostgreSQL schemas.
- **Scheduled Automated Syncing**: Includes a completely hands-off daily workflow driven by Windows Task Scheduler and batch scripts.
- **Power BI Integrated Reporting**: Outputs refined `.csv` datasets explicitly meant to back Power BI dashboards for finding market inefficiencies and bargains.
- **Heuristic Availability Detection**: Bypasses traditional roster-sync lag by querying explicit transactional metadata (UFA, RFA, CUT) to identify true free agents in real-time.

## Pipeline Architecture

1. **Clean/Init**: Resets or initializes PostgreSQL tables via `database.py`.
2. **Extract**: Pulls 2025 seasonal stats and 2026 active rosters.
3. **Transform**: Computes a specific value metric dividing `epa_per_play` by `aav_m`.
4. **Load**: Upserts fact and dimensional tables (protecting against duplicates).
5. **Validate**: Performs row count comparisons versus typical design parameters.
6. **Report**: Generates the final metrics and spits out Power BI-ready datasets. 

### Prerequisites
- Python 3.12+
- PostgreSQL server installed and running.
- Valid API keys assigned in your local `.env`.

