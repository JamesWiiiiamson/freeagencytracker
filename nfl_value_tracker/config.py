"""
Central configuration.

Secrets (API keys, DB credentials) are loaded from the .env file in this
directory via python-dotenv.  Non-secret runtime constants live here directly.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this file.
load_dotenv(Path(__file__).with_name(".env"))

# ---------------------------------------------------------------------------
# Secrets — never hardcode these here; set them in .env
# ---------------------------------------------------------------------------
SPORTRADAR_NFL_KEY          = os.environ.get("SPORTRADAR_NFL_KEY", "")
SPORTRADAR_TRANSACTIONS_KEY = os.environ.get("SPORTRADAR_TRANSACTIONS_KEY", "")
DB_URL                      = os.environ.get("DB_URL", "")

# ---------------------------------------------------------------------------
# Sportradar
# ---------------------------------------------------------------------------
SPORTRADAR_BASE = "https://api.sportradar.com/nfl/official/trial/v7/en"

# ---------------------------------------------------------------------------
# Stats pipeline
# ---------------------------------------------------------------------------
SEASON_YEAR      = 2025
SEASON_TYPE      = "REG"
MIN_PLAYS        = 100
FUZZY_THRESHOLD  = 85
RATE_LIMIT_DELAY = 1.1

# ---------------------------------------------------------------------------
# Free agency window to ingest (inclusive)
# ---------------------------------------------------------------------------
FREE_AGENCY_YEAR  = 2026
FREE_AGENCY_MONTH = 3  # March
