from __future__ import annotations

import re
import logging
from typing import Optional

import pandas as pd
from thefuzz import fuzz, process as fuzz_process

from config import FUZZY_THRESHOLD

logger = logging.getLogger(__name__)

# Suffixes that commonly differ between datasets (strip before comparing).
_SUFFIX_RE = re.compile(
    r"\b(jr\.?|sr\.?|ii|iii|iv|v)\b", flags=re.IGNORECASE
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


def _normalise(name: str) -> str:
    """
    Return a canonical form of a player name for comparison:
      1. Lowercase + strip
      2. Remove common generational suffixes (Jr., II, etc.)
      3. Strip all non-alphanumeric characters (handles D.J. → dj)
      4. Collapse multiple spaces
    """
    name = name.lower().strip()
    name = _SUFFIX_RE.sub("", name)
    name = _NON_ALNUM_RE.sub("", name)
    return " ".join(name.split())  # collapse runs of whitespace




def _build_lookup(stats_df: pd.DataFrame) -> dict[str, str]:
    """
    Build a {normalised_name: original_name} dict from the stats DataFrame.
    Allows O(1) exact-path look-ups.
    """
    return {_normalise(name): name for name in stats_df["player_name"].dropna()}


def _match_one(
    contracts_name: str,
    lookup: dict[str, str],
    fuzzy_choices: list[str],
    threshold: int,
) -> tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Attempt to find the best match for a single contracts-side player name.

    Returns
    -------
    (matched_stats_name, score, method)
      matched_stats_name : the original stats-side name if matched, else None
      score              : 100 for exact, fuzzy score, or None
      method             : "exact", "fuzzy", or None
    """
    key = _normalise(contracts_name)

    # --- Phase 1: exact fast path ---
    if key in lookup:
        return lookup[key], 100, "exact"

    # --- Phase 2: fuzzy fallback ---
    if not fuzzy_choices:
        return None, None, None

    result = fuzz_process.extractOne(
        key,
        fuzzy_choices,          # these are already normalised
        scorer=fuzz.token_sort_ratio,
    )

    if result is None:
        logger.warning(
            "NO_FUZZY_CANDIDATE | contracts: %r", contracts_name
        )
        return None, None, None

    best_key, score = result[0], result[1]

    if score >= threshold:
        return lookup[best_key], score, "fuzzy"

    # Below threshold — log and return unmatched.
    # Find the best *original* candidate name for the log.
    best_original = lookup.get(best_key, best_key)
    logger.warning(
        "BELOW_THRESHOLD | contracts: %r | best_candidate: %r | score: %d | threshold: %d",
        contracts_name,
        best_original,
        score,
        threshold,
    )
    return None, score, None


def match_and_merge(
    contracts_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    threshold: int = FUZZY_THRESHOLD,
) -> pd.DataFrame:
    """
    Join contracts_df (left) onto stats_df (right) by player name.

    Parameters
    ----------
    contracts_df : DataFrame from extract_contracts.extract_contracts()
    stats_df     : DataFrame from extract_stats.extract_nfl_stats()
    threshold    : Fuzzy score floor (0–100); pulled from config.FUZZY_THRESHOLD.

    Returns
    -------
    DataFrame with every contracts row preserved (left join semantics).
    Unmatched rows have NaN for all stats columns.
    Extra columns added: match_score (int), match_method (str).
    """
    if "player_name" not in contracts_df.columns:
        raise ValueError("contracts_df is missing 'player_name' column.")
    if "player_name" not in stats_df.columns:
        raise ValueError("stats_df is missing 'player_name' column.")

    lookup = _build_lookup(stats_df)
    # Pre-build the sorted list of normalised keys for fuzzy search.
    fuzzy_choices = sorted(lookup.keys())

    matched_stats_names: list[Optional[str]] = []
    scores: list[Optional[int]] = []
    methods: list[Optional[str]] = []

    for contracts_name in contracts_df["player_name"]:
        stats_name, score, method = _match_one(
            contracts_name, lookup, fuzzy_choices, threshold
        )
        matched_stats_names.append(stats_name)
        scores.append(score)
        methods.append(method)

    # Attach match metadata to contracts.
    work = contracts_df.copy()
    work["_stats_name"] = matched_stats_names
    work["match_score"] = scores
    work["match_method"] = methods

    # Merge with stats on the resolved stats-side name.
    stats_cols = [c for c in stats_df.columns if c != "player_name"]
    stats_keyed = stats_df.rename(columns={"player_name": "_stats_name"})

    # Guard: deduplicate stats on the join key to prevent fan-out (one contracts
    # row matching multiple stats rows for the same player).  If the stats
    # DataFrame has duplicate player names (e.g. QB listed as both QB and KN),
    # aggregate numeric columns with mean and drop duplicates elsewhere.
    dupes = stats_keyed["_stats_name"].duplicated(keep=False)
    if dupes.any():
        dup_names = stats_keyed.loc[dupes, "_stats_name"].unique().tolist()
        logger.warning(
            "[T-04] %d duplicate player name(s) in stats — aggregating: %s",
            len(dup_names), dup_names[:10],
        )
        numeric_cols = stats_keyed.select_dtypes("number").columns.tolist()
        non_numeric = [c for c in stats_cols if c not in numeric_cols]
        agg_dict = {c: "mean" for c in numeric_cols}
        agg_dict.update({c: "first" for c in non_numeric if c != "_stats_name"})
        stats_keyed = (
            stats_keyed.groupby("_stats_name", as_index=False)
            .agg(agg_dict)
            .reset_index(drop=True)
        )

    merged = work.merge(
        stats_keyed[["_stats_name"] + stats_cols],
        on="_stats_name",
        how="left",
    )
    merged = merged.drop(columns=["_stats_name"])


    # Summary log.
    n_exact = (merged["match_method"] == "exact").sum()
    n_fuzzy = (merged["match_method"] == "fuzzy").sum()
    n_none  = merged["match_method"].isna().sum()
    logger.info(
        "[T-04] Match summary | exact: %d | fuzzy: %d | unmatched: %d | total: %d",
        n_exact, n_fuzzy, n_none, len(merged),
    )
    print(
        f"[T-04] Match summary -> exact: {n_exact} | fuzzy: {n_fuzzy} "
        f"| unmatched: {n_none} | total: {len(merged)}"
    )

    return merged


# ---------------------------------------------------------------------------
# T-05 — Value metric calculation
# ---------------------------------------------------------------------------

# Tier boundaries (value_metric = epa_per_play / aav_m).
# These are intentionally generous to keep all brackets populated even for
# small datasets.  Tune thresholds once real data is loaded.
_TIER_BINS   = [-float("inf"), 0.0, 0.05, 0.15, float("inf")]
_TIER_LABELS = ["overpaid", "fair", "solid", "elite"]


def add_value_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append ``value_metric`` and ``value_tier`` columns to *df* (in-place copy).

    value_metric = epa_per_play / aav_m

    Edge-case handling
    ------------------
    * ``aav_m`` == 0  → NaN  (avoids ZeroDivisionError; player is on minimum)
    * ``epa_per_play`` is NaN (unmatched player) → NaN propagates naturally
    * Negative EPA is legal — overpaid players should surface as negative values

    value_tier is assigned via ``pd.cut`` using pre-defined bin edges so that
    the boundary semantics are stable regardless of dataset size.

    Parameters
    ----------
    df : DataFrame returned by ``match_and_merge``; must contain
         ``epa_per_play`` and ``aav_m`` columns.

    Returns
    -------
    New DataFrame with two extra columns: ``value_metric``, ``value_tier``.
    """
    required = {"epa_per_play", "aav_m"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"add_value_metrics: missing required columns: {missing}")

    out = df.copy()

    # Replace 0 AAV with NaN so division returns NaN instead of ±inf.
    safe_aav = out["aav_m"].replace(0, float("nan"))

    # epa_per_play NaN → NaN (propagates); negative EPA → negative metric (kept).
    out["value_metric"] = out["epa_per_play"] / safe_aav

    # Tier bucketing — include_lowest so the -inf boundary is inclusive.
    out["value_tier"] = pd.cut(
        out["value_metric"],
        bins=_TIER_BINS,
        labels=_TIER_LABELS,
        include_lowest=True,
    )

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    tier_counts = out["value_tier"].value_counts().reindex(_TIER_LABELS, fill_value=0)
    n_null_metric = out["value_metric"].isna().sum()

    print("\n" + "=" * 60)
    print("T-05 VALUE METRIC SUMMARY")
    print("=" * 60)
    for tier, count in tier_counts.items():
        print(f"  {tier:<10}: {count}")
    print(f"  {'null':<10}: {n_null_metric}")
    print(f"  {'total':<10}: {len(out)}")
    print("=" * 60)

    # -----------------------------------------------------------------------
    # Correctness assertion
    # -----------------------------------------------------------------------
    # Any row with a real EPA value AND a non-zero AAV must have a non-null
    # value_metric after the calculation above.
    has_epa  = out["epa_per_play"].notna()
    has_aav  = out["aav_m"].notna() & (out["aav_m"] != 0)
    bad_rows = out[has_epa & has_aav & out["value_metric"].isna()]
    if not bad_rows.empty:
        raise AssertionError(
            f"[T-05] {len(bad_rows)} player(s) with valid EPA + valid AAV "
            f"ended up with null value_metric:\n{bad_rows[['player_name', 'epa_per_play', 'aav_m', 'value_metric']]}"
        )
    print("[T-05] Assertion passed — no valid-EPA + valid-AAV rows have null value_metric ✓")

    logger.info(
        "[T-05] value_metric computed | tiers: %s | null_metric: %d",
        tier_counts.to_dict(), n_null_metric,
    )
    return out


def run_pattern_tests() -> None:

    print("\n" + "=" * 60)
    print("T-04 PATTERN TESTS")
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

    # --- T-05: value metrics ---
    valued = add_value_metrics(merged)
    print(f"\nSample value metrics (top 10 by value_metric):")
    cols = ["player_name", "aav_m", "epa_per_play", "value_metric", "value_tier"]
    available_cols = [c for c in cols if c in valued.columns]
    print(
        valued.sort_values("value_metric", ascending=False)
              .head(10)[available_cols]
              .to_string(index=False)
    )
