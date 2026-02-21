"""
Monitor Engine
===============
Generic, registry-driven engine that processes any monitor config.

For each monitor it:
1. Resolves the report date (using SOT table to ensure data completeness)
2. Loads the SQL template and injects dimension-specific parameters
3. Pulls data from BigQuery for the current period and comparison periods
4. Computes ratio metrics from raw counts
5. Calculates WoW and YoY deltas
6. Cross-validates key metrics against the SOT table
7. Applies deterministic anomaly detection rules
8. Returns a structured MonitorResult

Usage:
    from libs.monitor_engine import MonitorEngine
    from libs.bq_client import get_bq_client

    engine = MonitorEngine(get_bq_client())
    result = engine.run("revenue_funnel", date="2026-02-05")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from libs.bq_client import bq_to_df, get_bq_client
from libs.monitor_registry import MONITOR_REGISTRY, get_query_path

SOT_TABLE = "tt-dp-prod.sot_analytics.2026_daily_metrics"

SOT_COLUMN_MAP = {
    "intentful_visitors": "total_iv",
    "requests": "total_requests",
    "projects": "total_projects",
    "revenue": "total_revenue",
}

VALIDATION_TOLERANCE = 0.002


# ===================================================================
# Data structures
# ===================================================================

@dataclass
class Anomaly:
    """A single flagged anomaly from deterministic rules."""
    metric: str          # e.g., "rpp"
    metric_label: str    # e.g., "Revenue per Project (RPP)"
    dimension: str       # e.g., "SEM" or "overall"
    comparison: str      # "wow" or "yoy"
    current_value: float
    prior_value: float
    pct_change: float    # e.g., -0.12 for -12%
    threshold: float     # the threshold that was breached
    severity: str        # "high" or "medium"
    direction: str       # "up" means higher is better


@dataclass
class ValidationResult:
    """Cross-validation of a single metric against the SOT table."""
    metric: str
    our_value: float
    sot_value: float
    pct_diff: float
    passed: bool


@dataclass
class MonitorResult:
    """Complete output from running one monitor."""
    domain: str                          # registry key, e.g., "revenue_funnel"
    name: str                            # human-readable name
    report_date: str                     # the date this report covers
    summary_df: pd.DataFrame             # aggregated metrics with deltas
    detail_df: pd.DataFrame              # per-dimension metrics with deltas
    anomalies: list[Anomaly]             # flagged issues
    context: str                         # domain description for the AI agent
    raw_data: dict[str, pd.DataFrame] = field(default_factory=dict)
    validations: list[ValidationResult] = field(default_factory=list)


# ===================================================================
# Dimension parameter injection
# ===================================================================

# Maps dimension names to the SQL column expressions for each CTE.
# Uses channel_map CTE (defined in the SQL template) for 4-channel taxonomy:
# O&O, SEM, SEO, Partnership.
DIMENSION_SQL = {
    "overall": {
        "dimension_col_iv": "'overall'",
        "dimension_col": "'overall'",
        "dimension_col_project": "'overall'",
        "dimension_col_contact": "'overall'",
        "dimension_col_revenue": "'overall'",
    },
    "channel": {
        "dimension_col_iv": "COALESCE(cm_iv.channel, 'Other')",
        "dimension_col": "COALESCE(cm.channel, 'Other')",
        "dimension_col_project": "COALESCE(cm.channel, 'Other')",
        "dimension_col_contact": "COALESCE(cm.channel, 'Other')",
        "dimension_col_revenue": "COALESCE(cm.channel, 'Other')",
    },
}


# ===================================================================
# Engine
# ===================================================================

class MonitorEngine:
    """
    Generic engine that runs any monitor from the registry.

    Parameters
    ----------
    client : bigquery.Client, optional
        Reuse a BQ client. If None, creates one.
    """

    def __init__(self, client=None):
        self.client = client or get_bq_client()

    def resolve_report_date(self, date: str | None = None) -> str:
        """
        Determine the report date, ensuring it has complete data.

        If no date is provided, queries the SOT table for the most recent
        complete day. If a date is provided, validates it exists in the SOT.
        """
        if date:
            check_sql = f"""
                SELECT metric_date
                FROM `{SOT_TABLE}`
                WHERE metric_date = '{date}'
            """
            df = bq_to_df(check_sql, client=self.client)
            if df.empty:
                print(f"  WARNING: Date {date} not found in SOT table — data may be incomplete.")
                print(f"           Proceeding anyway, but results may be partial.")
            return date

        latest_sql = f"""
            SELECT MAX(metric_date) AS latest_date
            FROM `{SOT_TABLE}`
        """
        df = bq_to_df(latest_sql, client=self.client)
        resolved = str(df["latest_date"].iloc[0])
        print(f"  Auto-resolved report date to {resolved} (latest complete day in SOT)")
        return resolved

    def run(self, monitor_key: str, date: str | None = None) -> MonitorResult:
        """
        Run a monitor and return structured results.

        Parameters
        ----------
        monitor_key : str
            Key in MONITOR_REGISTRY (e.g., "revenue_funnel").
        date : str, optional
            Report date in YYYY-MM-DD. Defaults to latest complete day in SOT.

        Returns
        -------
        MonitorResult
        """
        config = MONITOR_REGISTRY[monitor_key]
        report_date = self.resolve_report_date(date)

        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"Date: {report_date}")
        print(f"{'='*60}")

        # 1. Pull data for each dimension
        raw_data = {}
        for dim in config["dimensions"]:
            print(f"\n  Pulling data for dimension: {dim}")
            df = self._pull_data(config, dim, report_date)
            raw_data[dim] = df
            print(f"    Got {len(df):,} rows")

        # 2. Compute ratio metrics and deltas for each dimension
        detail_frames = []
        for dim, df in raw_data.items():
            df_with_metrics = self._compute_metrics(df, config)
            df_with_deltas = self._compute_deltas(df_with_metrics, config, report_date)
            df_with_deltas["dim_type"] = dim
            detail_frames.append(df_with_deltas)

        detail_df = pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame()

        # 3. Build summary (overall dimension, most recent date)
        summary_df = detail_df[
            (detail_df["dim_type"] == "overall") & (detail_df["dimension"] == "overall")
        ].copy()

        # 4. Cross-validate against SOT
        validations = self._validate_against_sot(summary_df, report_date)

        # 5. Detect anomalies
        anomalies = self._detect_anomalies(detail_df, config)
        print(f"\n  Anomalies detected: {len(anomalies)}")

        return MonitorResult(
            domain=monitor_key,
            name=config["name"],
            report_date=report_date,
            summary_df=summary_df,
            detail_df=detail_df,
            anomalies=anomalies,
            context=config["context"],
            raw_data=raw_data,
            validations=validations,
        )

    # ----- SOT Validation -----

    def _validate_against_sot(
        self, summary_df: pd.DataFrame, report_date: str
    ) -> list[ValidationResult]:
        """Compare our overall metrics against the SOT table for the same date."""
        validations = []
        try:
            sot_sql = f"""
                SELECT total_iv, total_requests, total_projects, total_revenue,
                       total_revenue_per_project
                FROM `{SOT_TABLE}`
                WHERE metric_date = '{report_date}'
            """
            sot_df = bq_to_df(sot_sql, client=self.client)
        except Exception as e:
            print(f"  WARNING: SOT validation skipped — {e}")
            return validations

        if sot_df.empty:
            print(f"  WARNING: No SOT data for {report_date}, skipping validation")
            return validations

        sot_row = sot_df.iloc[0]
        print(f"\n  Cross-validating against SOT ({SOT_TABLE}):")

        if summary_df.empty:
            print("    No summary data to validate")
            return validations

        our_row = summary_df.iloc[0]

        for our_col, sot_col in SOT_COLUMN_MAP.items():
            our_val = our_row.get(our_col, np.nan)
            sot_val = sot_row.get(sot_col, np.nan)
            if pd.isna(our_val) or pd.isna(sot_val) or sot_val == 0:
                continue
            pct_diff = abs(our_val - sot_val) / abs(sot_val)
            passed = pct_diff <= VALIDATION_TOLERANCE
            status = "PASS" if passed else "FAIL"
            print(f"    {our_col}: ours={our_val:,.2f}, SOT={sot_val:,.2f}, "
                  f"diff={pct_diff:.2%} [{status}]")
            validations.append(ValidationResult(
                metric=our_col,
                our_value=our_val,
                sot_value=sot_val,
                pct_diff=pct_diff,
                passed=passed,
            ))

        failed = [v for v in validations if not v.passed]
        if failed:
            print(f"  WARNING: {len(failed)} metric(s) failed SOT validation!")
        else:
            print(f"  All {len(validations)} metrics passed SOT validation")

        return validations

    # ----- Data Pull -----

    def _pull_data(self, config: dict, dimension: str, report_date: str) -> pd.DataFrame:
        """Load SQL template, inject parameters, execute query."""
        query_path = get_query_path(config["query_file"])
        with open(query_path, "r") as f:
            sql_template = f.read()

        # Date range: 14 days for WoW, 380 days for YoY (364 + 14 day buffer)
        comparisons = config.get("comparison", ["wow"])
        if "yoy" in comparisons:
            lookback_days = 380
        else:
            lookback_days = 21  # 3 weeks for WoW

        start_date = (
            datetime.strptime(report_date, "%Y-%m-%d") - timedelta(days=lookback_days)
        ).strftime("%Y-%m-%d")

        date_filter = f"BETWEEN '{start_date}' AND '{report_date}'"

        # Get dimension-specific SQL column expressions
        dim_params = DIMENSION_SQL.get(dimension, DIMENSION_SQL["overall"])

        sql = sql_template.format(
            date_filter=date_filter,
            **dim_params,
        )

        return bq_to_df(sql, client=self.client)

    # ----- Metric Computation -----

    def _compute_metrics(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Compute ratio metrics from raw counts."""
        df = df.copy()

        for metric_key, metric_def in config["metrics"].items():
            num_col = metric_def["numerator"]
            den_col = metric_def["denominator"]
            if num_col in df.columns and den_col in df.columns:
                df[metric_key] = np.where(
                    df[den_col] > 0,
                    df[num_col] / df[den_col],
                    np.nan,
                )
        return df

    # ----- Delta Computation -----

    def _compute_deltas(
        self, df: pd.DataFrame, config: dict, report_date: str
    ) -> pd.DataFrame:
        """
        Compute WoW and YoY deltas for each dimension value.

        Returns one row per dimension with current values, prior values, and pct changes.
        """
        if df.empty:
            return df

        df = df.copy()
        df["metric_date"] = pd.to_datetime(df["metric_date"]).dt.date

        rd = datetime.strptime(report_date, "%Y-%m-%d").date()
        comparisons = config.get("comparison", ["wow"])

        # Define periods
        # Current: the report date (single day)
        # WoW prior: 7 days before the report date
        # YoY prior: 364 days before the report date (preserves day-of-week)
        current_date = rd
        wow_prior_date = rd - timedelta(days=7)
        yoy_prior_date = rd - timedelta(days=364)

        # Columns to compare: ratio metrics + volume metrics
        metric_cols = list(config["metrics"].keys())
        volume_cols = config.get("volume_metrics", [])
        all_cols = [c for c in metric_cols + volume_cols if c in df.columns]

        # Get current day data
        df_current = df[df["metric_date"] == current_date].copy()
        if df_current.empty:
            # Try the most recent date available
            max_date = df["metric_date"].max()
            print(f"    No data for {report_date}, using latest: {max_date}")
            df_current = df[df["metric_date"] == max_date].copy()
            current_date = max_date
            wow_prior_date = max_date - timedelta(days=7)
            yoy_prior_date = max_date - timedelta(days=364)

        results = []
        for _, row in df_current.iterrows():
            dim_value = row["dimension"]
            result = {"dimension": dim_value, "metric_date": current_date}

            # Current values
            for col in all_cols:
                result[col] = row.get(col, np.nan)

            # WoW comparison
            if "wow" in comparisons:
                df_wow = df[
                    (df["metric_date"] == wow_prior_date) & (df["dimension"] == dim_value)
                ]
                for col in all_cols:
                    prior_val = df_wow[col].iloc[0] if not df_wow.empty and col in df_wow.columns else np.nan
                    result[f"{col}_wow_prior"] = prior_val
                    if pd.notna(prior_val) and prior_val != 0:
                        result[f"{col}_wow"] = (row.get(col, 0) - prior_val) / prior_val
                    else:
                        result[f"{col}_wow"] = np.nan

            # YoY comparison
            if "yoy" in comparisons:
                df_yoy = df[
                    (df["metric_date"] == yoy_prior_date) & (df["dimension"] == dim_value)
                ]
                for col in all_cols:
                    prior_val = df_yoy[col].iloc[0] if not df_yoy.empty and col in df_yoy.columns else np.nan
                    result[f"{col}_yoy_prior"] = prior_val
                    if pd.notna(prior_val) and prior_val != 0:
                        result[f"{col}_yoy"] = (row.get(col, 0) - prior_val) / prior_val
                    else:
                        result[f"{col}_yoy"] = np.nan

            results.append(result)

        return pd.DataFrame(results) if results else pd.DataFrame()

    # ----- Anomaly Detection -----

    def _detect_anomalies(self, detail_df: pd.DataFrame, config: dict) -> list[Anomaly]:
        """
        Apply deterministic anomaly rules to YoY growth changes.

        Focuses on YoY as the primary signal for structural performance changes.
        Rules are defined in config['anomaly_rules']:
            - yoy_threshold: flag if absolute YoY change exceeds this
            - min_denominator: ignore dimensions with volume below this
        """
        if detail_df.empty:
            return []

        rules = config.get("anomaly_rules", {})
        yoy_thresh = rules.get("yoy_threshold", 0.10)
        min_denom = rules.get("min_denominator", 50)

        anomalies = []

        for metric_key, metric_def in config["metrics"].items():
            label = metric_def["label"]
            direction = metric_def["direction"]
            denom_col = metric_def["denominator"]

            for _, row in detail_df.iterrows():
                dim_value = row.get("dimension", "overall")

                if denom_col in row and pd.notna(row[denom_col]) and row[denom_col] < min_denom:
                    continue

                yoy_col = f"{metric_key}_yoy"
                if yoy_col in row and pd.notna(row[yoy_col]):
                    pct = row[yoy_col]
                    if abs(pct) > yoy_thresh:
                        is_bad = (direction == "up" and pct < 0) or (direction == "down" and pct > 0)
                        severity = "high" if is_bad and abs(pct) > yoy_thresh * 2 else "medium"
                        if is_bad:
                            anomalies.append(Anomaly(
                                metric=metric_key,
                                metric_label=label,
                                dimension=dim_value,
                                comparison="yoy",
                                current_value=row.get(metric_key, np.nan),
                                prior_value=row.get(f"{metric_key}_yoy_prior", np.nan),
                                pct_change=pct,
                                threshold=yoy_thresh,
                                severity=severity,
                                direction=direction,
                            ))

        anomalies.sort(key=lambda a: (0 if a.severity == "high" else 1, -abs(a.pct_change)))
        return anomalies
