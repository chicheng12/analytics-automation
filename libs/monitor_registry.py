"""
Monitor Registry
=================
Configuration-driven definitions for all monitors. Adding a new analysis
requires only a new entry here and (optionally) a new SQL template file.

Each monitor config defines:
    - name: human-readable label
    - query_file: path to the parameterized SQL template (relative to queries/)
    - dimensions: which breakdowns to compute (each gets its own data pull)
    - metrics: the ratio metrics to compute from the raw counts
    - anomaly_rules: deterministic thresholds for flagging anomalies
    - comparison: which comparisons to compute (wow = week-over-week, yoy = year-over-year)
    - context: brief description for the AI agent to understand the domain

Usage:
    from libs.monitor_registry import MONITOR_REGISTRY
    config = MONITOR_REGISTRY['revenue_funnel']
"""

import os

# Resolve the queries/ directory relative to this file
_QUERIES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queries")


def get_query_path(filename: str) -> str:
    """Get the absolute path to a SQL template file."""
    return os.path.join(_QUERIES_DIR, filename)


MONITOR_REGISTRY = {
    "revenue_funnel": {
        "name": "Revenue Funnel Daily Monitor",
        "query_file": "revenue_funnel_daily.sql",
        "dimensions": ["overall", "channel"],
        "metrics": {
            # Each metric is derived from the raw counts returned by the SQL.
            # 'numerator' and 'denominator' must match column names in the query output.
            # 'direction': 'up' means higher is better, 'down' means lower is better.
            "iv_to_request": {
                "label": "Requests per Intentful Visitor",
                "numerator": "requests",
                "denominator": "intentful_visitors",
                "direction": "up",
            },
            "ppr": {
                "label": "Projects per Request (PPR)",
                "numerator": "projects",
                "denominator": "requests",
                "direction": "up",
            },
            "rpp": {
                "label": "Revenue per Project (RPP)",
                "numerator": "revenue",
                "denominator": "projects",
                "direction": "up",
            },
            "cpp": {
                "label": "Contacts per Project (CPP)",
                "numerator": "contacts",
                "denominator": "projects",
                "direction": "down",
            },
            "rpc": {
                "label": "Revenue per Contact (RPC)",
                "numerator": "revenue",
                "denominator": "contacts",
                "direction": "up",
            },
        },
        "volume_metrics": [
            # Raw counts that are also interesting to track (not ratios).
            "intentful_visitors",
            "requests",
            "projects",
            "contacts",
            "revenue",
        ],
        "anomaly_rules": {
            "wow_threshold": 0.10,  # flag if WoW change > 10%
            "yoy_threshold": 0.15,  # flag if YoY change > 15%
            "min_denominator": 50,  # ignore dimensions with tiny volume
        },
        "comparison": ["wow", "yoy"],
        "context": (
            "Revenue funnel metrics tracking the health of the demand-to-revenue pipeline. "
            "The funnel flows: Intentful Visitors -> Requests -> Projects -> Revenue. "
            "Key ratio metrics: IV-to-Request conversion, PPR (project conversion rate), "
            "RPP (monetization per project), CPP (contacts per project), RPC (revenue per contact). "
            "Breakdowns available by channel (SEM, SEO, Direct App, Direct Web, Partnership, etc.)."
        ),
    },
    # -----------------------------------------------------------------------
    # Add new monitors here. Example:
    #
    # "hbo": {
    #     "name": "HBO Daily Monitor",
    #     "query_file": "hbo_daily.sql",
    #     "dimensions": ["overall", "category"],
    #     "metrics": { ... },
    #     ...
    # },
    # -----------------------------------------------------------------------
}
