"""
Orchestrator
=============
Main entry point that runs all configured monitors, feeds results to the
AI analyst agent, and distributes the report.

Designed to run as a Databricks notebook (sync this folder to workspace).

Usage on Databricks:
    1. Sync analytics_automation/ folder to /Workspace/Users/<you>/analytics_automation/
    2. Open this file as a notebook (or run via %run)
    3. Configure widgets at the top
    4. Run All

Usage locally (for testing):
    python orchestrator.py --date 2026-02-05
"""

# %% [markdown]
# ## 1. Setup

# %%
# -- Databricks widget setup (ignored when running locally) --
try:
    dbutils.widgets.text("date", "", "Report Date (YYYY-MM-DD, empty=yesterday)")
    dbutils.widgets.text("slack_webhook_url", "", "Slack Webhook URL (empty=skip)")
    dbutils.widgets.dropdown("run_ai_agent", "True", ["True", "False"], "Run AI Agent?")
    dbutils.widgets.text("monitors", "revenue_funnel", "Monitors to run (comma-separated)")

    REPORT_DATE = dbutils.widgets.get("date") or None
    SLACK_WEBHOOK_URL = dbutils.widgets.get("slack_webhook_url") or None
    RUN_AI_AGENT = dbutils.widgets.get("run_ai_agent") == "True"
    MONITOR_KEYS = [m.strip() for m in dbutils.widgets.get("monitors").split(",")]
    IS_DATABRICKS = True
except NameError:
    # Running locally
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=None)
    parser.add_argument("--slack-webhook-url", default=None)
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--monitors", default="revenue_funnel")
    args = parser.parse_args()

    REPORT_DATE = args.date
    SLACK_WEBHOOK_URL = args.slack_webhook_url
    RUN_AI_AGENT = not args.no_ai
    MONITOR_KEYS = [m.strip() for m in args.monitors.split(",")]
    IS_DATABRICKS = False

# %%
# -- Package installs (Databricks only, uncomment on first run) --
# %pip install google-cloud-bigquery google-cloud-bigquery-storage openai-agents db-dtypes
# dbutils.library.restartPython()

# %%
# -- Imports --
import sys
import os
from datetime import datetime, timedelta

# Ensure libs/ is importable (handles both Databricks workspace and local)
notebook_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
if notebook_dir not in sys.path:
    sys.path.insert(0, notebook_dir)

from libs.bq_client import get_bq_client
from libs.monitor_engine import MonitorEngine
from libs.monitor_registry import MONITOR_REGISTRY

print(f"Analytics Automation Orchestrator")
print(f"  Date: {REPORT_DATE or 'yesterday (auto)'}")
print(f"  Monitors: {MONITOR_KEYS}")
print(f"  AI Agent: {RUN_AI_AGENT}")
print(f"  Slack: {'configured' if SLACK_WEBHOOK_URL else 'skipped'}")
print(f"  Environment: {'Databricks' if IS_DATABRICKS else 'Local'}")

# %% [markdown]
# ## 2. Run Monitors

# %%
# -- OpenAI setup (do this early so it fails fast if key is missing) --
if RUN_AI_AGENT:
    from libs.agent_lib import setup_openai_env
    if IS_DATABRICKS:
        setup_openai_env(dbutils)
    else:
        setup_openai_env()  # expects OPENAI_API_KEY env var

# %%
# -- Initialize engine and run monitors --
client = get_bq_client()
engine = MonitorEngine(client)

monitor_results = []
for key in MONITOR_KEYS:
    if key not in MONITOR_REGISTRY:
        print(f"\n  WARNING: Monitor '{key}' not found in registry. Skipping.")
        continue
    result = engine.run(key, date=REPORT_DATE)
    monitor_results.append(result)

print(f"\n{'='*60}")
print(f"Completed {len(monitor_results)} monitor(s)")

# %% [markdown]
# ## 3. Display Results (Data Layer)

# %%
# -- Show summary metrics --
for result in monitor_results:
    print(f"\n--- {result.name} ({result.report_date}) ---")

    if not result.summary_df.empty:
        print("\nOverall Summary:")
        display(result.summary_df) if IS_DATABRICKS else print(result.summary_df.to_string())

    if not result.detail_df.empty:
        channel_df = result.detail_df[result.detail_df["dim_type"] == "channel"]
        if not channel_df.empty:
            print("\nBy Channel:")
            display(channel_df) if IS_DATABRICKS else print(channel_df.to_string())

    if result.anomalies:
        print(f"\nAnomalies ({len(result.anomalies)}):")
        for a in result.anomalies:
            print(f"  [{a.severity.upper()}] {a.metric_label} ({a.dimension}): "
                  f"{a.pct_change:+.1%} {a.comparison}")
    else:
        print("\nNo anomalies flagged.")

# %% [markdown]
# ## 4. AI Analysis (Tier 2)

# %%
analysis_output = None
agent_run_result = None

if RUN_AI_AGENT and monitor_results:
    import asyncio
    from libs.agent_lib import run_analysis, compute_cost

    # Run the analyst agent
    analysis_output, agent_run_result = asyncio.get_event_loop().run_until_complete(
        run_analysis(monitor_results)
    )

    # Display the narrative
    print("\n" + "=" * 60)
    print("AI ANALYSIS")
    print("=" * 60)

    if IS_DATABRICKS:
        from IPython.display import Markdown, display as ipy_display
        ipy_display(Markdown(analysis_output.markdown_report))
    else:
        print(analysis_output.markdown_report)

    # Display structured findings
    print(f"\nHeadline: {analysis_output.headline}")
    print(f"Findings: {len(analysis_output.findings)}")
    for i, f in enumerate(analysis_output.findings, 1):
        print(f"\n  Finding {i} [{f.severity}]: {f.summary}")
        print(f"    Action: {f.action}")
        for h in f.hypotheses:
            print(f"    Hypothesis [{h.confidence}]: {h.claim}")

    # Cost tracking
    if agent_run_result:
        cost = compute_cost(agent_run_result.context_wrapper.usage)
        print(f"\n  Token cost: ${cost['total_cost']:.4f} "
              f"({cost['input_tokens']} in, {cost['output_tokens']} out)")

elif not RUN_AI_AGENT:
    print("\n  AI agent skipped (run_ai_agent=False)")
else:
    print("\n  AI agent skipped (no monitor results)")

# %% [markdown]
# ## 5. Distribute Report

# %%
# -- Post to Slack --
if SLACK_WEBHOOK_URL and analysis_output:
    from libs.slack_lib import post_to_slack

    report_date_str = monitor_results[0].report_date if monitor_results else "unknown"
    post_to_slack(
        webhook_url=SLACK_WEBHOOK_URL,
        text=analysis_output.markdown_report,
        header=f"Daily Analytics Report — {report_date_str}",
    )
elif SLACK_WEBHOOK_URL and not analysis_output:
    # Post a basic summary without AI if agent was skipped
    from libs.slack_lib import post_to_slack

    lines = ["*Daily Analytics Report (data only, no AI narrative)*\n"]
    for result in monitor_results:
        lines.append(f"*{result.name}*")
        lines.append(f"Anomalies: {len(result.anomalies)}")
        for a in result.anomalies[:5]:
            lines.append(f"• [{a.severity}] {a.metric_label} ({a.dimension}): {a.pct_change:+.1%} {a.comparison}")
    post_to_slack(SLACK_WEBHOOK_URL, "\n".join(lines))
else:
    print("\n  Slack posting skipped (no webhook URL configured)")

# %% [markdown]
# ## 6. Archive to BigQuery (optional)

# %%
# -- Save report to BigQuery for historical tracking --
if analysis_output and monitor_results:
    import pandas as pd
    import json

    archive_data = {
        "report_date": [monitor_results[0].report_date],
        "created_timestamp": [pd.Timestamp.now()],
        "headline": [analysis_output.headline],
        "markdown_report": [analysis_output.markdown_report],
        "findings_json": [json.dumps([f.model_dump() for f in analysis_output.findings], default=str)],
        "anomaly_count": [sum(len(r.anomalies) for r in monitor_results)],
        "monitors_run": [",".join(MONITOR_KEYS)],
    }

    if agent_run_result:
        cost = compute_cost(agent_run_result.context_wrapper.usage)
        archive_data["token_cost_usd"] = [cost["total_cost"]]
    else:
        archive_data["token_cost_usd"] = [0.0]

    df_archive = pd.DataFrame(archive_data)

    # Uncomment to enable BQ archival (requires write access):
    # from google.cloud import bigquery as bq
    # table_id = "tt-dp-prod.reference.analytics_automation_reports"
    # job_config = bq.LoadJobConfig(
    #     schema=[
    #         bq.SchemaField("report_date", "DATE"),
    #         bq.SchemaField("created_timestamp", "TIMESTAMP"),
    #         bq.SchemaField("headline", "STRING"),
    #         bq.SchemaField("markdown_report", "STRING"),
    #         bq.SchemaField("findings_json", "STRING"),
    #         bq.SchemaField("anomaly_count", "INT64"),
    #         bq.SchemaField("monitors_run", "STRING"),
    #         bq.SchemaField("token_cost_usd", "FLOAT64"),
    #     ]
    # )
    # job = client.load_table_from_dataframe(df_archive, table_id, job_config=job_config)
    # job.result()
    # print(f"  Archived to {table_id}")

    print("\n  Archive data prepared (BQ write commented out — uncomment when ready)")
    if IS_DATABRICKS:
        display(df_archive)
    else:
        print(df_archive.to_string())

print(f"\n{'='*60}")
print("Done!")
