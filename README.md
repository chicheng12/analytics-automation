# Analytics Automation

Registry-driven daily metric monitoring with AI-powered analysis, designed to run on Databricks.

## Folder Structure

```
analytics_automation/
  libs/
    bq_client.py          # BigQuery connection (Databricks + local)
    monitor_registry.py   # Config-driven monitor definitions
    monitor_engine.py     # Generic engine: pull → compute → detect anomalies
    agent_lib.py          # AI analyst agent (OpenAI Agents SDK)
    slack_lib.py          # Slack webhook posting
  queries/
    revenue_funnel_daily.sql   # SQL template for revenue funnel monitor
  orchestrator.py         # Main entry point (runs on Databricks or locally)
```

## Quick Start (Databricks)

1. Sync this folder to `/Workspace/Users/<you>/analytics_automation/`
2. Open `orchestrator.py` as a Databricks notebook
3. First run: uncomment the `%pip install` cell and run it
4. Configure widgets:
   - `date`: leave empty for yesterday, or set `YYYY-MM-DD`
   - `run_ai_agent`: True/False
   - `slack_webhook_url`: your Slack webhook (empty = skip)
   - `monitors`: comma-separated monitor keys (default: `revenue_funnel`)
5. Run All

## Quick Start (Local)

```bash
# Prerequisites
pip install google-cloud-bigquery google-cloud-bigquery-storage openai-agents db-dtypes
gcloud auth application-default login
export OPENAI_API_KEY=your_key_here

# Run
python orchestrator.py --date 2026-02-05
python orchestrator.py --no-ai          # data only, skip AI
```

## Adding a New Monitor

1. Add a SQL template to `queries/your_monitor.sql`
2. Add a config entry to `libs/monitor_registry.py` → `MONITOR_REGISTRY`
3. Add dimension mappings to `libs/monitor_engine.py` → `DIMENSION_SQL` (if new dimensions)
4. Run via orchestrator: set `monitors` widget to include your new key

No changes needed to the engine, agent, or orchestrator.

## Architecture

```
Monitor Registry (config)
    ↓
Monitor Engine (deterministic: pull data → compute deltas → flag anomalies)
    ↓
Analyst Agent (AI: synthesize → hypothesize → recommend)
    ↓
Outputs (Slack, BigQuery archive)
```

The AI layer receives pre-computed, structured data — it does analytical reasoning
and narrative synthesis, not raw data processing. Anomaly detection is deterministic
and testable.
