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

## Dev Setup (Cursor + Databricks Sync)

This is the recommended workflow for day-to-day development. You edit code in
Cursor (with AI assistance), files auto-sync to Databricks, and you can run the
notebook and see output — all without leaving the IDE.

### Prerequisites

- [Cursor IDE](https://cursor.sh/) (or VS Code)
- Git access to this repo
- A Databricks account on the Thumbtack workspace

### One-time setup (5 minutes)

**1. Install the Databricks CLI**

```bash
# macOS (Apple Silicon or Intel)
curl -fsSL https://raw.githubusercontent.com/databricks/setup-cli/main/install.sh | sudo sh

# Verify
databricks --version
```

**2. Clone this repo and authenticate**

```bash
git clone git@github.com:chicheng12/analytics-automation.git
cd analytics-automation
make auth
```

This opens your browser — log in with your `@thumbtack.com` account.
When prompted for a profile name, press Enter to accept the default.

**3. Verify your email is set in Git** (used to determine your Databricks workspace path)

```bash
git config user.email
# Should show: yourname@thumbtack.com
```

That's it. You're ready to go.

### Daily workflow

Open **two terminal tabs** in Cursor:

| Tab | Command | What it does |
|-----|---------|-------------|
| 1 | `make sync` | Watches for file changes, auto-pushes to Databricks on every save |
| 2 | `make run CLUSTER_ID=xxxx` | Submits a notebook run and streams output to your terminal |

**Tab 1** — start the file watcher (leave running):

```bash
make sync
```

Every time you save a file in Cursor, it auto-syncs to
`/Users/<you>@thumbtack.com/analytics_automation/` in Databricks.

**Tab 2** — run the notebook and see output:

```bash
# Find your cluster ID: Databricks UI → Compute → your cluster → see URL or details
make run CLUSTER_ID=xxxx-xxxxxx-xxxxxxxx

# With options:
make run CLUSTER_ID=xxxx DATE=2026-02-05 RUN_AI=False
```

Output (including errors and tracebacks) streams directly into your Cursor terminal.
Fix the error in the editor, save (auto-syncs), re-run — no copy-pasting.

**Tip:** reference terminal output in Cursor AI chat using `@terminal` to have it
read errors and suggest fixes automatically.

### All available commands

```bash
make help       # Show all commands
make auth       # Authenticate with Databricks (one-time)
make sync       # Live-sync files on save (keep running in a tab)
make push       # One-time sync (no watch)
make pull       # Pull Databricks workspace files back to local
make run        # Submit notebook run, stream output to terminal
make run-local  # Run locally without Databricks
make status     # Check auth status
```

### Alternative: run directly in Databricks UI

If you prefer the Databricks notebook UI:

1. Run `make sync` (or `make push`) to push your code
2. Open `orchestrator.py` in Databricks at `/Users/<you>@thumbtack.com/analytics_automation/`
3. First run: uncomment the `%pip install` cell and execute it
4. Configure widgets:
   - `date`: leave empty for yesterday, or set `YYYY-MM-DD`
   - `run_ai_agent`: True/False
   - `slack_webhook_url`: your Slack webhook (empty = skip)
   - `monitors`: comma-separated monitor keys (default: `revenue_funnel`)
5. Run All

## Quick Start (Local, no Databricks)

```bash
# Prerequisites
pip install -r requirements.txt
gcloud auth application-default login
export OPENAI_API_KEY=your_key_here

# Run
make run-local                          # defaults: yesterday, revenue_funnel, AI on
make run-local DATE=2026-02-05          # specific date
make run-local RUN_AI=False             # data only, skip AI
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
