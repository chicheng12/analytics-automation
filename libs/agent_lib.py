"""
AI Agent Definitions
=====================
Tier 2 analyst agent that receives structured MonitorResults and produces
analytical narratives with hypothesis generation and cross-domain reasoning.

Uses the OpenAI Agents SDK with structured Pydantic output.

Usage:
    from libs.agent_lib import run_analysis, setup_openai_env

    # On Databricks: call setup_openai_env(dbutils) first
    setup_openai_env(dbutils)

    result = await run_analysis([monitor_result_1, monitor_result_2])
"""

from __future__ import annotations

import os
import json
from typing import TYPE_CHECKING

import pandas as pd
import numpy as np
from pydantic import BaseModel

if TYPE_CHECKING:
    from libs.monitor_engine import MonitorResult

MODEL_NAME = "o3"


# ===================================================================
# Structured output schemas
# ===================================================================

class Hypothesis(BaseModel):
    """A hypothesis explaining a finding, with evidence and next steps."""
    claim: str
    """One-sentence causal claim, e.g., 'RPP decline driven by category mix shift.'"""

    evidence: list[str]
    """List of data points supporting this claim."""

    confidence: str
    """'high', 'medium', or 'low'."""

    next_step: str
    """What to investigate next to confirm or reject this hypothesis."""


class Finding(BaseModel):
    """A single analytical finding ranked by business severity."""
    severity: str
    """'high', 'medium', or 'low'."""

    domains: list[str]
    """Which monitor domain(s) this finding relates to."""

    summary: str
    """One-line summary of the finding."""

    details: str
    """Full explanation with specific numbers from the data."""

    hypotheses: list[Hypothesis]
    """Possible explanations for this finding."""

    action: str
    """Recommended action: 'investigate', 'monitor', or 'no action'."""


class AnalysisResult(BaseModel):
    """Complete analytical output from the agent."""
    headline: str
    """Single most important takeaway for the day."""

    findings: list[Finding]
    """Findings ranked by severity (high first)."""

    markdown_report: str
    """Human-readable Slack-formatted report (*bold* headers, bullet points)."""


# ===================================================================
# OpenAI environment setup
# ===================================================================

def setup_openai_env(dbutils=None):
    """
    Set the OPENAI_API_KEY environment variable.

    On Databricks: reads from the 'openai' secrets scope.
    Locally: expects OPENAI_API_KEY already set in environment.

    Parameters
    ----------
    dbutils : object, optional
        Databricks dbutils object. Pass this when running on Databricks.
    """
    if dbutils is not None:
        api_key = dbutils.secrets.get(scope="openai", key="api_key")
        os.environ["OPENAI_API_KEY"] = api_key
        print("  OpenAI API key loaded from Databricks secrets")
    elif "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Either pass dbutils (Databricks) "
            "or set the environment variable locally."
        )


# ===================================================================
# Agent definition
# ===================================================================

ANALYST_INSTRUCTIONS = """
You are a senior business analyst at a two-sided home services marketplace
(connecting homeowners who need work done with service professionals).

You receive structured monitoring outputs from one or more business domains.
Each monitor provides:
- A summary table with current metric values, prior period values, and percent changes
- A detail table with the same metrics broken down by dimension (e.g., channel)
- A list of pre-flagged anomalies (from deterministic rules) with severity levels
- Context describing what this domain measures

## YOUR ANALYTICAL TASKS

### 1. CROSS-DOMAIN CONNECTIONS
- If multiple monitors are provided, look for causal links between findings.
  For example: if RPP dropped and PPR also dropped, that compounds revenue impact.
- If only one monitor is provided, connect findings across dimensions
  (e.g., "SEM channel shows RPP decline while SEO is stable").

### 2. HEADLINE
- Lead with the single most impactful finding in one sentence.
- Include the specific metric value and percent change.

### 3. FINDINGS (ranked by business impact)
For each finding:
- State what happened with specific numbers
- Propose WHY (hypothesis) with evidence
- Recommend what to do next
- Distinguish between structural trends and episodic one-off changes

### 4. SEVERITY RANKING
Rank by business impact, not just statistical magnitude:
- A 5% RPP drop across all channels is more impactful than a 30% swing in a tiny channel
- Consider both the rate change AND the volume of the dimension

### 5. HYPOTHESIS GENERATION
For each major finding, propose:
- A primary hypothesis with supporting evidence from the data
- What data would confirm or reject it
- Whether it looks structural (persistent trend) or episodic (one-off)

## RULES
- Every claim MUST cite specific numbers from the provided data
- NEVER invent metrics not present in the inputs
- Clearly distinguish between facts (from data) and hypotheses (your inference)
- If the anomalies list is empty for a domain, say "no notable anomalies flagged"
  rather than inventing a narrative
- Use SAFE_DIVIDE-style thinking: if a denominator is very small, note that the
  rate metric may be unreliable
- Use Slack formatting: *bold* for headers (not **bold**), bullet points with •

## MARKDOWN REPORT FORMAT (for Slack)
*Daily Revenue Funnel Analysis — {date}*

*Headline:*
• [Single most important finding]

*Key Findings:*
• [Finding 1 with numbers]
• [Finding 2 with numbers]
...

*Hypotheses & Next Steps:*
• [Hypothesis 1]: [Evidence] → [Next step]
...

*Anomaly Summary:*
• [Count] anomalies flagged: [brief list]
• No action needed / Investigate [X]
"""


def _format_monitor_for_prompt(result: MonitorResult) -> str:
    """
    Serialize a MonitorResult into a text block for the agent prompt.

    Converts DataFrames to readable tables and anomalies to structured text.
    """
    sections = []
    sections.append(f"## Monitor: {result.name}")
    sections.append(f"Report date: {result.report_date}")
    sections.append(f"Context: {result.context}")

    # Summary metrics
    if not result.summary_df.empty:
        sections.append("\n### Overall Summary")
        # Convert to dict for cleaner display
        summary_records = result.summary_df.to_dict(orient="records")
        for record in summary_records:
            # Round floats for readability
            clean = {
                k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                for k, v in record.items()
                if pd.notna(v)
            }
            sections.append(json.dumps(clean, indent=2, default=str))

    # Detail metrics (by dimension)
    if not result.detail_df.empty:
        detail_by_dim = result.detail_df[result.detail_df["dim_type"] != "overall"]
        if not detail_by_dim.empty:
            sections.append("\n### Breakdown by Dimension")
            detail_records = detail_by_dim.to_dict(orient="records")
            for record in detail_records:
                clean = {
                    k: (round(v, 4) if isinstance(v, (float, np.floating)) else v)
                    for k, v in record.items()
                    if pd.notna(v)
                }
                sections.append(json.dumps(clean, indent=2, default=str))

    # Anomalies
    sections.append(f"\n### Pre-Flagged Anomalies ({len(result.anomalies)} total)")
    if result.anomalies:
        for a in result.anomalies:
            sections.append(
                f"  - [{a.severity.upper()}] {a.metric_label} in {a.dimension}: "
                f"{a.pct_change:+.1%} {a.comparison} "
                f"(current: {a.current_value:.4f}, prior: {a.prior_value:.4f}, "
                f"threshold: {a.threshold:.0%})"
            )
    else:
        sections.append("  No anomalies flagged.")

    return "\n".join(sections)


async def run_analysis(
    monitor_results: list[MonitorResult],
    model: str = MODEL_NAME,
) -> AnalysisResult:
    """
    Run the analyst agent on one or more MonitorResults.

    Parameters
    ----------
    monitor_results : list[MonitorResult]
        Output from MonitorEngine.run() for one or more monitors.
    model : str
        OpenAI model to use (default: gpt-4o).

    Returns
    -------
    AnalysisResult
        Structured analysis with findings, hypotheses, and markdown report.
    """
    from agents import Agent, Runner

    # Build the agent
    analyst_agent = Agent(
        name="Daily Performance Analyst",
        instructions=ANALYST_INSTRUCTIONS,
        model=model,
        output_type=AnalysisResult,
    )

    # Format all monitor results into the prompt
    prompt_parts = []
    for result in monitor_results:
        prompt_parts.append(_format_monitor_for_prompt(result))

    full_prompt = "\n\n---\n\n".join(prompt_parts)

    # Run the agent
    print(f"\n  Running analyst agent ({model})...")
    agent_result = await Runner.run(analyst_agent, full_prompt)
    print(f"  Agent completed.")

    return agent_result.final_output, agent_result


def compute_cost(usage, model: str = MODEL_NAME) -> dict:
    """
    Compute the dollar cost of an agent run from its token usage.

    Parameters
    ----------
    usage : object
        The usage object from agent_result.context_wrapper.usage
    model : str
        Model name for pricing lookup.

    Returns
    -------
    dict with token counts and costs.
    """
    # Model pricing (update when switching models)
    # Source: https://openai.com/api/pricing/
    PRICING = {
        "o3": {"input": 10.00 / 1_000_000, "cached": 2.50 / 1_000_000, "output": 40.00 / 1_000_000},
        "o4-mini": {"input": 1.10 / 1_000_000, "cached": 0.275 / 1_000_000, "output": 4.40 / 1_000_000},
        "gpt-4o": {"input": 2.50 / 1_000_000, "cached": 1.25 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini": {"input": 0.15 / 1_000_000, "cached": 0.075 / 1_000_000, "output": 0.60 / 1_000_000},
        "gpt-4.1": {"input": 2.00 / 1_000_000, "cached": 0.50 / 1_000_000, "output": 8.00 / 1_000_000},
    }
    prices = PRICING.get(model, PRICING["o3"])

    input_tokens = usage.input_tokens
    cached_tokens = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
    output_tokens = usage.output_tokens

    cost_input = (input_tokens - cached_tokens) * prices["input"]
    cost_cached = cached_tokens * prices["cached"]
    cost_output = output_tokens * prices["output"]

    return {
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "cost_input": round(cost_input, 6),
        "cost_cached": round(cost_cached, 6),
        "cost_output": round(cost_output, 6),
        "total_cost": round(cost_input + cost_cached + cost_output, 6),
    }
