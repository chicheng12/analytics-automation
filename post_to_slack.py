"""
Post-run Slack poster.
Fetches the notebook output from a Databricks run and posts it to Slack.

Usage:
    python post_to_slack.py <run_output.json> <bot_token> <channel> [--profile PROFILE]
"""

import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def get_notebook_output(run_id: str, profile: str) -> str | None:
    """Fetch notebook output using databricks CLI."""
    cmd = ["databricks", "jobs", "get-run-output", str(run_id), "--profile", profile]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error fetching run output: {result.stderr.strip()}")
        return None

    data = json.loads(result.stdout)
    return data.get("notebook_output", {}).get("result")


def main():
    if len(sys.argv) < 4:
        print("Usage: python post_to_slack.py <run_output.json> <bot_token> <channel>")
        sys.exit(1)

    run_output_path = sys.argv[1]
    bot_token = sys.argv[2]
    channel = sys.argv[3]
    profile = sys.argv[4] if len(sys.argv) > 4 else None

    with open(run_output_path) as f:
        data = json.load(f)

    # Find the task run_id and profile
    task_run_id = None
    for task in data.get("tasks", []):
        task_run_id = task.get("run_id")
        if task_run_id:
            break

    if not task_run_id:
        print("  No task run_id found in output")
        sys.exit(1)

    if not profile:
        # Try to detect from the run_page_url
        url = data.get("run_page_url", "")
        if "2939161070393309" in url:
            profile = "2939161070393309"
        elif "1977877856098707" in url:
            profile = "1977877856098707"
        else:
            print("  Cannot detect profile, pass as 4th argument")
            sys.exit(1)

    print(f"  Fetching notebook output for run {task_run_id}...")
    notebook_output = get_notebook_output(task_run_id, profile)

    if not notebook_output:
        print("  No notebook output found (AI agent may have been skipped)")
        sys.exit(0)

    report_data = json.loads(notebook_output)
    report = report_data.get("report", "")
    report_date = report_data.get("report_date", "unknown")

    if not report:
        print("  Report is empty, nothing to post")
        sys.exit(0)

    print(f"  Report date: {report_date}")
    print(f"  Report length: {len(report)} chars")

    from libs.slack_lib import post_to_slack
    post_to_slack(
        text=report,
        bot_token=bot_token,
        channel=channel,
        header=f"Daily Analytics Report — {report_date}",
    )


if __name__ == "__main__":
    main()
