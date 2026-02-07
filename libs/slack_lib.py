"""
Slack Integration
==================
Simple Slack webhook posting for report distribution.

Usage:
    from libs.slack_lib import post_to_slack

    post_to_slack(webhook_url, "Report text here")
"""

import requests


def post_to_slack(
    webhook_url: str,
    text: str,
    header: str | None = None,
) -> bool:
    """
    Post a message to a Slack channel via incoming webhook.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL.
    text : str
        The message body (supports Slack markdown: *bold*, • bullets).
    header : str, optional
        If provided, prepended as a bold header line.

    Returns
    -------
    bool
        True if the message was posted successfully.
    """
    if header:
        message = f"*{header}*\n\n{text}"
    else:
        message = text

    payload = {"text": message}

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"  Slack: message posted successfully")
            return True
        else:
            print(f"  Slack: failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  Slack: error posting message: {e}")
        return False
