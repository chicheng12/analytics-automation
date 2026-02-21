"""
Slack Integration
==================
Slack posting for report distribution. Supports two methods:
1. Bot token (chat.postMessage API) — preferred, richer formatting
2. Incoming webhook — simpler, but requires webhook URL per channel

Usage:
    from libs.slack_lib import post_to_slack

    # Via bot token (preferred):
    post_to_slack(text="Report here", channel="#analytics-automation-test",
                  bot_token="xoxb-...")

    # Via webhook (legacy):
    post_to_slack(text="Report here", webhook_url="https://hooks.slack.com/...")
"""

import requests


def post_to_slack(
    text: str,
    webhook_url: str | None = None,
    bot_token: str | None = None,
    channel: str | None = None,
    header: str | None = None,
) -> bool:
    """
    Post a message to Slack via bot token or webhook.

    Parameters
    ----------
    text : str
        The message body (supports Slack mrkdwn).
    webhook_url : str, optional
        Slack incoming webhook URL. Used if bot_token is not provided.
    bot_token : str, optional
        Slack bot token (xoxb-...). Preferred over webhook.
    channel : str, optional
        Channel name or ID. Required when using bot_token.
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

    if bot_token and channel:
        return _post_via_bot(bot_token, channel, message)
    elif webhook_url:
        return _post_via_webhook(webhook_url, message)
    else:
        print("  Slack: no bot_token+channel or webhook_url provided, skipping")
        return False


def _post_via_bot(token: str, channel: str, text: str) -> bool:
    """Post via Slack chat.postMessage API."""
    url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "channel": channel,
        "text": text,
        "unfurl_links": False,
        "unfurl_media": False,
    }

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        data = resp.json()
        if data.get("ok"):
            print(f"  Slack: posted to {channel} successfully")
            return True
        else:
            print(f"  Slack: API error — {data.get('error', 'unknown')}")
            return False
    except Exception as e:
        print(f"  Slack: error posting message: {e}")
        return False


def _post_via_webhook(webhook_url: str, text: str) -> bool:
    """Post via Slack incoming webhook."""
    payload = {"text": text}
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"  Slack: message posted successfully via webhook")
            return True
        else:
            print(f"  Slack: webhook failed with status {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  Slack: error posting message: {e}")
        return False
