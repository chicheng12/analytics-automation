"""
BigQuery Client Wrapper
========================
Provides a consistent BigQuery connection interface that works in both
Databricks (cluster-managed credentials) and local (ADC) environments.

Usage:
    from libs.bq_client import get_bq_client, bq_to_df

    client = get_bq_client()
    df = bq_to_df("SELECT 1 as test")
"""

import pandas as pd
from google.cloud import bigquery

DEFAULT_PROJECT = "tt-dp-prod"


def get_bq_client(project: str = DEFAULT_PROJECT) -> bigquery.Client:
    """
    Create a BigQuery client.

    On Databricks (both Prod and LLM workspaces): uses google.auth.default()
    with drive + cloud-platform scopes (per Thumbtack LLM playbook).
    Locally: uses Application Default Credentials (gcloud auth application-default login).
    """
    try:
        import google.auth
        credentials, auth_project = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/drive",
                "https://www.googleapis.com/auth/cloud-platform",
            ]
        )
        return bigquery.Client(credentials=credentials, project=project)
    except Exception:
        # Fallback: let bigquery.Client find credentials itself (works locally with ADC)
        return bigquery.Client(project=project)


def bq_to_df(
    query: str,
    project: str = DEFAULT_PROJECT,
    client: bigquery.Client | None = None,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Execute a BigQuery query and return results as a pandas DataFrame.

    Parameters
    ----------
    query : str
        The SQL query to execute.
    project : str
        GCP project ID (default: tt-dp-prod).
    client : bigquery.Client, optional
        Reuse an existing client. If None, creates a new one.
    progress_bar : bool
        Show tqdm progress bar during download (default: True).

    Returns
    -------
    pd.DataFrame
    """
    if client is None:
        client = get_bq_client(project)

    job = client.query(query)
    bar_type = "tqdm" if progress_bar else None
    return job.to_dataframe(
        progress_bar_type=bar_type,
        create_bqstorage_client=True,
    )
