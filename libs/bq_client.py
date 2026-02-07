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

    On Databricks: uses cluster-attached service account credentials.
    Locally: uses Application Default Credentials (gcloud auth application-default login).
    """
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
