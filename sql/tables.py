"""MIMIC-IV BigQuery dataset ids. No GCP client import — safe for notebook rebuilds."""
from __future__ import annotations

import os
import shutil
import subprocess

# Dataset ids as they appear under the starred physionet-data project.
# Override if your console names differ (e.g. mimiciv_v3_1_icu).
HOSP = os.environ.get("MIMIC_BQ_HOSP", "physionet-data.mimiciv_3_1_hosp")
ICU = os.environ.get("MIMIC_BQ_ICU", "physionet-data.mimiciv_3_1_icu")

PROJECT_HELP = """
No GCP project ID set. BigQuery bills YOUR project, not physionet-data.

1. Create a project if you do not have one:
   https://console.cloud.google.com/projectcreate
2. Copy the Project ID (e.g. mimic-bq-123456), then run:

   gcloud config set project YOUR_PROJECT_ID
   gcloud auth application-default set-quota-project YOUR_PROJECT_ID
   export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID

3. Re-run: .venv/bin/python sql/proto.py
""".strip()


def t(table: str, *, icu: bool = False) -> str:
    return f"`{(ICU if icu else HOSP)}.{table}`"


def _gcloud_project() -> str | None:
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return None
    try:
        out = subprocess.check_output(
            [gcloud, "config", "get-value", "project"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return None
    if not out or out == "(unset)":
        return None
    return out


def billing_project() -> str | None:
    for key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "BQ_PROJECT"):
        val = os.environ.get(key)
        if val:
            return val
    return _gcloud_project()
