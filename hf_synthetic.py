"""Fake heart-failure cohort with the same columns as the BigQuery pull.

Not MIMIC. Not for submission. It exists so the notebook, figures, and
slides can be built without querying physionet-data. The outcome is a
noisy function of the features, with a patient-level effect, so a
patient-grouped split and a random split do not match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sql.hf_ml_cohort import (
    CAT_FEATURES,
    COMORBIDITIES,
    DRUGS,
    GROUP_COL,
    LAB_ITEMS,
    LACE_ED_COL,
    NUM_FEATURES,
    OUTPUT_COLUMNS,
    TARGET,
)

ADMISSION_TYPES = ["EW EMER.", "URGENT", "ELECTIVE", "OBSERVATION ADMIT", "DIRECT EMER."]
ADMISSION_LOCATIONS = ["EMERGENCY ROOM", "PHYSICIAN REFERRAL", "TRANSFER FROM HOSPITAL", "CLINIC REFERRAL"]
DISCHARGE_LOCATIONS = ["HOME", "HOME HEALTH CARE", "SKILLED NURSING FACILITY", "REHAB", "CHRONIC/LONG TERM ACUTE CARE"]
INSURANCE = ["MEDICARE", "MEDICAID", "OTHER", "PRIVATE"]


def make_cohort(n_patients: int = 4000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # About one extra stay for every three patients, so leakage is visible.
    n_stays = rng.integers(1, 4, size=n_patients)
    subject_ids = np.repeat(np.arange(1, n_patients + 1), n_stays)
    n = len(subject_ids)
    stay_ix = np.concatenate([np.arange(k) for k in n_stays])

    age = np.clip(rng.normal(72, 12, size=n_patients), 22, 100)
    frailty = rng.normal(0, 1, size=n_patients)
    age_row = np.repeat(age, n_stays)
    frailty_row = np.repeat(frailty, n_stays)

    gender = rng.choice(["F", "M"], size=n_patients, p=[0.48, 0.52])
    insurance = rng.choice(INSURANCE, size=n_patients, p=[0.62, 0.12, 0.14, 0.12])
    ckd_p = rng.random(n_patients) < (0.22 + 0.25 * (age > 75))
    dm_p = rng.random(n_patients) < 0.42

    rows_gender = np.repeat(gender, n_stays)
    rows_insurance = np.repeat(insurance, n_stays)
    ckd = np.repeat(ckd_p.astype(int), n_stays)
    dm = np.repeat(dm_p.astype(int), n_stays)

    admission_type = rng.choice(ADMISSION_TYPES, size=n, p=[0.55, 0.15, 0.12, 0.10, 0.08])
    admission_location = rng.choice(ADMISSION_LOCATIONS, size=n, p=[0.6, 0.2, 0.12, 0.08])
    emergency = np.isin(admission_type, ["EW EMER.", "URGENT", "DIRECT EMER."]).astype(float)

    los_days = np.round(np.clip(rng.lognormal(mean=1.3, sigma=0.55, size=n), 0.4, 40), 2)
    icu_flag = (rng.random(n) < (0.28 + 0.1 * emergency)).astype(int)
    icu_los = np.round(np.where(icu_flag == 1, np.clip(rng.normal(3.5, 2.0, size=n), 0.2, 20), 0), 2)
    weekend = rng.integers(0, 2, size=n)

    n_prior = np.clip(stay_ix + rng.poisson(0.4, size=n), 0, 12).astype(int)
    n_ed_365 = np.clip(rng.binomial(n_prior + 1, 0.55), 0, None).astype(int)
    n_ed_180 = np.minimum(n_ed_365, rng.binomial(n_ed_365 + 1, 0.6)).astype(int)
    days_since = np.where(stay_ix == 0, np.nan, rng.integers(15, 400, size=n).astype(float))

    other_flags = {
        name: (rng.random(n) < rate).astype(int)
        for name, rate in {
            "copd": 0.24,
            "afib": 0.38,
            "cad": 0.45,
            "htn": 0.72,
            "anemia": 0.28,
            "depression": 0.16,
            "liver": 0.08,
            "obesity": 0.18,
        }.items()
    }
    n_diagnoses = rng.integers(4, 22, size=n)
    charlson = (
        1
        + 2 * ckd
        + dm
        + other_flags["copd"]
        + other_flags["liver"]
        + other_flags["cad"]
        + rng.integers(0, 3, size=n)
    )

    def _lab(mu, sigma, missing, bump=0.0):
        val = rng.normal(mu + bump, sigma, size=n)
        val[rng.random(n) < missing] = np.nan
        return np.round(val, 2)

    creat_last = _lab(1.3, 0.7, 0.04, bump=0.6 * ckd + 0.15 * frailty_row)
    creat_first = np.where(np.isnan(creat_last), np.nan, np.round(creat_last - rng.normal(0.05, 0.25, size=n), 2))
    bun_last = _lab(28, 14, 0.05, bump=8 * ckd)
    bun_first = np.where(np.isnan(bun_last), np.nan, np.round(bun_last - rng.normal(1, 4, size=n), 2))
    sodium_last = _lab(137, 4, 0.03)
    sodium_first = np.where(np.isnan(sodium_last), np.nan, np.round(sodium_last + rng.normal(0, 1.5, size=n), 2))
    potassium_last = _lab(4.1, 0.5, 0.04)
    potassium_first = np.where(np.isnan(potassium_last), np.nan, np.round(potassium_last + rng.normal(0, 0.2, size=n), 2))
    hemoglobin_last = _lab(11.2, 1.8, 0.06, bump=-0.4 * other_flags["anemia"])
    hemoglobin_first = np.where(np.isnan(hemoglobin_last), np.nan, np.round(hemoglobin_last + rng.normal(0.2, 0.4, size=n), 2))
    albumin_last = _lab(3.4, 0.6, 0.22)
    albumin_first = np.where(np.isnan(albumin_last), np.nan, np.round(albumin_last + rng.normal(0, 0.15, size=n), 2))
    glucose_last = _lab(140, 45, 0.05, bump=25 * dm)
    glucose_first = np.where(np.isnan(glucose_last), np.nan, np.round(glucose_last + rng.normal(0, 15, size=n), 2))
    bnp_last = _lab(1800, 900, 0.35, bump=400 * frailty_row)
    bnp_first = np.where(np.isnan(bnp_last), np.nan, np.round(bnp_last * rng.uniform(0.8, 1.3, size=n), 2))

    drugs = {name: (rng.random(n) < rate).astype(int) for name, rate in {
        "gdmt_diuretic": 0.72,
        "gdmt_bb": 0.58,
        "gdmt_acei": 0.41,
        "gdmt_mra": 0.18,
        "gdmt_sglt2": 0.09,
    }.items()}

    # Discharge location depends on frailty so it carries signal.
    loc_score = frailty_row + 0.3 * ckd + 0.02 * (los_days - 4)
    discharge = np.full(n, "HOME", dtype=object)
    discharge[loc_score > 0.2] = "HOME HEALTH CARE"
    discharge[loc_score > 0.8] = "SKILLED NURSING FACILITY"
    discharge[loc_score > 1.5] = "REHAB"
    discharge[loc_score > 2.2] = "CHRONIC/LONG TERM ACUTE CARE"

    logit = (
        -1.55
        + 0.55 * frailty_row
        + 0.018 * (age_row - 70)
        + 0.35 * ckd
        + 0.22 * (n_prior >= 2)
        + 0.15 * emergency
        + 0.25 * np.isin(discharge, ["SKILLED NURSING FACILITY", "CHRONIC/LONG TERM ACUTE CARE"])
        + 0.12 * (los_days > 7)
        + 0.2 * (np.nan_to_num(creat_last, nan=1.3) > 1.8)
        - 0.15 * drugs["gdmt_bb"]
        + rng.normal(0, 0.35, size=n)
    )
    prob = 1 / (1 + np.exp(-logit))
    y = rng.random(n) < prob

    data = {
        GROUP_COL: subject_ids,
        "hadm_id": np.arange(1, n + 1),
        "gender": rows_gender,
        "insurance": rows_insurance,
        "admission_type": admission_type,
        "admission_location": admission_location,
        "discharge_location": discharge,
        "age": np.round(age_row, 1),
        "los_days": los_days,
        "icu_flag": icu_flag,
        "icu_los": icu_los,
        "weekend_discharge": weekend,
        "n_prior_adm_365": n_prior,
        "n_prior_ed_365": n_ed_365,
        "days_since_prev_disch": days_since,
        "n_diagnoses": n_diagnoses,
        "charlson": charlson.astype(int),
        "ckd": ckd,
        "dm": dm,
        **other_flags,
        "creat_first": creat_first,
        "creat_last": creat_last,
        "bun_first": bun_first,
        "bun_last": bun_last,
        "sodium_first": sodium_first,
        "sodium_last": sodium_last,
        "potassium_first": potassium_first,
        "potassium_last": potassium_last,
        "hemoglobin_first": hemoglobin_first,
        "hemoglobin_last": hemoglobin_last,
        "albumin_first": albumin_first,
        "albumin_last": albumin_last,
        "glucose_first": glucose_first,
        "glucose_last": glucose_last,
        "bnp_first": bnp_first,
        "bnp_last": bnp_last,
        **drugs,
        LACE_ED_COL: n_ed_180,
        TARGET: y.astype(int),
    }
    df = pd.DataFrame(data)
    # Extra stays are near-copies of that patient's first stay. A random
    # split then tests the model on a row it has essentially already fit,
    # which is the leak a patient split is there to stop.
    first_ix = df.groupby(GROUP_COL, sort=False).head(1).set_index(GROUP_COL)
    extra = df.groupby(GROUP_COL).cumcount() > 0
    if extra.any():
        copied = first_ix.loc[df.loc[extra, GROUP_COL]].copy()
        jitter_cols = [c for c in copied.columns if c.endswith("_first") or c.endswith("_last") or c == "los_days"]
        for col in jitter_cols:
            noise = rng.normal(0, 0.02, size=len(copied))
            copied[col] = copied[col] * (1 + noise)
        flip = rng.random(len(copied)) < 0.05
        copied.loc[flip, TARGET] = 1 - copied.loc[flip, TARGET].to_numpy()
        # Prior-use fields are allowed to change. They are known at discharge
        # and they are not a copy of the earlier label.
        for col in ("hadm_id", "n_prior_adm_365", "n_prior_ed_365", LACE_ED_COL, "days_since_prev_disch"):
            copied[col] = df.loc[extra, col].to_numpy()
        df.loc[extra, copied.columns] = copied.to_numpy()

    missing = [c for c in OUTPUT_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in OUTPUT_COLUMNS]
    if missing or extra:
        raise RuntimeError(f"schema drift missing={missing} extra={extra}")
    # Silence unused-import linters for the re-exported names used by callers.
    _ = (CAT_FEATURES, NUM_FEATURES, LAB_ITEMS, COMORBIDITIES, DRUGS)
    return df[OUTPUT_COLUMNS]


def synthetic_attrition(n_final: int) -> pd.DataFrame:
    """Funnel shaped like the real exclusions. Counts are invented."""
    scales = [1.34, 1.27, 1.11, 1.05, 1.0]
    labels = [
        "Heart-failure discharges",
        "Age 18 or older",
        "Alive at discharge",
        "Not discharged to hospice",
        "Final cohort",
    ]
    return pd.DataFrame({
        "step": labels,
        "n": [int(round(n_final * s)) for s in scales],
    })


if __name__ == "__main__":
    cohort = make_cohort()
    print("rows", len(cohort), "patients", cohort[GROUP_COL].nunique())
    print("readmit_n", int(cohort[TARGET].sum()))
    print("columns", len(cohort.columns))
