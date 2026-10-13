"""Heart-failure 30-day readmission cohort for the ML tutorial.

One BigQuery Standard SQL pull against MIMIC-IV on physionet-data (dataset
ids come from sql/tables.py). The result is one row per index admission.
Rows are patient-level, so the notebook must not print them. Aggregates
only.

Index stay
    Any admission with an ICD-10 I50* or ICD-9 428* diagnosis, age 18 or
    older, alive at discharge, not sent to hospice. Patients who die within
    30 days without coming back are dropped (competing risk). Everyone else
    stays in the denominator.

Label
    readmit_30d = 1 when the patient's next admission, of any kind, starts
    0 to 30 days after this discharge. The next admission is LEAD() over
    every admission for that patient, not just heart-failure stays.

Features are knowable at discharge. Race is not selected. labevents is
touched only through an itemid list.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

# Load sql/tables.py by path. A bare `import tables` hits PyTables if that
# package is installed, and the dataset ids live in this repo's module.
_spec = importlib.util.spec_from_file_location(
    "mimic_bq_tables", Path(__file__).with_name("tables.py")
)
_tables = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tables)
t = _tables.t

# Same heart-failure definition as the SQL assignment.
HF = r"REGEXP_CONTAINS(UPPER(icd_code), r'^(I50|428)')"

# d_labitems itemids. One concept each, so first/last are not mixed across
# blood-gas and chemistry assays. Confirm labels against d_labitems if a
# site's extract differs.
LAB_ITEMS = {
    50912: "creat",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    51222: "hemoglobin",
    50862: "albumin",
    50931: "glucose",
    50963: "bnp",
}

# Flags the model sees. Charlson uses these plus a few extra categories
# that are summed inside the query and not exported on their own.
COMORBIDITIES = {
    "ckd": r"^(N18|585)",
    "dm": r"^(E1[0-4]|250)",
    "copd": r"^(J44|491|492|496)",
    "afib": r"^(I48|42731)",
    "cad": r"^(I25|414)",
    "htn": r"^(I1[0-6]|40[1-5])",
    "anemia": r"^(D5[0-3]|D6[34]|280|281|285)",
    "depression": r"^(F32|F33|2962|2963|311)",
    "liver": r"^(K70|K7[234]|571)",
    "obesity": r"^(E66|2780)",
}

# Extra Charlson pieces. Weights follow Charlson 1987. Metastatic cancer
# replaces the solid-tumor weight instead of adding to it. Heart failure
# itself is +1 for every row, because the cohort is defined by it.
CHARLSON_EXTRA = {
    "dementia": (r"^(F0[0-3]|G30|290)", 1),
    "pvd": (r"^(I7[013]|44[013])", 1),
    "cvd": (r"^(I6[0-9]|43[0-8])", 1),
    "pud": (r"^(K2[5-8]|53[1-4])", 1),
    "rheum": (r"^(M0[56]|M32|714|7100)", 1),
    "plegia": (r"^(G8[12]|342|344)", 2),
    "hiv": (r"^(B2[0-4]|042)", 6),
    "cancer": (r"^(C[0-6][0-9]|C7[0-6]|1[4-9][0-9])", 2),
    "metastatic": (r"^(C7[7-9]|C80|19[6-9])", 6),
}

DRUGS = {
    "gdmt_diuretic": r"(furosemide|torsemide|bumetanide|metolazone)",
    "gdmt_bb": r"(metoprolol|carvedilol|bisoprolol|labetalol)",
    "gdmt_acei": r"(lisinopril|enalapril|captopril|ramipril|losartan|valsartan|sacubitril)",
    "gdmt_mra": r"(spironolactone|eplerenone)",
    "gdmt_sglt2": r"(empagliflozin|dapagliflozin)",
}

TARGET = "readmit_30d"
GROUP_COL = "subject_id"
CAT_FEATURES = [
    "gender",
    "insurance",
    "admission_type",
    "admission_location",
    "discharge_location",
]
NUM_FEATURES = [
    "age",
    "los_days",
    "icu_flag",
    "icu_los",
    "weekend_discharge",
    "n_prior_adm_365",
    "n_prior_ed_365",
    "days_since_prev_disch",
    "n_diagnoses",
    "charlson",
    *COMORBIDITIES,
    *[f"{name}_first" for name in LAB_ITEMS.values()],
    *[f"{name}_last" for name in LAB_ITEMS.values()],
    *DRUGS,
]
# Used by the LACE baseline only, so the model does not see both ED windows.
LACE_ED_COL = "n_prior_ed_180"
OUTPUT_COLUMNS = [GROUP_COL, "hadm_id", *CAT_FEATURES, *NUM_FEATURES, LACE_ED_COL, TARGET]


def _flag(name: str, pattern: str) -> str:
    return (
        f"MAX(IF(REGEXP_CONTAINS(UPPER(icd_code), r'{pattern}'), 1, 0)) AS {name}"
    )


def _charlson_expr() -> str:
    """Simplified Charlson. See the module docstring for the shortcuts.

    Every input is IFNULL'd because the comorbidity join can miss a stay
    that has a heart-failure code only in a row the regex still matched
    (it should not) or, more realistically, because the LEFT JOIN is empty
    when diagnoses failed to attach. Heart failure still contributes 1.
    """
    def bit(name: str) -> str:
        return f"IFNULL(comorb.{name}, 0)"

    parts = ["1"]  # heart failure, by cohort definition
    parts.append(f"2 * {bit('ckd')}")
    parts.append(f"1 * {bit('dm')}")
    parts.append(f"1 * {bit('copd')}")
    parts.append(f"1 * {bit('liver')}")
    for name, (_pattern, weight) in CHARLSON_EXTRA.items():
        if name == "cancer":
            continue
        parts.append(f"{weight} * {bit(name)}")
    # Solid tumor counts only when there is no metastatic code.
    parts.append(f"IF({bit('metastatic')} = 1, 0, 2 * {bit('cancer')})")
    return " + ".join(parts)


def cohort_sql() -> str:
    """Modeling table. One row per eligible heart-failure discharge."""
    comorb_flags = ",\n        ".join(
        _flag(name, pattern) for name, pattern in COMORBIDITIES.items()
    )
    extra_flags = ",\n        ".join(
        _flag(name, pattern) for name, (pattern, _weight) in CHARLSON_EXTRA.items()
    )
    lab_cols = ",\n        ".join(
        f"MAX(IF(itemid = {itemid}, first_val, NULL)) AS {name}_first,\n"
        f"        MAX(IF(itemid = {itemid}, last_val, NULL)) AS {name}_last"
        for itemid, name in LAB_ITEMS.items()
    )
    drug_cols = ",\n        ".join(
        f"MAX(IF(REGEXP_CONTAINS(LOWER(drug), r'{pattern}'), 1, 0)) AS {name}"
        for name, pattern in DRUGS.items()
    )
    item_list = ", ".join(str(i) for i in LAB_ITEMS)
    return f"""
WITH hf_hadm AS (
    SELECT DISTINCT hadm_id
    FROM {t("diagnoses_icd")}
    WHERE hadm_id IS NOT NULL
      AND {HF}
),
adm AS (
    -- Windows run over every admission for the patient. Filtering to
    -- heart-failure stays before LEAD() would miss a non-HF readmission.
    SELECT
        a.subject_id,
        a.hadm_id,
        a.admittime,
        a.dischtime,
        a.edregtime,
        a.hospital_expire_flag,
        COALESCE(UPPER(TRIM(a.admission_type)), 'NOT RECORDED') AS admission_type,
        COALESCE(UPPER(TRIM(a.admission_location)), 'NOT RECORDED') AS admission_location,
        COALESCE(UPPER(TRIM(a.discharge_location)), 'NOT RECORDED') AS discharge_location,
        COALESCE(UPPER(TRIM(a.insurance)), 'NOT RECORDED') AS insurance,
        p.gender,
        p.dod,
        CAST(p.anchor_age AS INT64)
            + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
        LEAD(a.admittime) OVER (
            PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
        ) AS next_admit,
        LAG(a.dischtime) OVER (
            PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
        ) AS prev_disch
    FROM {t("admissions")} AS a
    JOIN {t("patients")} AS p USING (subject_id)
),
indexed AS (
    SELECT
        adm.*,
        TIMESTAMP_DIFF(
            CAST(next_admit AS TIMESTAMP),
            CAST(dischtime AS TIMESTAMP),
            DAY
        ) AS days_to_next,
        TIMESTAMP_DIFF(
            CAST(admittime AS TIMESTAMP),
            CAST(prev_disch AS TIMESTAMP),
            DAY
        ) AS days_since_prev_disch,
        ROUND(
            TIMESTAMP_DIFF(
                CAST(dischtime AS TIMESTAMP),
                CAST(admittime AS TIMESTAMP),
                HOUR
            ) / 24.0,
            2
        ) AS los_days,
        IF(EXTRACT(DAYOFWEEK FROM dischtime) IN (1, 7), 1, 0) AS weekend_discharge
    FROM adm
    JOIN hf_hadm USING (hadm_id)
),
index_stays AS (
    SELECT
        subject_id,
        hadm_id,
        admittime,
        dischtime,
        gender,
        age,
        insurance,
        admission_type,
        admission_location,
        discharge_location,
        los_days,
        weekend_discharge,
        GREATEST(days_since_prev_disch, 0) AS days_since_prev_disch,
        IF(days_to_next BETWEEN 0 AND 30, 1, 0) AS readmit_30d
    FROM indexed
    WHERE dischtime IS NOT NULL
      AND los_days >= 0
      AND age BETWEEN 18 AND 120
      AND hospital_expire_flag = 0
      AND NOT REGEXP_CONTAINS(discharge_location, r'HOSPICE|DIED')
      AND NOT (
          dod IS NOT NULL
          AND DATE_DIFF(DATE(dod), DATE(dischtime), DAY) BETWEEN 0 AND 30
          AND (days_to_next IS NULL OR days_to_next > 30)
      )
),
priors AS (
    -- Left side is the index stay only. The right side is any earlier
    -- admission for that patient, heart failure or not.
    SELECT
        idx.hadm_id,
        COUNTIF(
            CAST(prev.admittime AS TIMESTAMP) < CAST(idx.admittime AS TIMESTAMP)
            AND CAST(prev.admittime AS TIMESTAMP) >= TIMESTAMP_SUB(CAST(idx.admittime AS TIMESTAMP), INTERVAL 365 DAY)
        ) AS n_prior_adm_365,
        COUNTIF(
            prev.edregtime IS NOT NULL
            AND CAST(prev.admittime AS TIMESTAMP) < CAST(idx.admittime AS TIMESTAMP)
            AND CAST(prev.admittime AS TIMESTAMP) >= TIMESTAMP_SUB(CAST(idx.admittime AS TIMESTAMP), INTERVAL 365 DAY)
        ) AS n_prior_ed_365,
        COUNTIF(
            prev.edregtime IS NOT NULL
            AND CAST(prev.admittime AS TIMESTAMP) < CAST(idx.admittime AS TIMESTAMP)
            AND CAST(prev.admittime AS TIMESTAMP) >= TIMESTAMP_SUB(CAST(idx.admittime AS TIMESTAMP), INTERVAL 180 DAY)
        ) AS n_prior_ed_180
    FROM index_stays AS idx
    LEFT JOIN {t("admissions")} AS prev
        ON idx.subject_id = prev.subject_id
       AND prev.hadm_id != idx.hadm_id
    GROUP BY idx.hadm_id
),
comorb AS (
    SELECT
        d.hadm_id,
        COUNT(DISTINCT d.icd_code) AS n_diagnoses,
        {comorb_flags},
        {extra_flags}
    FROM {t("diagnoses_icd")} AS d
    JOIN index_stays USING (hadm_id)
    GROUP BY d.hadm_id
),
icu AS (
    SELECT
        hadm_id,
        1 AS icu_flag,
        ROUND(SUM(los), 2) AS icu_los
    FROM {t("icustays", icu=True)}
    JOIN index_stays USING (hadm_id)
    GROUP BY hadm_id
),
lab_long AS (
    -- itemid list is the entire labevents scan. Do not drop it.
    SELECT
        l.hadm_id,
        l.itemid,
        ARRAY_AGG(l.valuenum IGNORE NULLS ORDER BY l.charttime LIMIT 1)[OFFSET(0)] AS first_val,
        ARRAY_AGG(l.valuenum IGNORE NULLS ORDER BY l.charttime DESC LIMIT 1)[OFFSET(0)] AS last_val
    FROM {t("labevents")} AS l
    JOIN index_stays AS s
        ON l.hadm_id = s.hadm_id
       AND l.charttime BETWEEN s.admittime AND s.dischtime
    WHERE l.itemid IN ({item_list})
      AND l.valuenum IS NOT NULL
    GROUP BY l.hadm_id, l.itemid
),
labs AS (
    SELECT
        hadm_id,
        {lab_cols}
    FROM lab_long
    GROUP BY hadm_id
),
meds AS (
    -- Inpatient orders only. Home medications that were never rewritten
    -- as an order look like "not given" here.
    SELECT
        p.hadm_id,
        {drug_cols}
    FROM {t("prescriptions")} AS p
    JOIN index_stays USING (hadm_id)
    GROUP BY p.hadm_id
)
SELECT
    s.subject_id,
    s.hadm_id,
    s.gender,
    s.insurance,
    s.admission_type,
    s.admission_location,
    s.discharge_location,
    s.age,
    s.los_days,
    IFNULL(icu.icu_flag, 0) AS icu_flag,
    IFNULL(icu.icu_los, 0) AS icu_los,
    s.weekend_discharge,
    IFNULL(priors.n_prior_adm_365, 0) AS n_prior_adm_365,
    IFNULL(priors.n_prior_ed_365, 0) AS n_prior_ed_365,
    s.days_since_prev_disch,
    IFNULL(comorb.n_diagnoses, 0) AS n_diagnoses,
    ({_charlson_expr()}) AS charlson,
    {", ".join(f"IFNULL(comorb.{name}, 0) AS {name}" for name in COMORBIDITIES)},
    {", ".join(f"labs.{name}_first" for name in LAB_ITEMS.values())},
    {", ".join(f"labs.{name}_last" for name in LAB_ITEMS.values())},
    {", ".join(f"IFNULL(meds.{name}, 0) AS {name}" for name in DRUGS)},
    IFNULL(priors.n_prior_ed_180, 0) AS n_prior_ed_180,
    s.readmit_30d
FROM index_stays AS s
LEFT JOIN priors USING (hadm_id)
LEFT JOIN comorb USING (hadm_id)
LEFT JOIN icu USING (hadm_id)
LEFT JOIN labs USING (hadm_id)
LEFT JOIN meds USING (hadm_id)
""".strip()


def attrition_sql() -> str:
    """Counts at each exclusion. No patient identifiers."""
    return f"""
WITH hf_hadm AS (
    SELECT DISTINCT hadm_id
    FROM {t("diagnoses_icd")}
    WHERE hadm_id IS NOT NULL
      AND {HF}
),
adm AS (
    SELECT
        a.subject_id,
        a.hadm_id,
        a.dischtime,
        a.hospital_expire_flag,
        COALESCE(UPPER(TRIM(a.discharge_location)), 'NOT RECORDED') AS discharge_location,
        p.dod,
        CAST(p.anchor_age AS INT64)
            + (EXTRACT(YEAR FROM a.admittime) - p.anchor_year) AS age,
        TIMESTAMP_DIFF(
            CAST(LEAD(a.admittime) OVER (
                PARTITION BY a.subject_id ORDER BY a.admittime, a.hadm_id
            ) AS TIMESTAMP),
            CAST(a.dischtime AS TIMESTAMP),
            DAY
        ) AS days_to_next,
        TIMESTAMP_DIFF(
            CAST(a.dischtime AS TIMESTAMP),
            CAST(a.admittime AS TIMESTAMP),
            HOUR
        ) / 24.0 AS los_days
    FROM {t("admissions")} AS a
    JOIN {t("patients")} AS p USING (subject_id)
),
base AS (
    SELECT adm.*
    FROM adm
    JOIN hf_hadm USING (hadm_id)
    WHERE dischtime IS NOT NULL
      AND los_days >= 0
),
steps AS (
    SELECT 'Heart-failure discharges' AS step, 1 AS ord, COUNT(*) AS n
    FROM base
    UNION ALL
    SELECT 'Age 18 or older', 2, COUNT(*) FROM base WHERE age BETWEEN 18 AND 120
    UNION ALL
    SELECT 'Alive at discharge', 3, COUNT(*)
    FROM base
    WHERE age BETWEEN 18 AND 120 AND hospital_expire_flag = 0
    UNION ALL
    SELECT 'Not discharged to hospice', 4, COUNT(*)
    FROM base
    WHERE age BETWEEN 18 AND 120
      AND hospital_expire_flag = 0
      AND NOT REGEXP_CONTAINS(discharge_location, r'HOSPICE|DIED')
    UNION ALL
    SELECT 'Final cohort', 5, COUNT(*)
    FROM base
    WHERE age BETWEEN 18 AND 120
      AND hospital_expire_flag = 0
      AND NOT REGEXP_CONTAINS(discharge_location, r'HOSPICE|DIED')
      AND NOT (
          dod IS NOT NULL
          AND DATE_DIFF(DATE(dod), DATE(dischtime), DAY) BETWEEN 0 AND 30
          AND (days_to_next IS NULL OR days_to_next > 30)
      )
)
SELECT step, n FROM steps ORDER BY ord
""".strip()
