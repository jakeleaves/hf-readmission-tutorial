"""Assemble the heart-failure readmission tutorial notebook.

Running this file does not query BigQuery and does not read a cohort.
The notebook it writes does the querying, and only when HF_SOURCE=bigquery.
"""
from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from sql.hf_ml_cohort import attrition_sql, cohort_sql

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "build", "hf_ml_template.ipynb")


def md(text: str):
    return new_markdown_cell(text.strip("\n"))


def code(text: str):
        return new_code_cell(text.strip("\n"))


def sql_block(title: str, sql: str) -> str:
    return f"{title}\n\n~~~~sql\n{sql}\n~~~~"


SETUP = r'''
import os
from pathlib import Path

# Matplotlib's font cache must sit inside the workspace. Set this before
# importing pyplot. LOKY_MAX_CPU_COUNT avoids a noisy core-count warning.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "build" / "mpl"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")

from sql.hf_ml_cohort import (
    CAT_FEATURES,
    GROUP_COL,
    LACE_ED_COL,
    NUM_FEATURES,
    OUTPUT_COLUMNS,
    TARGET,
    attrition_sql,
    cohort_sql,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ROOT = Path.cwd()
SOURCE = os.environ.get("HF_SOURCE", "bigquery")
if SOURCE == "synthetic":
    DATA = ROOT / "data" / "hf_synthetic"
    ART = ROOT / "build" / "hf_synthetic"
elif SOURCE == "bigquery":
    # Real MIMIC output. This directory is gitignored and cursorignored.
    DATA = ROOT / "data" / "hf_private"
    ART = ROOT / "build" / "hf_private"
else:
    raise RuntimeError("HF_SOURCE must be 'bigquery' or 'synthetic'.")

DATA.mkdir(parents=True, exist_ok=True)
ART.mkdir(parents=True, exist_ok=True)
FEATURES = CAT_FEATURES + NUM_FEATURES
MIN_CELL = 10  # hide aggregates with fewer than this many stays

print("source", SOURCE)
print("sklearn_ok", True)
print("torch", torch.__version__)
'''

LOAD = r'''
def _gcloud_project():
    import shutil
    import subprocess
    gcloud = shutil.which("gcloud")
    if not gcloud:
        return None
    try:
        out = subprocess.check_output(
            [gcloud, "config", "get-value", "project"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None
    if not out or out == "(unset)":
        return None
    return out


def billing_project():
    for key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "BQ_PROJECT"):
        val = os.environ.get(key)
        if val:
            return val
    found = _gcloud_project()
    if found:
        return found
    raise RuntimeError(
        "Set GOOGLE_CLOUD_PROJECT to your own billing project. "
        "Queries are billed to you, not to the data host."
    )


def _counts(frame):
    # Aggregates only. Do not print frame.head() or any id column.
    return (
        f"rows {len(frame)} "
        f"patients {frame[GROUP_COL].nunique()} "
        f"readmit {int(frame[TARGET].sum())}"
    )


cache = DATA / "cohort.parquet"
attrition_cache = DATA / "attrition.parquet"
refresh = os.environ.get("HF_REFRESH") == "1"

if cache.exists() and attrition_cache.exists() and not refresh:
    cohort = pd.read_parquet(cache)
    attrition = pd.read_parquet(attrition_cache)
    print("loaded cache")
else:
    if SOURCE == "synthetic":
        from hf_synthetic import make_cohort, synthetic_attrition
        cohort = make_cohort(n_patients=4000, seed=SEED)
        attrition = synthetic_attrition(len(cohort))
    else:
        from google.cloud import bigquery
        client = bigquery.Client(project=billing_project())
        # create_bqstorage_client=False avoids a second download path.
        cohort = client.query(cohort_sql()).result().to_dataframe(create_bqstorage_client=False)
        attrition = client.query(attrition_sql()).result().to_dataframe(create_bqstorage_client=False)
    cohort.to_parquet(cache, index=False)
    attrition.to_parquet(attrition_cache, index=False)
    print("wrote cache")

missing_cols = [c for c in OUTPUT_COLUMNS if c not in cohort.columns]
if missing_cols:
    raise RuntimeError("cohort missing columns: " + ", ".join(missing_cols))

for col in NUM_FEATURES + [LACE_ED_COL, TARGET]:
    cohort[col] = pd.to_numeric(cohort[col], errors="coerce")
for col in CAT_FEATURES:
    cohort[col] = cohort[col].astype("string").fillna("NOT RECORDED")

cohort = cohort.dropna(subset=[TARGET, GROUP_COL]).copy()
cohort[TARGET] = cohort[TARGET].astype(int)
if cohort[TARGET].nunique() < 2:
    raise RuntimeError("The cohort has only one class. There is nothing to train.")

print(_counts(cohort))
attrition
'''

SPLIT = r'''
def patient_split(df, test_size, seed):
    """Split stays by patient, stratifying on whether the patient had any readmission.

    A random row split leaks: two stays from the same person land on both
    sides, and the model is graded on someone it has already seen.
    """
    rng = np.random.default_rng(seed)
    patient_label = df.groupby(GROUP_COL)[TARGET].max()
    ids = patient_label.index.to_numpy()
    labels = patient_label.to_numpy()
    held_out = []
    for cls in (0, 1):
        pool = ids[labels == cls].copy()
        rng.shuffle(pool)
        k = int(round(len(pool) * test_size))
        k = min(max(k, 1), len(pool) - 1)
        held_out.append(pool[:k])
    held_ids = set(np.concatenate(held_out).tolist())
    mask = df[GROUP_COL].isin(held_ids)
    return df.loc[~mask].copy(), df.loc[mask].copy()


# 15% of patients -> test, then 15% of the original cohort -> validation.
# 0.15 / 0.85 is that second cut, taken from what remains after the test holdout.
# Different seeds keep the two cuts from sharing the same shuffle order.
rest, test = patient_split(cohort, 0.15, SEED)
train, val = patient_split(rest, 0.15 / 0.85, SEED + 1)

overlap = set(train[GROUP_COL]).intersection(test[GROUP_COL], val[GROUP_COL])
if overlap or set(train[GROUP_COL]) & set(val[GROUP_COL]) or set(val[GROUP_COL]) & set(test[GROUP_COL]):
    raise RuntimeError("A patient is in more than one split.")

print(
    "train", len(train), "patients", train[GROUP_COL].nunique(),
    "| val", len(val), "patients", val[GROUP_COL].nunique(),
    "| test", len(test), "patients", test[GROUP_COL].nunique(),
)
'''

EDA = r'''
def savefig(fig, name):
    path = ART / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rate_table(frame, column):
    """Readmission rate by a category. Levels with n < MIN_CELL are dropped."""
    grouped = (
        frame.groupby(column, dropna=False)[TARGET]
        .agg(n="size", n_readmit="sum")
        .reset_index()
    )
    grouped = grouped[grouped["n"] >= MIN_CELL].copy()
    grouped["rate"] = grouped["n_readmit"] / grouped["n"]
    return grouped.sort_values("n", ascending=False)


fig, ax = plt.subplots(figsize=(8.2, 4.2))
steps = attrition["step"].tolist()
ns = attrition["n"].tolist()
ax.barh(steps[::-1], ns[::-1], color="#2F6F9F")
ax.set_xlabel("Stays")
ax.set_title("Who remains in the cohort")
fig.tight_layout()
savefig(fig, "attrition.png")

disp = rate_table(cohort, "discharge_location")
fig, ax = plt.subplots(figsize=(8.2, 4.4))
ax.barh(disp["discharge_location"][::-1], disp["rate"][::-1], color="#D98F2B")
ax.set_xlabel("30-day readmission rate")
ax.set_xlim(0, max(0.5, float(disp["rate"].max()) + 0.05))
ax.set_title("Readmission rate by discharge location")
fig.tight_layout()
savefig(fig, "disposition.png")

miss_pct = (cohort[NUM_FEATURES].isna().mean() * 100).sort_values(ascending=False)
miss_pct = miss_pct[miss_pct > 0]
fig, ax = plt.subplots(figsize=(8.2, 4.6))
if len(miss_pct):
    ax.barh(miss_pct.index[::-1], miss_pct.values[::-1], color="#2F6F9F")
ax.set_xlabel("Percent missing")
ax.set_title("Missing numeric features")
fig.tight_layout()
savefig(fig, "missingness.png")

print("final_n", int(attrition["n"].iloc[-1]))
print("features_with_missing", int((cohort[NUM_FEATURES].isna().any()).sum()))
disp[["discharge_location", "n", "rate"]]
'''

LEAKAGE = r'''
def fit_gbm(train_df, test_df):
    """Small, untuned booster used only to show the split effect."""
    model = HistGradientBoostingClassifier(
        max_iter=80,
        max_depth=3,
        learning_rate=0.08,
        min_samples_leaf=30,
        early_stopping=False,  # its internal split is by row, which reintroduces leakage
        random_state=SEED,
    )
    pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
    pipe.fit(train_df[FEATURES], train_df[TARGET])
    pred = pipe.predict_proba(test_df[FEATURES])[:, 1]
    return float(roc_auc_score(test_df[TARGET], pred))


def make_preprocessor():
    # Median impute, then a 0/1 column for "this value was missing".
    # days_since_prev_disch is missing on the first stay on purpose, not at random.
    # Scaling is for the logistic model and the neural net. The booster does not need it.
    # drop="first" gives each categorical an explicit reference level for odds ratios.
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUM_FEATURES),
        ("cat", categorical, CAT_FEATURES),
    ])


# Both experiments stay inside train+val. The test patients are not used here.
pool = pd.concat([train, val], ignore_index=True)
rng = np.random.default_rng(SEED)
row_order = rng.permutation(len(pool))
cut = int(round(len(pool) * 0.8))
random_train = pool.iloc[row_order[:cut]]
random_test = pool.iloc[row_order[cut:]]
group_train, group_test = patient_split(pool, 0.20, SEED + 7)

random_auroc = fit_gbm(random_train, random_test)
grouped_auroc = fit_gbm(group_train, group_test)
print("random_split_auroc", round(random_auroc, 3))
print("grouped_split_auroc", round(grouped_auroc, 3))

fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.bar(
    ["Random stay split", "Patient split"],
    [random_auroc, grouped_auroc],
    color=["#9AA4B0", "#2F6F9F"],
)
ax.set_ylim(0.5, max(0.85, random_auroc, grouped_auroc) + 0.05)
ax.set_ylabel("AUROC")
ax.set_title("Same model, two ways to hold out data")
fig.tight_layout()
savefig(fig, "leakage.png")
'''

MODELS = r'''
def lace_points(frame):
    """LACE points from van Walraven 2010, with a shortened Charlson.

    L  length of stay, in days
    A  3 if the admission was emergency or urgent, else 0
    C  Charlson band: 0, 1, 2, 3, or 5 when the score is 4 or more
    E  emergency visits in the prior 6 months, capped at 4

    The Charlson piece uses the simplified score from the SQL file, so this
    is a LACE-style baseline, not a claim that the published index was reproduced.
    """
    los = np.select(
        [
            frame["los_days"] < 1,
            frame["los_days"] < 2,
            frame["los_days"] < 3,
            frame["los_days"] < 4,
            frame["los_days"] < 7,
            frame["los_days"] < 14,
        ],
        [0, 1, 2, 3, 4, 5],
        default=7,
    ).astype(float)
    acute = frame["admission_type"].str.contains("EMER|URGENT", regex=True).fillna(False).astype(float) * 3
    charlson = frame["charlson"].fillna(0).clip(lower=0)
    comorbidity = np.where(charlson >= 4, 5, charlson).astype(float)
    emergency_visits = frame[LACE_ED_COL].fillna(0).clip(0, 4).astype(float)
    return los + acute + comorbidity + emergency_visits


def fit_predict_proba(pipe, train_df, test_df):
    pipe.fit(train_df[FEATURES], train_df[TARGET])
    return pipe.predict_proba(test_df[FEATURES])[:, 1]


# Hyperparameters are chosen on train only. The probability models that do
# not need a validation stream are then refit on train+val. The test set
# is scored once, below.
trainval = pd.concat([train, val], ignore_index=True)
for frame in (train, val, test, trainval):
    frame["lace"] = lace_points(frame)

# Unweighted logistic of the outcome on the LACE total. This is a calibrator:
# it turns points into a probability. No class_weight, or the probabilities
# would not match observed frequencies.
lace_model = LogisticRegression(max_iter=300, random_state=SEED)
lace_model.fit(trainval[["lace"]], trainval[TARGET])
p_lace = lace_model.predict_proba(test[["lace"]])[:, 1]

# C=1.0 is a mild L2 penalty. No class_weight: the outcome rate is about
# one in four, and weighting would push probabilities off the diagonal.
logit = Pipeline([
    ("prep", make_preprocessor()),
    ("model", LogisticRegression(C=1.0, max_iter=500, solver="lbfgs", random_state=SEED)),
])
p_logit = fit_predict_proba(logit, trainval, test)

# Grouped CV inside the training patients. early_stopping stays off because
# that feature carves a row-wise validation slice. The four grids stay small
# on purpose: this is a teaching search, not a full Optuna sweep.
grid = [
    {"learning_rate": 0.08, "max_depth": 3, "max_iter": 120, "min_samples_leaf": 40},
    {"learning_rate": 0.05, "max_depth": 3, "max_iter": 200, "min_samples_leaf": 40},
    {"learning_rate": 0.08, "max_depth": 6, "max_iter": 120, "min_samples_leaf": 30},
    {"learning_rate": 0.05, "max_depth": 4, "max_iter": 160, "min_samples_leaf": 50},
]
cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
best_params, best_cv = None, -1.0
for params in grid:
    fold_scores = []
    for tr_idx, va_idx in cv.split(train, train[TARGET], train[GROUP_COL]):
        model = HistGradientBoostingClassifier(
            early_stopping=False, random_state=SEED, **params,
        )
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        fold_train = train.iloc[tr_idx]
        fold_val = train.iloc[va_idx]
        pipe.fit(fold_train[FEATURES], fold_train[TARGET])
        pred = pipe.predict_proba(fold_val[FEATURES])[:, 1]
        if fold_val[TARGET].nunique() < 2:
            continue
        fold_scores.append(roc_auc_score(fold_val[TARGET], pred))
    mean_score = float(np.mean(fold_scores))
    print("cv", round(mean_score, 3), "max_depth", params["max_depth"], "lr", params["learning_rate"])
    if mean_score > best_cv:
        best_cv = mean_score
        best_params = params

gbm = Pipeline([
    ("prep", make_preprocessor()),
    ("model", HistGradientBoostingClassifier(
        early_stopping=False, random_state=SEED, **best_params,
    )),
])
p_gbm = fit_predict_proba(gbm, trainval, test)
print("chosen_cv_auroc", round(best_cv, 3))
'''

MLP = r'''
# The network sees the same imputed, scaled matrix as the logistic model.
# The preprocessor is fit on train only. Validation is used to stop training,
# so this model is not refit on train+val (that would remove the stop signal).
prep = make_preprocessor()
X_train = prep.fit_transform(train[FEATURES]).astype(np.float32)
X_val = prep.transform(val[FEATURES]).astype(np.float32)
X_test = prep.transform(test[FEATURES]).astype(np.float32)
y_train = train[TARGET].to_numpy().astype(np.float32)
y_val = val[TARGET].to_numpy().astype(np.float32)

torch.manual_seed(SEED)
net = torch.nn.Sequential(
    torch.nn.Linear(X_train.shape[1], 64),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(64, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)
# About 1 in 4 stays is a readmission, so we do not up-weight the positive
# class. pos_weight would chase recall and pull probabilities away from the
# diagonal on the calibration plot. Try it as an exercise.
loss_fn = torch.nn.BCEWithLogitsLoss()
# weight_decay is L2 on the weights. lr=1e-3 is Adam's usual default.
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

def val_loss():
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(X_val)).squeeze(1)
        return float(loss_fn(logits, torch.from_numpy(y_val)))

# Patience of 4 epochs without a 1e-4 drop in validation loss. Cap at 20.
best_state, best_epoch, best_loss, wait = None, 0, float("inf"), 0
batch = 512
rng = np.random.default_rng(SEED)
for epoch in range(1, 21):
    net.train()
    order = rng.permutation(len(X_train))
    for start in range(0, len(X_train), batch):
        batch_ix = order[start:start + batch]
        optimizer.zero_grad()
        logits = net(torch.from_numpy(X_train[batch_ix])).squeeze(1)
        loss = loss_fn(logits, torch.from_numpy(y_train[batch_ix]))
        loss.backward()
        optimizer.step()
    current = val_loss()
    if current < best_loss - 1e-4:
        best_loss = current
        best_epoch = epoch
        best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= 4:
            break

# Restore the best validation weights before scoring the test set once.
net.load_state_dict(best_state)
net.eval()
with torch.no_grad():
    p_mlp = torch.sigmoid(net(torch.from_numpy(X_test)).squeeze(1)).numpy()
print("mlp_best_epoch", best_epoch, "val_loss", round(best_loss, 4))
'''

EVAL = r'''
def bootstrap_auroc(y, pred, n_boot=300):
    """Percentile interval for the AUROC. The point estimate is the full test set."""
    rng = np.random.default_rng(SEED)
    y = np.asarray(y)
    pred = np.asarray(pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if np.unique(y[idx]).size < 2:
            continue
        scores.append(roc_auc_score(y[idx], pred[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def pack(y, pred):
    lo, hi = bootstrap_auroc(y, pred)
    return {
        "auroc": float(roc_auc_score(y, pred)),
        "auroc_lo": lo,
        "auroc_hi": hi,
        "auprc": float(average_precision_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
    }


y_test = test[TARGET].to_numpy()
predictions = {
    "LACE": p_lace,
    "Logistic": p_logit,
    "GBM": p_gbm,
    "MLP": p_mlp,
}
model_metrics = {name: pack(y_test, pred) for name, pred in predictions.items()}
for name, row in model_metrics.items():
    print(
        name,
        "auroc", round(row["auroc"], 3),
        "auprc", round(row["auprc"], 3),
        "brier", round(row["brier"], 3),
    )

colors = {"LACE": "#9AA4B0", "Logistic": "#2F6F9F", "GBM": "#D98F2B", "MLP": "#2E7D5B"}
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for name, pred in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, pred)
    ax.plot(fpr, tpr, color=colors[name], label=f"{name}  {model_metrics[name]['auroc']:.2f}")
ax.plot([0, 1], [0, 1], color="#D0D5DC", lw=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("Test-set ROC")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "roc.png")

fig, ax = plt.subplots(figsize=(6.2, 5.2))
for name, pred in predictions.items():
    prec, rec, _ = precision_recall_curve(y_test, pred)
    ax.plot(rec, prec, color=colors[name], label=name)
ax.axhline(y_test.mean(), color="#D0D5DC", lw=1)
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Test-set precision-recall")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "pr.png")

# Quantile bins keep roughly equal mass in each point. Equal-width bins would
# crowd the low-probability end when most predictions sit near the prevalence.
fig, ax = plt.subplots(figsize=(6.2, 5.2))
for name, pred in predictions.items():
    fraction, mean_pred = calibration_curve(y_test, pred, n_bins=8, strategy="quantile")
    ax.plot(mean_pred, fraction, marker="o", color=colors[name], label=name)
ax.plot([0, 1], [0, 1], color="#D0D5DC", lw=1)
ax.set_xlabel("Predicted probability")
ax.set_ylabel("Observed readmission rate")
ax.set_title("Calibration")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "calibration.png")
'''

DECIDE = r'''
def net_benefit(y, pred, threshold):
    """Decision curve. Benefit of a true positive minus the harm of a false
    positive, with the exchange rate threshold/(1-threshold).
    """
    called = pred >= threshold
    tp = np.sum(called & (y == 1))
    fp = np.sum(called & (y == 0))
    n = len(y)
    odds = threshold / (1 - threshold)
    return tp / n - fp / n * odds


# Flag the highest-risk fifth under the gradient booster. Ties can push the
# flag rate a little above 20%.
cutoff = np.quantile(p_gbm, 0.80)
flagged = p_gbm >= cutoff
op_precision = float(precision_score(y_test, flagged, zero_division=0))
op_recall = float(recall_score(y_test, flagged, zero_division=0))
print(
    "flag_rate", round(float(flagged.mean()), 3),
    "precision", round(op_precision, 3),
    "recall", round(op_recall, 3),
)

thresholds = np.linspace(0.05, 0.50, 19)
fig, ax = plt.subplots(figsize=(7.2, 4.6))
for name in ("LACE", "GBM"):
    benefits = [net_benefit(y_test, predictions[name], t) for t in thresholds]
    ax.plot(thresholds, benefits, color=colors[name], label=name)
prevalence = float(y_test.mean())
treat_all = [prevalence - (1 - prevalence) * (t / (1 - t)) for t in thresholds]
ax.plot(thresholds, treat_all, color="#1D2433", lw=1, label="Flag everyone")
ax.axhline(0, color="#D0D5DC", lw=1)
ax.set_xlabel("Threshold probability")
ax.set_ylabel("Net benefit")
ax.set_title("Decision curve")
ax.legend(frameon=False)
fig.tight_layout()
savefig(fig, "dca.png")
'''

EXPLAIN = r'''
# Odds ratios are per 1 SD for numeric inputs, and versus the dropped
# reference level for categories. They are not "per year of age".
names = logit.named_steps["prep"].get_feature_names_out()
coef = logit.named_steps["model"].coef_.ravel()
order = np.argsort(np.abs(coef))[::-1][:8]
odds_rows = [
    {"feature": names[i].split("__", 1)[-1], "odds_ratio": float(np.exp(coef[i]))}
    for i in order
]
print("top_odds_ratios", len(odds_rows))

# TreeExplainer reads the booster's splits. A sample is enough for a bar
# chart of mean absolute contribution and keeps the run short.
prep_gbm = gbm.named_steps["prep"]
booster = gbm.named_steps["model"]
transformed = prep_gbm.transform(test[FEATURES])
gbm_names = np.array([n.split("__", 1)[-1] for n in prep_gbm.get_feature_names_out()])
rng = np.random.default_rng(SEED)
take = rng.choice(len(transformed), size=min(250, len(transformed)), replace=False)
explainer = shap.TreeExplainer(booster)
shap_values = explainer.shap_values(transformed[take])
if isinstance(shap_values, list):
    shap_values = shap_values[-1]
shap_values = np.asarray(shap_values)
if shap_values.ndim == 3:
    shap_values = shap_values[:, :, -1]
mean_abs = np.abs(shap_values).mean(axis=0)
shap_order = np.argsort(mean_abs)[::-1][:8]
shap_rows = [
    {"feature": str(gbm_names[i]), "mean_abs": float(mean_abs[i])}
    for i in shap_order
]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
plot_rows = list(reversed(shap_rows))
ax.barh([r["feature"] for r in plot_rows], [r["mean_abs"] for r in plot_rows], color="#D98F2B")
ax.set_xlabel("Mean absolute SHAP")
ax.set_title("What moves the gradient booster")
fig.tight_layout()
savefig(fig, "shap.png")

# The network has no trees. Shuffle one column at a time and record the
# drop in test AUROC. This is permutation importance, not SHAP.
def mlp_proba(matrix):
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(matrix.astype(np.float32))).squeeze(1)
        return torch.sigmoid(logits).numpy()

mlp_names = np.array([n.split("__", 1)[-1] for n in prep.get_feature_names_out()])
sample_n = min(400, len(X_test))
sample_ix = rng.choice(len(X_test), size=sample_n, replace=False)
X_s = X_test[sample_ix].copy()
y_s = y_test[sample_ix]
baseline = roc_auc_score(y_s, mlp_proba(X_s))
drops = np.zeros(X_s.shape[1])
for col in range(X_s.shape[1]):
    scores = []
    for _ in range(3):
        shuffled = X_s.copy()
        rng.shuffle(shuffled[:, col])
        scores.append(baseline - roc_auc_score(y_s, mlp_proba(shuffled)))
    drops[col] = float(np.mean(scores))
perm_order = np.argsort(drops)[::-1][:8]
perm_rows = [
    {"feature": str(mlp_names[i]), "auroc_drop": float(drops[i])}
    for i in perm_order
]
print("shap_top", shap_rows[0]["feature"], "mlp_top", perm_rows[0]["feature"])
'''

AUDIT = r'''
def subgroup_rows(frame, pred, column):
    rows = []
    # pred is in row order. The frame must already be reset to 0..n-1.
    pred = np.asarray(pred)
    for level, part in frame.groupby(column):
        y = part[TARGET].to_numpy()
        p = pred[part.index.to_numpy()]
        n = int(len(y))
        n_pos = int(y.sum())
        # Small cells are identifiable and the AUROC is noise. Leave them out.
        if n < MIN_CELL or n_pos < MIN_CELL or (n - n_pos) < MIN_CELL:
            continue
        rows.append({
            "group": column,
            "level": str(level),
            "n": n,
            "auroc": float(roc_auc_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
        })
    return rows


# Index-aligned predictions. GBM is pre-specified for this audit so we do
# not pick the winner after seeing the test metrics.
test_aud = test.reset_index(drop=True)
subgroups = subgroup_rows(test_aud, p_gbm, "gender") + subgroup_rows(test_aud, p_gbm, "insurance")

if subgroups:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    labels = [f"{r['group']}={r['level']}" for r in subgroups]
    ax.barh(labels[::-1], [r["auroc"] for r in subgroups][::-1], color="#2F6F9F")
    ax.axvline(model_metrics["GBM"]["auroc"], color="#D98F2B", lw=1)
    ax.set_xlabel("AUROC")
    ax.set_title("Gradient booster by gender and insurance")
    fig.tight_layout()
    savefig(fig, "subgroups.png")
print("subgroup_rows", len(subgroups))
'''

SAVE = r'''
def convert(value):
    if isinstance(value, dict):
        return {k: convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [convert(v) for v in value]
    if isinstance(value, (np.floating, float)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


metrics = {
    "source": SOURCE,
    "n_stays": int(len(cohort)),
    "n_patients": int(cohort[GROUP_COL].nunique()),
    "n_readmit": int(cohort[TARGET].sum()),
    "readmit_rate": float(cohort[TARGET].mean()),
    "n_train": int(len(train)),
    "n_val": int(len(val)),
    "n_test": int(len(test)),
    "n_train_patients": int(train[GROUP_COL].nunique()),
    "n_test_patients": int(test[GROUP_COL].nunique()),
    "attrition": attrition.to_dict(orient="records"),
    "disposition": [
        {"level": str(r.discharge_location), "n": int(r.n), "rate": float(r.rate)}
        for r in disp.itertuples(index=False)
    ],
    "missing": [
        {"feature": str(idx), "pct": float(val)}
        for idx, val in miss_pct.items()
    ],
    "leakage": {"random_auroc": random_auroc, "grouped_auroc": grouped_auroc},
    "gbm_params": best_params,
    "gbm_cv_auroc": best_cv,
    "mlp_best_epoch": int(best_epoch),
    "mlp_val_loss": float(best_loss),
    "models": model_metrics,
    "odds_ratios": odds_rows,
    "operating_point": {
        "precision": op_precision,
        "recall": op_recall,
        "flag_rate": float(flagged.mean()),
        "prevalence": float(y_test.mean()),
    },
    "shap_top": shap_rows,
    "mlp_perm_top": perm_rows,
    "subgroups": subgroups,
}
# hadm_id and subject_id are never written here.
(ART / "metrics.json").write_text(json.dumps(convert(metrics), indent=2))
print("wrote metrics", ART.name)
'''


def build():
    sql_cells = [
        md(
            """
# Predicting 30-day readmission after a heart-failure discharge

This notebook builds a cohort from MIMIC-IV, then compares four models that score the risk of an all-cause readmission within 30 days: a LACE-style point score, L2 logistic regression, a gradient booster, and a small neural net.

The code comments are the tutorial. Read them in order. Do not print individual stays. Every table in this notebook is an aggregate, and groups smaller than 10 stays are dropped.

**Data use.** MIMIC-IV is credentialed. Do not commit the parquet cache, paste rows into chat, or put patient-level output in a slide. The cache for a real run lives under `data/hf_private/` and `build/hf_private/`.
            """
        ),
        md(
            """
## 1. Setup

`HF_SOURCE=bigquery` reads MIMIC-IV. `HF_SOURCE=synthetic` builds a fake cohort with the same columns so the code can be checked without the dataset. Synthetic numbers are not results.

Set `GOOGLE_CLOUD_PROJECT` to your own billing project before a real run. The query is billed to that project.
            """
        ),
        code(SETUP),
        md(
            """
## 2. Cohort, label, and features

An index stay is a heart-failure admission (ICD-10 `I50*` or ICD-9 `428*`), age 18 or older, the patient left alive, and the discharge location is not hospice. Patients who die within 30 days without a readmission are removed so they are not counted as successes.

The label is 1 when that patient's **next admission of any kind** starts 0 to 30 days after discharge. `LEAD()` is computed over all of the patient's admissions, then the heart-failure stays are filtered. Doing it in the other order would miss a readmission coded as something other than heart failure.

Features are limited to what is known by discharge: demographics, the stay, prior utilization, comorbidity flags, a simplified Charlson score, first and last in-stay labs, and inpatient heart-failure medication orders. Race is not a feature. `labevents` is filtered to eight itemids. Do not remove that list.

The full query is below. It is generated from `sql/hf_ml_cohort.py`, which is the file to edit if you change the cohort.
            """
        ),
        md(sql_block("### Cohort query", cohort_sql())),
        md(sql_block("### Attrition query", attrition_sql())),
        md(
            """
## 3. Load

A real run downloads once and caches a parquet file. Later cells read the cache. `HF_REFRESH=1` forces a new download. The synthetic path never contacts BigQuery.

The printed line is a count of stays, patients, and readmissions. There is no row preview.
            """
        ),
        code(LOAD),
        md(
            """
## 4. Patient-level split

Hold out patients, not stays. Stratify on whether the patient had any 30-day readmission so the test set keeps the outcome rate. The fractions are about 70% train, 15% validation, 15% test.
            """
        ),
        code(SPLIT),
        md(
            """
## 5. A first look

The bar chart is the exclusion funnel. The readmission rates are by discharge location, with any location that has fewer than 10 stays removed. Missingness is why the preprocessor keeps a missingness indicator: a first stay has no "days since last discharge", and that absence is information.
            """
        ),
        code(EDA),
        md(
            """
## 6. Leakage, then the preprocessor

The same small booster is fit twice on the training patients only. A random stay split lets two admissions from one person fall on both sides. The patient split does not. If the random split scores higher, that gap is leakage, not a better model.

The test patients are not used in this cell.

The preprocessor is defined here and reused for every feature model: median impute, a missingness indicator, standard scaling, and one-hot encoding with the first level as the reference.
            """
        ),
        code(LEAKAGE),
        md(
            """
## 7. Four models

**LACE.** Points for length of stay, emergent admission, a Charlson band, and emergency visits in the prior 6 months (van Walraven 2010). A one-variable logistic regression, fit on train+validation, turns the points into a probability. The Charlson input is the simplified score from the SQL, so treat this as a bedside-style baseline.

**Logistic regression.** L2 penalty, `C=1`, fit on the preprocessed features. No class weight: about a quarter of stays are readmissions, and weighting would make the calibration plot harder to read.

**Gradient booster.** `HistGradientBoostingClassifier`. Four small grids, scored with 3-fold `StratifiedGroupKFold` inside the training patients. Early stopping is off, because the booster's own validation slice is a row split. The winning settings are refit on train+validation.

**Neural net.** The next cell. It is the only model that spends the validation set on early stopping, so it is not refit on train+validation.
            """
        ),
        code(MODELS),
        md(
            """
## 8. Neural net

A three-layer network trained with Adam and binary cross-entropy on logits. Dropout is on during training and off when we score. We keep the epoch with the lowest validation loss and restore those weights. Training stops after four epochs without an improvement, or at 20 epochs.
            """
        ),
        code(MLP),
        md(
            """
## 9. Test set, once

AUROC ranks patients. Average precision is the companion when the outcome is the minority class. Brier score and the calibration plot ask whether a predicted 0.30 actually happens about 30% of the time. The ribbon in the metrics is a percentile bootstrap of the AUROC. The point estimate is the whole test set, not the bootstrap mean.
            """
        ),
        code(EVAL),
        md(
            """
## 10. A threshold, and a decision curve

Flagging the top 20% by the gradient booster's score is an operating point you could staff: a case manager calls that fifth of discharges. Precision is the share of those calls who truly return. Recall is the share of all readmissions you catch.

The decision curve asks a different question. At a threshold probability *t*, you are willing to flag *1/t* patients to catch one readmission. Net benefit counts true positives and charges a penalty for false positives at that exchange rate. "Flag everyone" is the competitor. A model helps only where its curve is above both zero and flag-everyone.
            """
        ),
        code(DECIDE),
        md(
            """
## 11. What the models are using

Logistic odds ratios are per one standard deviation, or versus the reference category, because the preprocessor scaled the numbers and dropped the first category level.

SHAP attributes the booster's output to each feature for a sample of test stays. The bar is the mean absolute value: how much the feature moves the score, not the direction.

The network is explained with permutation importance: shuffle one column and measure the drop in AUROC. A large drop means the network was relying on that column.
            """
        ),
        code(EXPLAIN),
        md(
            """
## 12. Does the score work in each group?

The audit is pre-specified on the gradient booster, so we are not crowning whichever model won on the test set and then slicing it. Gender and insurance only. A level needs at least 10 stays, 10 readmissions, and 10 non-readmissions, or it is omitted. Race was not a predictor and is not an audit column here.

A lower AUROC in a large group is a reason to look at the features, not a reason to deploy a cutoff that differs by group without a clinical review.
            """
        ),
        code(AUDIT),
        md(
            """
## 13. Save aggregates

`metrics.json` feeds the slides. It stores counts, rates, and importances. It does not store identifiers, predictions, or rows.
            """
        ),
        code(SAVE),
        md(
            """
## 14. What this does not show

- The label is all-cause. A planned procedure at day 12 counts. This is not a CMS Hospital Readmissions Reduction Program measure.
- One health system, and dates are shifted. Nothing here is an external validation.
- ICD regexes are not a chart review. Inpatient orders miss home medications that were never rewritten.
- The Charlson score inside LACE is a shortened version. Say so if you quote the points.
- Labs are the first and last in-stay values, not a time series.
- No notes, no imaging, no outpatient fills.
- The model is not a decision tool. A high score is a reason to look, not an order.
            """
        ),
        md(
            """
## 15. Try it yourself

These three exercises match the practice slide. Do them on the synthetic source first so you can see the traceback if something breaks.

1. **Swap the booster.** Replace `HistGradientBoostingClassifier` with `RandomForestClassifier` in the models cell. Keep the same preprocessor and the same patient-level CV. Compare AUROC and the calibration plot to the gradient booster.
2. **Move the threshold.** In the decision-curve cell, change the call-list cutoff from the top fifth (`np.quantile(..., 0.80)`) to another fraction. Say how many discharges get a call and what happens to precision and recall.
3. **Add one lab.** Edit `sql/hf_ml_cohort.py` to pull one more `labevents` item id into the cohort, add it to `NUM_FEATURES`, and refit the logistic model. Watch how the odds ratio and the missingness chart change.

Optional extras: change the label so the next admission must carry a heart-failure code; add `pos_weight` to the network loss and look at calibration again; report the calibration slope, not just the curve.
            """
        ),
    ]
    nb = new_notebook(
        cells=sql_cells,
        metadata={
            "kernelspec": {
                "display_name": "Python (mimic-ml)",
                "language": "python",
                "name": "mimic-ml",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        nbf.write(nb, f)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
