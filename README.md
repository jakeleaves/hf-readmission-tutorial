# 30-day readmission after a heart-failure discharge

Tutorial code for a MIMIC-IV cohort and four risk models: a LACE-style
score, L2 logistic regression, a gradient booster, and a small neural net.

The notebook comments are the lesson. Nothing in this repository is patient
data.

## Check the code without MIMIC

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-ml.txt
.venv/bin/python run_hf_ml.py --source synthetic
```

That builds a fake cohort with the same columns, runs the notebook, writes
slides, and exports a PDF under `build/hf_synthetic/`. Synthetic numbers are
a rehearsal, not a result.

## Credentialed run (MIMIC-IV on BigQuery)

```bash
export GOOGLE_CLOUD_PROJECT=your-billing-project
HF_CODE_URL=https://your-anonymous-repo .venv/bin/python run_hf_ml.py --source bigquery
```

Queries are billed to `GOOGLE_CLOUD_PROJECT`, not to the data host.

Outputs land in `build/hf_private/`:

- `hf_readmission_ml_tutorial.ipynb`
- `hf_readmission_ml_tutorial.pdf`
- `hf_readmission_ml_slides.pptx`

The cohort cache is `data/hf_private/cohort.parquet`. Do not commit either
directory. Set `HF_CODE_URL` to this repository's URL before the slide build
if you want the link in the title-slide speaker notes. `run_hf_ml.py` reads
that variable when it builds the deck.

`HF_REFRESH=1` ignores the cache and queries again.
