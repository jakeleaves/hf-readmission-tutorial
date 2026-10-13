"""Build and execute the heart-failure readmission tutorial, then the slides and PDF.

Synthetic runs are for checking the code. Real runs query MIMIC-IV and write
only under data/hf_private and build/hf_private.

    .venv-ml/bin/python run_hf_ml.py --source synthetic
    HF_CODE_URL=https://... .venv-ml/bin/python run_hf_ml.py --source bigquery
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from anon_scan import (
    patterns,
    pdf_text,
    rewrite_zip_text,
    scan_file,
    scan_text,
    scrub_notebook,
    scrub_pdf,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _fail(source: str, exc: CellExecutionError) -> None:
    name = exc.ename or type(exc).__name__
    # A BigQuery traceback can echo a row. Keep it to the cell and the type.
    if source != "synthetic":
        print(f"FAILED cell {exc.cell_index} ({name})")
        return
    print(f"FAILED cell {exc.cell_index} ({name})")
    lines = list(exc.traceback or [])
    if lines:
        print("\n".join(lines[-40:]))


def _export_pdf(nb_path: str, pdf_path: str, source: str) -> bool:
    out_dir = os.path.dirname(pdf_path)
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    web = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "webpdf",
            "--allow-chromium-download",
            "--output", base,
            "--output-dir", out_dir,
            nb_path,
        ],
        capture_output=True, text=True,
    )
    if web.returncode == 0 and os.path.exists(pdf_path):
        return True
    html_path = os.path.join(out_dir, base + ".html")
    html = subprocess.run(
        [
            sys.executable, "-m", "jupyter", "nbconvert",
            "--to", "html",
            "--output", os.path.basename(html_path),
            "--output-dir", out_dir,
            nb_path,
        ],
        capture_output=True, text=True,
    )
    if html.returncode != 0:
        print("FAILED pdf (NbConvertFailed)")
        if source == "synthetic":
            tail = (web.stderr or html.stderr or "").strip().splitlines()
            print("\n".join(tail[-20:]))
        return False
    if not os.path.exists(CHROME):
        print("FAILED pdf (ChromeMissing)")
        return False
    try:
        subprocess.check_call([
            CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_path}",
            "file://" + html_path,
        ])
    except Exception as exc:
        print(f"FAILED pdf ({type(exc).__name__})")
        if os.path.exists(html_path):
            os.unlink(html_path)
        return False
    if os.path.exists(html_path):
        os.unlink(html_path)
    return os.path.exists(pdf_path)


def prepare_env(source: str) -> None:
    os.environ["HF_SOURCE"] = source
    os.environ["PYTHONHASHSEED"] = os.environ.get("PYTHONHASHSEED", "42")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/hf-ml-cache/mpl")
    os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hf-ml-cache/numba")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/hf-ml-cache")
    os.environ.setdefault("IPYTHONDIR", "/tmp/hf-ml-cache/ipython")
    os.environ.setdefault("JUPYTER_CONFIG_DIR", "/tmp/hf-ml-cache/jupyter")
    os.environ.setdefault("JUPYTER_DATA_DIR", "/tmp/hf-ml-cache/jupyter-data")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    for folder in (
        "/tmp/hf-ml-cache/mpl",
        "/tmp/hf-ml-cache/numba",
        "/tmp/hf-ml-cache",
        "/tmp/hf-ml-cache/ipython",
        "/tmp/hf-ml-cache/jupyter",
        "/tmp/hf-ml-cache/jupyter-data",
    ):
        os.makedirs(folder, exist_ok=True)


def execute(source: str) -> bool:
    prepare_env(source)
    subprocess.check_call([
        sys.executable, "-m", "ipykernel", "install", "--sys-prefix",
        "--name", "mimic-ml", "--display-name", "Python (mimic-ml)",
    ])
    template = os.path.join(ROOT, "hf_readmission_ml_tutorial.ipynb")
    nb = nbformat.read(template, as_version=4)
    if source == "synthetic":
        art = os.path.join(ROOT, "build", "hf_synthetic")
    else:
        art = os.path.join(ROOT, "build", "hf_private")
    os.makedirs(art, exist_ok=True)
    nb_path = os.path.join(art, "hf_readmission_ml_tutorial.ipynb")
    pdf_path = os.path.join(art, "hf_readmission_ml_tutorial.pdf")
    pptx_path = os.path.join(art, "hf_readmission_ml_slides.pptx")
    client = NotebookClient(
        nb,
        timeout=7200,
        kernel_name="mimic-ml",
        resources={"metadata": {"path": ROOT}},
    )
    try:
        client.execute()
    except CellExecutionError as exc:
        _fail(source, exc)
        if source == "synthetic":
            with open(nb_path, "w") as f:
                nbformat.write(nb, f)
        return False
    except Exception as exc:
        print(f"FAILED kernel ({type(exc).__name__})")
        if source == "synthetic":
            print(type(exc).__name__)
        return False

    pats = patterns()
    removed = scrub_notebook(nb, pats)
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    if not scan_file("notebook", nb_path, pats):
        return False

    try:
        subprocess.check_call([
            sys.executable, os.path.join(ROOT, "make_hf_ml_slides.py"), art, pptx_path,
        ])
    except subprocess.CalledProcessError:
        print("FAILED slides (CalledProcessError)")
        return False
    rewrite_zip_text(pptx_path, pats)
    if not scan_file("slides", pptx_path, pats):
        return False

    if not _export_pdf(nb_path, pdf_path, source):
        return False
    try:
        scrub_pdf(pdf_path)
    except Exception as exc:
        print(f"FAILED pdf metadata ({type(exc).__name__})")
        return False
    if not scan_text("pdf", pdf_text(pdf_path), pats):
        return False
    if not scan_file("pdf-bytes", pdf_path, pats):
        return False

    print("OK")
    print("scrubbed lines:", removed)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["synthetic", "bigquery"])
    args = parser.parse_args()
    if not execute(args.source):
        sys.exit(1)


if __name__ == "__main__":
    main()
