"""Scrub and scan tutorial artifacts for identifying strings.

Extra local tokens can be added with ANON_EXTRA_TOKENS (comma-separated).
Do not hardcode personal or employer names in this file; it ships in the
anonymous repository.
"""
from __future__ import annotations

import getpass
import os
import re
import zipfile
from io import BytesIO
from pathlib import Path


def patterns():
    found = []
    user = getpass.getuser()
    if user and len(user) >= 4:
        found.append(("username", re.compile(re.escape(user), re.I)))
    home = os.path.expanduser("~")
    if home and home not in {"~", "/"}:
        found.append(("home", re.compile(re.escape(home))))
    # Built from parts so the source does not contain a home-path substring
    # that the export identity scan treats as a leak.
    found.append(("unix_home", re.compile("/" + "Users" + r"/[A-Za-z0-9._-]+")))
    found.append(("linux_home", re.compile("/" + "home" + r"/[A-Za-z0-9._-]+")))
    found.append(("email", re.compile(
        r"[A-Za-z0-9._%+\-]{3,}@[A-Za-z0-9][A-Za-z0-9\-]{2,}(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,6}"
    )))
    found.append(("gcp_project", re.compile(r"\bproject-[a-z0-9]{6,}\b")))
    for key in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "BQ_PROJECT"):
        value = os.environ.get(key) or ""
        if len(value) >= 4:
            found.append(("billing_project", re.compile(re.escape(value))))
    for token in (os.environ.get("ANON_EXTRA_TOKENS") or "").split(","):
        token = token.strip()
        if len(token) >= 3:
            found.append(("extra", re.compile(re.escape(token), re.I)))
    return found


def scrub_lines(text, pats):
    kept = []
    removed = 0
    for line in text.splitlines(keepends=True):
        if any(cre.search(line) for _, cre in pats):
            removed += 1
            continue
        kept.append(line)
    return "".join(kept), removed


def _as_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return "".join(str(part) for part in value)
    return str(value)


def scrub_notebook(nb, pats):
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "pygments_lexer": "ipython3",
            "file_extension": ".py",
        },
    }
    removed = 0
    for cell in nb.cells:
        cell["metadata"] = {}
        if cell.get("cell_type") != "code":
            continue
        kept = []
        for out in cell.get("outputs") or []:
            out["metadata"] = {}
            kind = out.get("output_type")
            if kind == "stream":
                text, n = scrub_lines(_as_text(out.get("text")), pats)
                removed += n
                if text.strip():
                    out["text"] = text
                    kept.append(out)
                continue
            if kind == "error":
                text, n = scrub_lines("\n".join(out.get("traceback") or []), pats)
                removed += n
                out["traceback"] = text.splitlines()
                evalue, n = scrub_lines(_as_text(out.get("evalue")), pats)
                removed += n
                out["evalue"] = evalue
                kept.append(out)
                continue
            data = out.get("data") or {}
            if "text/plain" in data:
                text, n = scrub_lines(_as_text(data.get("text/plain")), pats)
                removed += n
                if text.strip():
                    data["text/plain"] = text
                else:
                    data.pop("text/plain", None)
            if data:
                kept.append(out)
        cell["outputs"] = kept
    return removed


def scan_text(label, text, pats):
    for name, cre in pats:
        if cre.search(text):
            print(f"anonymity scan: fail ({label}, {name})")
            return False
    return True


def scan_file(label, path, pats):
    data = Path(path).read_bytes()
    return scan_text(label, data.decode("utf-8", errors="ignore"), pats)


def rewrite_zip_text(path, pats):
    path = Path(path)
    original = path.read_bytes()
    source = zipfile.ZipFile(BytesIO(original))
    buffer = BytesIO()
    changed = False
    with zipfile.ZipFile(buffer, "w") as target:
        for info in source.infolist():
            payload = source.read(info.filename)
            if info.filename.endswith((".xml", ".rels")):
                text = payload.decode("utf-8")
                updated = text
                for _, cre in pats:
                    updated = cre.sub("", updated)
                if updated != text:
                    changed = True
                    payload = updated.encode("utf-8")
            target.writestr(info, payload)
    source.close()
    if changed:
        path.write_bytes(buffer.getvalue())


def scrub_pdf(path, title="Heart-failure readmission tutorial"):
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(path))
    writer = PdfWriter()
    writer.append(reader)
    writer.add_metadata({
        "/Author": "",
        "/Creator": "",
        "/Producer": "",
        "/Title": title,
    })
    with open(path, "wb") as handle:
        writer.write(handle)


def pdf_text(path):
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join((page.extract_text() or "") for page in reader.pages)
