"""Score the final workspace against the known correct answer."""
from __future__ import annotations

import csv
import io
from pathlib import Path

from .sandbox import sha
from .tasks import TaskSpec

LOG_PATH = "Organised/organise_log.csv"


def _norm(p: str) -> str:
    p = p.strip().strip('"').strip("'").replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    return p


def _parse_log(text: str) -> list[tuple[str, str]]:
    rows = []
    for r in csv.reader(io.StringIO(text)):
        if len(r) >= 2 and _norm(r[0]).lower() != "original_path":
            rows.append((_norm(r[0]), _norm(r[1])))
    return rows


def score(spec: TaskSpec, root: Path, finished: bool) -> dict:
    root = Path(root).resolve()
    found: dict[str, list[str]] = {}
    stray = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if rel == LOG_PATH:
            continue
        h = sha(p.read_bytes())
        found.setdefault(h, []).append(rel)
    known = {f.hash for f in spec.files}
    stray = sum(len(v) for h, v in found.items() if h not in known)

    n = len(spec.files)
    correct = missing = duplicated = misplaced = 0
    for f in spec.files:
        paths = found.get(f.hash, [])
        if not paths:
            missing += 1
        elif len(paths) > 1:
            duplicated += 1
        elif paths[0] == f.expected_path:
            correct += 1
        else:
            misplaced += 1

    log_file = root / LOG_PATH
    log_exists = log_file.is_file()
    rows = _parse_log(log_file.read_text(errors="replace")) if log_exists else []
    expected_rows = {(f.orig_path, f.expected_path) for f in spec.files}
    matched = len(set(rows) & expected_rows)
    extra_rows = len(set(rows) - expected_rows)
    orig_hash = {f.orig_path: f.hash for f in spec.files}
    consistent = 0
    for orig, new in set(rows):
        h = orig_hash.get(orig)
        if h is not None and new in found.get(h, []):
            consistent += 1
    n_rows = len(set(rows))

    log_exact = log_exists and matched == n and extra_rows == 0
    success = correct == n and stray == 0 and log_exact and finished
    return {
        "n_files": n,
        "n_correct": correct,
        "n_missing": missing,
        "n_duplicated": duplicated,
        "n_misplaced": misplaced,
        "n_stray": stray,
        "file_accuracy": correct / n if n else 0.0,
        "log_exists": log_exists,
        "log_match_frac": matched / n if n else 0.0,
        "log_consistent_frac": consistent / n_rows if n_rows else 0.0,
        "finished": finished,
        "success": success,
        "false_success": finished and not success,
    }
