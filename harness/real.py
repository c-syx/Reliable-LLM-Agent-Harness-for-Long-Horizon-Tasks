"""Run the agent on a real folder of files and export the organised result.

Nothing here contains a module list, a mapping from topics to modules, or a naming
rule. The model has to read the files and decide. The only answer key is an optional
labels.csv that you write by hand, and it is used for scoring after the run, never
shown to the agent.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from .agent import BaselineAgent
from .readers import extract_text
from .sandbox import Sandbox

REAL_PROMPT = """You are organising a student's folder of files using the tools provided. All paths are relative to the workspace root. The files to organise are in Downloads, including all of its subfolders.

Work out which university module each file belongs to by reading what is inside it. Do not rely on file names alone, because they can be misleading. A module code is a short course identifier made of letters followed by digits, for example ABC1234. It often appears in slide titles, headers, code comments or report text. Some files may not mention a code, so use the topic and the other files as evidence.

Rules
1. Create one folder per module inside Organised, named by the module code only, for example Organised/ABC1234. Only create folders for modules that really appear in the files.
2. Move each file into the folder of its module and keep its original file name.
3. If a file does not belong to any module, or you cannot tell, move it to Organised/Unsorted. Do not guess without evidence.
4. If two files would end up with the same name in one folder, rename one of them with a short suffix.
5. Write a CSV log to Organised/organise_log.csv with the header original_path,new_path,module,reason and one row per file. The reason is a few words on the evidence you used.
6. Do not delete files, and do not leave any file behind in Downloads.
7. Images and videos have no readable text. Use the file name and the other files in the same folder as evidence, or move them to Organised/Unsorted.
8. Call finish once every file is handled and you have checked your work."""

TIMETABLE_PROMPT = """You are organising a student's folder of files using the tools provided. All paths are relative to the workspace root. The files to organise are in Downloads, including all of its subfolders.

The student's timetable lists these modules. Most files will belong to one of them.
{modules}

Work out which module each file belongs to by reading what is inside it. Do not rely on file names alone, because they can be misleading. A module code is a short course identifier made of letters followed by digits. It often appears in slide titles, headers, code comments or report text.

Rules
1. Create one folder per module inside Organised, named by the module code only, for example Organised/ABC1234. Only create a folder when at least one file goes in it. Use the codes from the timetable where they fit.
2. Move each file into the folder of its module and keep its original file name.
3. Many files never state a module code. For those, use the topic, the wording, the other files in the same folder, and what each timetable module is about to pick the most likely module from the list. Use Organised/Unsorted only when the file is clearly not coursework, or when it does not plausibly fit any listed module.
4. If a file states a module code that is not on the timetable, trust the file and create a folder for that code.
5. If two files would end up with the same name in one folder, rename one of them with a short suffix.
6. Write a CSV log to Organised/organise_log.csv with the header original_path,new_path,module,reason and one row per file. The reason must start with the word stated if the module code was written in the file, or the word inferred if you chose the module from the topic or the timetable. Then add a few words on the evidence.
7. Do not delete files, and do not leave any file behind in Downloads.
8. Images and videos have no readable text. Use the file name and the other files in the same folder as evidence, or move them to Organised/Unsorted.
9. Call finish once every file is handled and you have checked your work."""

EXTRACT_PROMPT = """Below is the text of a student's timetable or module list. List every university module it contains.
Reply with only a JSON array and nothing else. Each item is an object with "code" (the module code exactly as written) and "name" (the module title if it is written in the text, otherwise an empty string). Each module appears once even if it has several classes. Do not add modules that are not in the text.

TIMETABLE TEXT
"""

# The general shape of a module code, letters then four digits. It is only used as a safety net
# when the model's answer cannot be parsed. It contains no actual module.
CODE_SHAPE = re.compile(r"\b[A-Z]{2,4}\d{4}[A-Z]{0,2}\b")


# ---- timetable and module list ---------------------------------------
def _clean_modules(items) -> list[dict]:
    out, seen = [], set()
    for m in items or []:
        if isinstance(m, str):
            code, name = m, ""
        elif isinstance(m, dict):
            code, name = str(m.get("code", "")), str(m.get("name", "") or "")
        else:
            continue
        code = re.sub(r"\s+", "", code).upper()
        if code and code not in seen:
            seen.add(code)
            out.append({"code": code, "name": name.strip()})
    return out


def _parse_json_list(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end <= start:
        return []
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def extract_registry(path, llm) -> dict:
    """Read a timetable and return {"modules": [{"code", "name"}], "method", "file"}.

    A saved registry .json is loaded as it is, which costs no model call. For any other file
    the text is extracted, the model lists the modules, and every code the model returns is
    checked against the timetable text so an invented module cannot get through.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Timetable file not found, {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        mods = _clean_modules(data["modules"] if isinstance(data, dict) else data)
        return {"modules": mods, "method": "saved registry", "file": path.name}

    text = extract_text(path, 20000)
    if text.startswith(("NOTE", "(no ")):
        raise ValueError(f"Could not get text out of {path.name}. {text} "
                         "Use a text based export (PDF, Word, Excel, ICS or a plain text list). "
                         "A screenshot of a timetable will not work.")
    squashed = re.sub(r"\s+", "", text).upper()
    mods = [m for m in _clean_modules(_parse_json_list(llm.chat([{"role": "user", "content": EXTRACT_PROMPT + text}], []).content))
            if m["code"] in squashed]
    method = "model"
    if not mods:
        mods = _clean_modules(sorted(set(CODE_SHAPE.findall(text))))
        method = "pattern fallback, names unknown"
    return {"modules": mods, "method": method, "file": path.name}


def save_registry(registry: dict, path) -> None:
    Path(path).write_text(json.dumps(registry, indent=1), encoding="utf-8")


def build_prompt(modules: list[dict] | None = None) -> str:
    if not modules:
        return REAL_PROMPT
    listing = "\n".join(f"- {m['code']}" + (f" ({m['name']})" if m.get("name") else "") for m in modules)
    return TIMETABLE_PROMPT.format(modules=listing)


# ---- helpers ---------------------------------------------------------
def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _locations(root: Path) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for p in _files(root):
        found.setdefault(_hash(p), []).append(p.relative_to(root).as_posix())
    return found


def _module_of(loc: str) -> str:
    parts = loc.split("/")
    if parts[0] == "Organised" and len(parts) >= 3:
        return parts[1]
    return "LEFT_BEHIND" if parts[0] == "Downloads" else "OTHER"


# ---- labels (optional answer key, written by hand) -------------------
def make_labels_template(input_dir: str | Path, out_csv: str | Path) -> int:
    input_dir = Path(input_dir)
    files = _files(input_dir)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "module"])
        for p in files:
            w.writerow([p.relative_to(input_dir).as_posix(), ""])
    return len(files)


def load_labels(path: str | Path) -> dict[str, str]:
    labels = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            f = (r.get("file") or "").strip().replace("\\", "/")
            m = (r.get("module") or "").strip()
            if f and m:  # blank rows are simply not scored
                labels[f] = m
    return labels


def score_labels(labels: dict[str, str], orig_hash: dict[str, str], locations: dict[str, list[str]]) -> dict:
    correct, wrong = 0, []
    for f, expected in labels.items():
        h = orig_hash.get(f)
        got = sorted({_module_of(l) for l in locations.get(h, [])}) if h else []
        if any(g.lower() == expected.lower() for g in got):
            correct += 1
        else:
            wrong.append({"file": f, "expected": expected, "got": ", ".join(got) or "MISSING"})
    n = len(labels)
    return {"n_labelled": n, "n_correct": correct, "accuracy": correct / n if n else 0.0, "wrong": wrong}


# ---- the run ---------------------------------------------------------
def run_organise(input_dir, output_dir, llm, max_steps: int = 250, labels: dict | None = None, on_step=None,
                 modules: list[dict] | None = None) -> dict:
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found, {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("The output folder must not be inside the input folder.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"{output_dir} already has files in it. Choose a new output folder or empty this one.")

    work = Path(tempfile.mkdtemp(prefix="organise_"))
    try:
        shutil.copytree(input_dir, work / "Downloads")  # the original files are never touched
        orig_hash = {p.relative_to(input_dir).as_posix(): _hash(p) for p in _files(input_dir)}
        sb = Sandbox(work)
        result = BaselineAgent(llm, max_steps=max_steps, on_step=on_step).run(
            SimpleNamespace(prompt=build_prompt(modules)), sb)
        locations = _locations(work)

        output_dir.mkdir(parents=True, exist_ok=True)
        if (work / "Organised").is_dir():
            shutil.copytree(work / "Organised", output_dir / "Organised")
        left = work / "Downloads"
        for p in _files(left):
            dest = output_dir / "_left_behind" / p.relative_to(left)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

        rows, per_module = [], Counter()
        for orig, h in orig_hash.items():
            locs = locations.get(h, [])
            mods = sorted({_module_of(l) for l in locs})
            per_module[mods[0] if mods else "MISSING"] += 1
            rows.append({"original": orig, "final_location": " | ".join(locs) or "MISSING",
                         "module_folder": ", ".join(mods) or "MISSING"})
        with open(output_dir / "moves_report.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["original", "final_location", "module_folder"])
            w.writeheader()
            w.writerows(rows)

        listed = {m["code"].upper() for m in modules or []}
        special = {"UNSORTED", "LEFT_BEHIND", "MISSING", "OTHER"}
        summary = {
            "n_files": len(orig_hash),
            "per_module": dict(per_module.most_common()),
            "unsorted": per_module.get("Unsorted", 0),
            "timetable_modules": sorted(listed),
            # module folders the agent made that are not on the timetable, only meaningful when a timetable was used
            "unlisted_folders": sorted(k for k in per_module if listed and k.upper() not in listed | special),
            "left_behind": per_module.get("LEFT_BEHIND", 0),
            "missing": per_module.get("MISSING", 0),
            "steps_used": len(result.steps),
            "stop_reason": result.stop_reason,
            "tool_errors": result.n_tool_errors,
            "output": str(output_dir),
        }
        if labels:
            summary["score"] = score_labels(labels, orig_hash, locations)
        run_dir = output_dir / "_run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "transcript.json").write_text(
            json.dumps({"summary": summary, "steps": result.steps, "move_events": sb.events}, indent=1))
        return summary
    finally:
        shutil.rmtree(work, ignore_errors=True)