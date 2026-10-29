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
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from .agent import BaselineAgent
from .checkpoint import RESUME_NOTE, load_checkpoint, save_checkpoint
from .improved import ImprovedAgent
from .verified_agent import VerifiedAgent
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

# Added to the prompt of the improved agent only. The baseline prompt has no folder rule.
FOLDER_SECTION = """

Folders
Keep the folder structure. A file that sits inside subfolders of Downloads keeps the same subfolder path inside its module folder. For example Downloads/Lab 5/starter/a.java goes to Organised/ABC1234/Lab 5/starter/a.java. Never flatten a subfolder, and never turn a subfolder into a module folder. If a whole subfolder belongs to one module, move it in one step with move_folder to Organised/<module>/<folder name>. If a subfolder holds files from different modules, recreate the same subfolder path under each module that receives files from it. Files directly in Downloads go directly into the module folder."""

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


def build_prompt(modules: list[dict] | None = None, folders: bool = False) -> str:
    """folders=True adds the folder rules. Only the improved agent uses it."""
    if not modules:
        text = REAL_PROMPT
    else:
        listing = "\n".join(f"- {m['code']}" + (f" ({m['name']})" if m.get("name") else "") for m in modules)
        text = TIMETABLE_PROMPT.format(modules=listing)
    return text + FOLDER_SECTION if folders else text


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


def folder_kept(original: str, locs: list[str]):
    """Did the file keep its subfolder path inside the module folder?

    Returns None when the file is not inside any module folder, otherwise (kept, folder_it_ended_up_in).
    Only the folder part is compared, so a renamed file after a name clash still counts.
    """
    want = "/".join(original.split("/")[:-1])
    placed = [l.split("/") for l in locs if l.startswith("Organised/") and len(l.split("/")) >= 3]
    if not placed:
        return None
    for parts in placed:
        if "/".join(parts[2:-1]) == want:
            return True, want
    return False, "Organised/" + "/".join(placed[0][1:-1])


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
AGENT_CLASSES = {"baseline": BaselineAgent, "verified": VerifiedAgent, "improved": ImprovedAgent}


def run_organise(input_dir, output_dir, llm, max_steps: int = 250, labels: dict | None = None, on_step=None,
                 modules: list[dict] | None = None, agent_type: str = "baseline",
                 resume: bool = False, fresh_context: bool = False) -> dict:
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found, {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("The output folder must not be inside the input folder.")
    if agent_type not in AGENT_CLASSES:
        raise ValueError(f"Unknown agent type: {agent_type}")
    agent_cls = AGENT_CLASSES[agent_type]
    checkpointing = getattr(agent_cls, "supports_checkpoint", False)
    if (resume or fresh_context) and not checkpointing:
        raise ValueError(f"Only the improved agent can be resumed, not the {agent_type} agent.")
    if fresh_context and not resume:
        raise ValueError("fresh context only applies when resuming.")

    orig_hash = {p.relative_to(input_dir).as_posix(): _hash(p) for p in _files(input_dir)}
    ckpt_path = output_dir / "_run" / "checkpoint.json"
    saved = None
    if resume:
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Nothing to resume, there is no checkpoint at {ckpt_path}.")
        saved = load_checkpoint(ckpt_path)
        if saved.get("complete"):
            raise ValueError("That run already finished. Start a new run with a new output folder.")
        if saved["agent"] != agent_type:
            raise ValueError(f"That run used the {saved['agent']} agent, so resume it with --agent {saved['agent']}.")
        if saved["orig_hash"] != orig_hash:
            raise ValueError("The input folder has changed since the run was saved, so it cannot be resumed.")
        work = output_dir / "_run" / "work"
        if not (work / "Downloads").is_dir():
            raise FileNotFoundError("The saved workspace is missing, so the run cannot be resumed.")
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"{output_dir} already has files in it. Choose a new output folder or empty this one.")
        # A resumable run keeps its workspace inside the output folder. The others use a throwaway folder.
        work = output_dir / "_run" / "work" if checkpointing else Path(tempfile.mkdtemp(prefix="organise_"))
        work.mkdir(parents=True, exist_ok=True)

    complete = False
    try:
        if not resume:
            shutil.copytree(input_dir, work / "Downloads")  # the original files are never touched
        sb = Sandbox(work)
        agent = agent_cls(llm, max_steps=max_steps, on_step=on_step)
        if saved and modules is None:
            modules = saved.get("modules")  # the timetable is not read again on resume
        prompt = saved["prompt"] if saved else build_prompt(modules, folders=getattr(agent, "uses_folder_rules", False))
        model_id = str(getattr(llm, "model", "unknown"))
        models = list(saved.get("models", [])) if saved else []
        if model_id not in models:
            models.append(model_id)
        resumes = saved.get("resumes", 0) + 1 if saved else 0

        resume_state = None
        if saved:
            sb.events = saved.get("events", [])
            history = saved["messages"][:2] if fresh_context else saved["messages"]  # first two are the system and task prompts
            resume_state = {"messages": history + [{"role": "user", "content": RESUME_NOTE}], "steps": saved["steps"],
                            "n_errors": saved["n_errors"], "step": saved["step"],
                            "folder_origin": saved.get("folder_origin", {})}
        if checkpointing:
            def save(state: dict) -> None:
                save_checkpoint(ckpt_path, {**state, "events": sb.events, "agent": agent_type, "models": models,
                                            "prompt": prompt, "orig_hash": orig_hash, "complete": False, "resumes": resumes,
                                            "modules": modules})
            agent.checkpoint_cb = save

        task = SimpleNamespace(prompt=prompt)
        result = agent.run(task, sb, resume_state) if checkpointing else agent.run(task, sb)
        complete = result.stop_reason in ("finish", "no_tool_call")
        if checkpointing and ckpt_path.is_file():  # record how the run ended
            final = load_checkpoint(ckpt_path)
            final.update({"complete": complete, "stop_reason": result.stop_reason, "stop_note": result.note})
            save_checkpoint(ckpt_path, final)
        locations = _locations(work)

        output_dir.mkdir(parents=True, exist_ok=True)
        for old in ("Organised", "_left_behind"):  # a resumed run exports its latest state again
            shutil.rmtree(output_dir / old, ignore_errors=True)
        if (work / "Organised").is_dir():
            shutil.copytree(work / "Organised", output_dir / "Organised")
        left = work / "Downloads"
        for p in _files(left):
            dest = output_dir / "_left_behind" / p.relative_to(left)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)

        rows, per_module = [], Counter()
        structure = {"checked": 0, "kept": 0, "changed": []}
        for orig, h in orig_hash.items():
            locs = locations.get(h, [])
            fk = folder_kept(orig, locs)
            if fk is not None:
                structure["checked"] += 1
                if fk[0]:
                    structure["kept"] += 1
                else:
                    structure["changed"].append({"file": orig, "original_folder": "/".join(orig.split("/")[:-1]) or "(top level)",
                                                 "ended_up_in": fk[1]})
            mods = sorted({_module_of(l) for l in locs})
            per_module[mods[0] if mods else "MISSING"] += 1
            rows.append({"original": orig, "final_location": " | ".join(locs) or "MISSING",
                         "module_folder": ", ".join(mods) or "MISSING"})
        # The file name carries the time the report is written, for example moves_report_20261025_12.20pm.csv
        stamp = datetime.now().strftime("%Y%m%d_%I.%M%p").lower()
        report_path = output_dir / f"moves_report_{stamp}.csv"
        with open(report_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["original", "final_location", "module_folder"])
            w.writeheader()
            w.writerows(rows)

        listed = {m["code"].upper() for m in modules or []}
        special = {"UNSORTED", "LEFT_BEHIND", "MISSING", "OTHER"}
        summary = {
            "agent": agent_type,
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
            "report_file": report_path.name,
            "folder_structure": structure,
            "stop_note": result.note,
            "resumable": checkpointing and not complete,
            "models_used": models,
            "resumes": resumes,
        }
        if labels:
            summary["score"] = score_labels(labels, orig_hash, locations)
        run_dir = output_dir / "_run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "transcript.json").write_text(
            json.dumps({"summary": summary, "steps": result.steps, "move_events": sb.events}, indent=1))
        return summary
    finally:
        if not checkpointing or complete:  # an unfinished resumable run keeps its workspace
            shutil.rmtree(work, ignore_errors=True)