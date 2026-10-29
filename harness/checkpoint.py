"""Save and load the state of a run so an interrupted run can carry on later.

The state is written after every step, to a temporary file first and then renamed, so a crash
in the middle of a write cannot leave a broken checkpoint behind.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

RESUME_NOTE = (
    "The run was interrupted and has now resumed. The workspace is exactly as it was when you stopped. "
    "Call list_dir on Downloads and on Organised to see what is already done and what is left, "
    "then carry on. Do not redo moves that are already done."
)


def save_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)  # atomic, the old checkpoint is replaced only once the new one is complete


def load_checkpoint(path: Path) -> dict:
    try:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"The checkpoint at {path} could not be read ({e}), so the run cannot be resumed.")
    for key in ("messages", "steps", "step", "n_errors", "agent", "prompt", "orig_hash"):
        if key not in state:
            raise ValueError(f"The checkpoint at {path} is incomplete (missing {key}), so the run cannot be resumed.")
    return state