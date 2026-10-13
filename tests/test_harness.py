import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.agent import BaselineAgent
from harness.mock import BrokenArgsLLM, LoopingLLM, OracleLLM
from harness.sandbox import Sandbox
from harness.scoring import score
from harness.tasks import generate_task, make_task_set
from run_experiment import run_once


# ---- task generator ---------------------------------------------------
def test_generation_is_deterministic():
    a, b = generate_task(7), generate_task(7)
    assert [(f.orig_path, f.content, f.expected_path) for f in a.files] == \
           [(f.orig_path, f.content, f.expected_path) for f in b.files]


def test_files_are_unique_and_counted():
    for spec in make_task_set(10, 10):
        assert len(spec.files) == 10
        assert len({f.hash for f in spec.files}) == 10
        assert len({f.orig_path for f in spec.files}) == 10
        assert len({f.expected_path for f in spec.files}) == 10
        assert sum(f.module == "NONE" for f in spec.files) == 2


def test_larger_tasks_still_valid():
    spec = generate_task(3, n_files=20)
    assert len({f.expected_path for f in spec.files}) == 20


# ---- sandbox ----------------------------------------------------------
@pytest.fixture
def box(tmp_path):
    spec = generate_task(1)
    spec.materialise(tmp_path)
    return spec, Sandbox(tmp_path, spec.truth())


def test_path_escape_blocked(box):
    _, sb = box
    assert sb.read_file("../secret.txt").startswith("ERROR")
    assert sb.make_dir("../../evil").startswith("ERROR")
    assert sb.move_file("Downloads/x.txt", "../x.txt").startswith("ERROR")


def test_move_refuses_to_overwrite_and_needs_folder(box):
    spec, sb = box
    f0, f1 = spec.files[0], spec.files[1]
    assert sb.move_file(f0.orig_path, "Organised/A/x.txt").startswith("ERROR")  # folder missing
    sb.make_dir("Organised/A")
    assert sb.move_file(f0.orig_path, "Organised/A/x.txt").startswith("OK")
    assert sb.move_file(f1.orig_path, "Organised/A/x.txt").startswith("ERROR")  # exists
    assert sb.move_file(f0.orig_path, "Organised/A/y.txt").startswith("ERROR")  # already moved


def test_move_events_flag_wrong_module(box):
    spec, sb = box
    f = next(x for x in spec.files if x.module != "NONE")
    sb.make_dir("Organised/WRONG")
    sb.move_file(f.orig_path, "Organised/WRONG/a.txt")
    assert sb.events[-1]["module_ok"] is False


# ---- scoring ----------------------------------------------------------
def test_untouched_workspace_scores_zero(box):
    spec, sb = box
    m = score(spec, sb.root, finished=False)
    assert m["n_correct"] == 0 and m["n_misplaced"] == len(spec.files) and not m["success"]


def test_oracle_succeeds_end_to_end():
    spec = generate_task(11)
    out = run_once(BaselineAgent, OracleLLM(spec), spec, max_steps=80)["metrics"]
    assert out["success"] and out["file_accuracy"] == 1.0 and out["stop_reason"] == "finish"
    assert out["n_wrong_moves"] == 0 and out["first_error_step"] is None


def test_wrong_move_is_a_false_success_and_cascades_are_counted():
    spec = generate_task(11)
    out = run_once(BaselineAgent, OracleLLM(spec, wrong_move_index=2), spec, max_steps=80)["metrics"]
    assert not out["success"] and out["false_success"]
    assert out["n_correct"] == len(spec.files) - 1 and out["n_wrong_moves"] == 1
    assert out["first_error_step"] is not None and out["moves_after_first_error"] > 0
    assert out["log_match_frac"] == 1.0  # log claims the right place...
    assert out["log_consistent_frac"] < 1.0  # ...but disagrees with the real state


def test_missing_log_fails():
    spec = generate_task(11)
    out = run_once(BaselineAgent, OracleLLM(spec, skip_log=True), spec, max_steps=80)["metrics"]
    assert out["file_accuracy"] == 1.0 and not out["log_exists"] and not out["success"]


# ---- agent loop -------------------------------------------------------
def test_step_limit_is_respected():
    spec = generate_task(5)
    out = run_once(BaselineAgent, LoopingLLM(), spec, max_steps=7)["metrics"]
    assert out["steps_used"] == 7 and out["stop_reason"] == "max_steps" and not out["finished"]


def test_invalid_json_arguments_become_tool_errors():
    spec = generate_task(5)
    out = run_once(BaselineAgent, BrokenArgsLLM(), spec, max_steps=10)["metrics"]
    assert out["n_tool_errors"] == 1 and out["stop_reason"] == "finish"


def test_history_grows_with_every_step():
    spec = generate_task(5)
    seen = []

    class Spy(LoopingLLM):
        def chat(self, messages, tools):
            seen.append(len(messages))
            return super().chat(messages, tools)

    run_once(BaselineAgent, Spy(), spec, max_steps=4)
    assert seen == [2, 4, 6, 8]  # system + user, then one assistant and one tool message per step
