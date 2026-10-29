import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.llm import LLMReply, LLMUnavailable, ToolCall
from harness.real import run_organise


def move(src, dst, kind="move_file"):
    return kind, {"src": src, "dst": dst}


class ScriptLLM:
    """Plays a fixed list of tool calls. Raises an error instead at fail_at, if given. Records what it was sent."""

    model = "test/model"

    def __init__(self, calls, fail_at=None, error=None):
        self.calls, self.i, self.fail_at, self.error = list(calls), 0, fail_at, error
        self.sent = []  # number of messages at each call

    def chat(self, messages, tools):
        self.sent.append(len(messages))
        if self.fail_at is not None and len(self.sent) - 1 == self.fail_at:
            raise self.error or LLMUnavailable("daily quota used up")
        if self.i >= len(self.calls):
            return LLMReply("done", [])
        name, args = self.calls[self.i]
        self.i += 1
        return LLMReply("", [ToolCall(f"c{self.i}", name, json.dumps(args))])


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "test_files"
    (d / "Lab").mkdir(parents=True)
    for n in ("a.txt", "b.txt", "c.txt"):
        (d / n).write_text(f"file {n}")
    (d / "Lab" / "x.txt").write_text("lab file")
    return d


FIRST = [("make_dir", {"path": "Organised"}), ("make_dir", {"path": "Organised/M"}),
         move("Downloads/a.txt", "Organised/M/a.txt"), move("Downloads/b.txt", "Organised/M/b.txt")]
REST = [move("Downloads/c.txt", "Organised/M/c.txt"), move("Downloads/Lab", "Organised/M/Lab", "move_folder"),
        ("finish", {"summary": "done"})]


def checkpoint(out):
    return json.loads((out / "_run" / "checkpoint.json").read_text())


# ---- stopping and resuming --------------------------------------------
def test_interrupted_run_keeps_its_work_and_resume_finishes_it(folder, tmp_path):
    out = tmp_path / "out"
    first = run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    assert first["resumable"] and first["stop_reason"] == "llm_unavailable" and "quota" in first["stop_note"]
    assert (out / "_run" / "work" / "Downloads" / "c.txt").is_file()      # the workspace is kept
    assert (out / "Organised" / "M" / "b.txt").is_file()                  # and the partial result is exported
    assert checkpoint(out)["complete"] is False and checkpoint(out)["step"] == 4

    second = run_organise(folder, out, ScriptLLM(REST), agent_type="improved", resume=True)
    assert not second["resumable"] and second["stop_reason"] == "finish" and second["resumes"] == 1
    assert second["steps_used"] == 7 and second["left_behind"] == 0 and second["missing"] == 0
    assert (out / "Organised" / "M" / "Lab" / "x.txt").is_file() and (out / "Organised" / "M" / "c.txt").is_file()
    assert second["folder_structure"] == {"checked": 4, "kept": 4, "changed": []}
    assert not (out / "_run" / "work").exists()                           # cleaned up once finished
    assert checkpoint(out)["complete"] is True
    assert len(list(out.glob("moves_report_*.csv"))) >= 1


def test_resume_gives_the_model_its_history_plus_a_note(folder, tmp_path):
    out = tmp_path / "out"
    run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    llm = ScriptLLM(REST)
    run_organise(folder, out, llm, agent_type="improved", resume=True)
    # 2 prompts + 4 steps of (assistant + tool) = 10 messages, plus the resume note
    assert llm.sent[0] == 2 + 4 * 2 + 1
    note = checkpoint(out)["messages"][2 + 4 * 2]
    assert note["role"] == "user" and "interrupted" in note["content"]


def test_fresh_context_resume_starts_the_history_clean(folder, tmp_path):
    out = tmp_path / "out"
    run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    llm = ScriptLLM(REST)
    summary = run_organise(folder, out, llm, agent_type="improved", resume=True, fresh_context=True)
    assert llm.sent[0] == 3 and summary["stop_reason"] == "finish"
    assert summary["steps_used"] == 7  # the step count and the record of earlier steps are kept


def test_resume_can_use_a_different_model_and_it_is_recorded(folder, tmp_path):
    out = tmp_path / "out"
    run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    other = ScriptLLM(REST)
    other.model = "other/model"
    summary = run_organise(folder, out, other, agent_type="improved", resume=True)
    assert summary["models_used"] == ["test/model", "other/model"]


def test_folder_check_memory_survives_a_resume(folder, tmp_path):
    out = tmp_path / "out"
    calls = [("make_dir", {"path": "Organised"}), ("make_dir", {"path": "Organised/Old"}), ("make_dir", {"path": "Organised/New"}),
             move("Downloads/Lab", "Organised/Old/Lab", "move_folder")]
    run_organise(folder, out, ScriptLLM(calls, fail_at=4), agent_type="improved")
    flatten = [move("Organised/Old/Lab/x.txt", "Organised/New/x.txt"), ("finish", {"summary": "x"})]
    summary = run_organise(folder, out, ScriptLLM(flatten), agent_type="improved", resume=True)
    assert summary["tool_errors"] == 1                                    # the flatten was still blocked after resuming
    assert (out / "Organised" / "Old" / "Lab" / "x.txt").is_file()


# ---- other failures ---------------------------------------------------
def test_an_error_that_waiting_cannot_fix_is_also_saved_by_the_improved_agent(folder, tmp_path):
    err = RuntimeError("OpenRouter error 402: insufficient credits")
    summary = run_organise(folder, tmp_path / "out", ScriptLLM(FIRST, fail_at=4, error=err), agent_type="improved")
    assert summary["stop_reason"] == "llm_error" and summary["resumable"] and "402" in summary["stop_note"]


def test_the_baseline_has_no_checkpoint_and_still_fails_hard(folder, tmp_path):
    with pytest.raises(RuntimeError):
        run_organise(folder, tmp_path / "out", ScriptLLM(FIRST, fail_at=4, error=RuntimeError("402")))
    out = tmp_path / "ok"
    run_organise(folder, out, ScriptLLM(FIRST + REST[:1] + [("finish", {"summary": "x"})]))
    assert not (out / "_run" / "checkpoint.json").exists()


def test_a_finished_improved_run_leaves_no_workspace(folder, tmp_path):
    out = tmp_path / "out"
    summary = run_organise(folder, out, ScriptLLM(FIRST + REST), agent_type="improved")
    assert not summary["resumable"] and not (out / "_run" / "work").exists()
    assert not list((out / "_run").glob("*.tmp"))                         # the atomic write left nothing behind


# ---- refusals ---------------------------------------------------------
def test_resume_refuses_when_it_cannot_be_done_safely(folder, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(FileNotFoundError):                                # nothing saved yet
        run_organise(folder, out, ScriptLLM([]), agent_type="improved", resume=True)
    run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    with pytest.raises(ValueError):                                       # the baseline cannot resume
        run_organise(folder, out, ScriptLLM([]), agent_type="baseline", resume=True)
    with pytest.raises(ValueError):                                       # fresh context needs resume
        run_organise(folder, tmp_path / "other", ScriptLLM([]), agent_type="improved", fresh_context=True)
    (folder / "d.txt").write_text("a new file")
    with pytest.raises(ValueError):                                       # the input changed
        run_organise(folder, out, ScriptLLM([]), agent_type="improved", resume=True)
    (folder / "d.txt").unlink()
    run_organise(folder, out, ScriptLLM(REST), agent_type="improved", resume=True)
    with pytest.raises(ValueError):                                       # already finished
        run_organise(folder, out, ScriptLLM([]), agent_type="improved", resume=True)


def test_a_new_run_still_needs_an_empty_output_folder(folder, tmp_path):
    out = tmp_path / "out"
    run_organise(folder, out, ScriptLLM(FIRST, fail_at=4), agent_type="improved")
    with pytest.raises(FileExistsError):
        run_organise(folder, out, ScriptLLM([]), agent_type="improved")