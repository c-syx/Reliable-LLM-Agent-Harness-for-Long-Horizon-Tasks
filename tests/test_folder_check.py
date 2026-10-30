import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.agent import BaselineAgent
from harness.folder_check import FolderCheck
from harness.improved import ImprovedAgent
from harness.llm import LLMReply, ToolCall
from harness.sandbox import Sandbox


def move(src, dst, kind="move_file"):
    return kind, {"src": src, "dst": dst}


# ---- the check on its own --------------------------------------------
def test_flattening_a_file_is_stopped_with_the_right_path():
    fc = FolderCheck()
    msg = fc.check(*move("Downloads/Lab 5/starter/a.java", "Organised/CS2030DE/a.java"))
    assert msg.startswith("ERROR: folder check failed")
    assert "Organised/CS2030DE/Lab 5/starter/a.java" in msg


def test_keeping_the_path_is_allowed_and_top_level_files_stay_top_level():
    fc = FolderCheck()
    assert fc.check(*move("Downloads/Lab 5/starter/a.java", "Organised/CS2030DE/Lab 5/starter/a.java")) is None
    assert fc.check(*move("Downloads/notes.txt", "Organised/ES1103/notes.txt")) is None
    assert fc.check(*move("Downloads/notes.txt", "Organised/ES1103/extra/notes.txt")) is not None  # wrapped in a new folder


def test_unsorted_follows_the_same_rule_and_a_clash_rename_is_fine():
    fc = FolderCheck()
    assert fc.check(*move("Downloads/Lab/a.txt", "Organised/Unsorted/a.txt")) is not None
    assert fc.check(*move("Downloads/Lab/a.txt", "Organised/Unsorted/Lab/a_2.txt")) is None


def test_folder_moves():
    fc = FolderCheck()
    ok = move("Downloads/Lab 5", "Organised/CS2030DE/Lab 5", "move_folder")
    assert fc.check(*ok) is None
    assert "must not become a module folder" in fc.check(*move("Downloads/Lab 5", "Organised/Lab 5", "move_folder"))
    assert fc.check(*move("Downloads/Lab 5", "Organised/CS2030DE/Lab", "move_folder")) is not None  # renamed
    assert fc.check(*move("Downloads/a/b", "Organised/M/b", "move_folder")) is not None            # lost its parent folder
    assert fc.check(*move("Downloads/a/b", "Organised/M/a/b", "move_folder")) is None


def test_it_remembers_origins_so_a_later_move_is_still_judged():
    fc = FolderCheck()
    fc.record("move_folder", {"src": "Downloads/Lab 5", "dst": "Organised/Old/Lab 5"}, "OK: moved folder")
    assert fc.check(*move("Organised/Old/Lab 5/x.java", "Organised/New/x.java")) is not None
    assert fc.check(*move("Organised/Old/Lab 5/x.java", "Organised/New/Lab 5/x.java")) is None
    fc.record("move_file", {"src": "Downloads/b/y.txt", "dst": "Organised/M/b/y.txt"}, "ERROR: failed")  # failed moves are not recorded
    assert fc.original_of("Organised/M/b/y.txt") is None


def test_things_it_cannot_judge_are_left_alone():
    fc = FolderCheck()
    assert fc.check(*move("Organised/X/a.txt", "Organised/Y/a.txt")) is None      # origin unknown
    assert fc.check(*move("Downloads/a/b.txt", "Downloads/c/b.txt")) is None      # not into Organised
    assert fc.check("read_file", {"path": "Downloads/a.txt"}) is None
    assert fc.check("move_file", {"src": 3, "dst": None}) is None
    assert fc.check(*move("Downloads/a/b.txt", "../Organised/M/b.txt")) is None


# ---- inside the agent loop -------------------------------------------
class ScriptLLM:
    def __init__(self, calls):
        self.calls, self.i = list(calls), 0

    def chat(self, messages, tools):
        if self.i >= len(self.calls):
            return LLMReply("done", [])
        name, args = self.calls[self.i]
        self.i += 1
        return LLMReply("", [ToolCall(f"c{self.i}", name, json.dumps(args))])


def workspace(tmp_path):
    (tmp_path / "Downloads" / "Lab 5").mkdir(parents=True)
    (tmp_path / "Downloads" / "Lab 5" / "a.java").write_text("class A {}")
    (tmp_path / "Organised" / "CS2030DE").mkdir(parents=True)
    return Sandbox(tmp_path)


def test_improved_agent_blocks_the_flatten_then_accepts_the_fix(tmp_path):
    sb = workspace(tmp_path)
    calls = [move("Downloads/Lab 5/a.java", "Organised/CS2030DE/a.java"),
             ("make_dir", {"path": "Organised/CS2030DE/Lab 5"}),
             move("Downloads/Lab 5/a.java", "Organised/CS2030DE/Lab 5/a.java"),
             ("finish", {"summary": "x"})]
    res = ImprovedAgent(ScriptLLM(calls), max_steps=10).run(SimpleNamespace(prompt="p"), sb)
    first = res.steps[0]["tool_calls"][0]["result"]
    assert first.startswith("ERROR: folder check failed") and "Organised/CS2030DE/Lab 5/a.java" in first
    assert res.n_tool_errors == 1 and res.stop_reason == "finish"
    assert (tmp_path / "Organised/CS2030DE/Lab 5/a.java").is_file()
    assert not (tmp_path / "Organised/CS2030DE/a.java").exists()  # the blocked move never happened


def test_baseline_has_no_folder_check_and_still_flattens(tmp_path):
    sb = workspace(tmp_path)
    calls = [move("Downloads/Lab 5/a.java", "Organised/CS2030DE/a.java"), ("finish", {"summary": "x"})]
    res = BaselineAgent(ScriptLLM(calls), max_steps=10).run(SimpleNamespace(prompt="p"), sb)
    assert res.n_tool_errors == 0 and (tmp_path / "Organised/CS2030DE/a.java").is_file()


def test_each_run_starts_with_a_fresh_memory(tmp_path):
    agent = ImprovedAgent(ScriptLLM([]), max_steps=2)
    agent.run(SimpleNamespace(prompt="p"), workspace(tmp_path))
    first = agent.folder_check
    agent.run(SimpleNamespace(prompt="p"), Sandbox(tmp_path))
    assert agent.folder_check is not first