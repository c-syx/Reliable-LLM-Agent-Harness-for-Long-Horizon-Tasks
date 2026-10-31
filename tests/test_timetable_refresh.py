import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.llm import LLMReply
from harness.real import load_or_refresh_registry, save_registry


class ListLLM:
    """Answers with whatever module codes are in the timetable text it is shown, and counts the calls."""

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        text = messages[0]["content"]
        codes = sorted({w for w in text.split() if len(w) == 6 and w[:2].isalpha() and w[2:].isdigit() and w.isupper()})
        return LLMReply(json.dumps([{"code": c, "name": ""} for c in codes]), [])


def codes(reg):
    return [m["code"] for m in reg["modules"]]


@pytest.fixture
def tt(tmp_path):
    p = tmp_path / "timetable.txt"
    p.write_text("AB1234 lecture Mon\nCD5678 tutorial Tue\n")
    return p


def refresh(tt, tmp_path, llm):
    out = tmp_path / "module_registry.json"
    reg, save_to = load_or_refresh_registry(tt, llm, out)
    if save_to:
        save_registry(reg, save_to)
    return reg, save_to, out


def test_first_run_builds_and_saves_the_list_with_a_fingerprint(tt, tmp_path):
    llm = ListLLM()
    reg, save_to, out = refresh(tt, tmp_path, llm)
    assert codes(reg) == ["AB1234", "CD5678"] and save_to == str(out) and llm.calls == 1
    saved = json.loads(out.read_text())
    assert saved["source_hash"] and saved["source_path"].endswith("timetable.txt")


def test_unchanged_timetable_reuses_the_saved_list_without_a_model_call(tt, tmp_path):
    refresh(tt, tmp_path, ListLLM())
    llm = ListLLM()
    reg, save_to, _ = refresh(tt, tmp_path, llm)
    assert codes(reg) == ["AB1234", "CD5678"] and save_to is None and llm.calls == 0


def test_changed_timetable_rebuilds_instead_of_using_the_old_list(tt, tmp_path):
    refresh(tt, tmp_path, ListLLM())
    tt.write_text("AB1234 lecture Mon\nEF9012 lab Wed\n")  # the student changed their timetable
    llm = ListLLM()
    reg, save_to, out = refresh(tt, tmp_path, llm)
    assert codes(reg) == ["AB1234", "EF9012"] and llm.calls == 1
    assert [m["code"] for m in json.loads(out.read_text())["modules"]] == ["AB1234", "EF9012"]


def test_giving_the_saved_list_still_checks_the_timetable_it_came_from(tt, tmp_path):
    _, _, out = refresh(tt, tmp_path, ListLLM())
    llm = ListLLM()
    reg, save_to = load_or_refresh_registry(out, llm, out)
    assert codes(reg) == ["AB1234", "CD5678"] and save_to is None and llm.calls == 0   # unchanged, used as saved
    tt.write_text("ZZ9999 new module\n")
    reg, save_to = load_or_refresh_registry(out, llm, out)
    assert codes(reg) == ["ZZ9999"] and save_to == str(out) and "changed" in reg["method"]  # changed, rebuilt


def test_a_saved_list_with_no_record_of_its_source_is_used_as_it_is(tmp_path):
    out = tmp_path / "hand_made.json"
    out.write_text(json.dumps({"modules": [{"code": "AB1234", "name": ""}]}))
    llm = ListLLM()
    reg, save_to = load_or_refresh_registry(out, llm, out)
    assert codes(reg) == ["AB1234"] and save_to is None and llm.calls == 0


def test_an_unreadable_saved_list_is_just_rebuilt(tt, tmp_path):
    out = tmp_path / "module_registry.json"
    out.write_text("not json at all")
    llm = ListLLM()
    reg, save_to = load_or_refresh_registry(tt, llm, out)
    assert codes(reg) == ["AB1234", "CD5678"] and save_to == str(out)


def test_a_missing_timetable_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_or_refresh_registry(tmp_path / "nope.ics", ListLLM(), tmp_path / "r.json")