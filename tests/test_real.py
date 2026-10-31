import csv
import json
import re
import sys
from pathlib import Path, PurePosixPath

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.llm import LLMReply, ToolCall
from harness.readers import extract_text
from harness.real import load_labels, make_labels_template, run_organise
from harness.sandbox import Sandbox

docx = pytest.importorskip("docx")
pptx = pytest.importorskip("pptx")
openpyxl = pytest.importorskip("openpyxl")
pytest.importorskip("pypdf")


def make_pdf(path: Path, text: str) -> None:
    content = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(content)} >>\nstream\n{content}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out, offsets = "%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    path.write_bytes(out.encode("latin-1"))


@pytest.fixture
def folder(tmp_path):
    d = tmp_path / "test_files"
    (d / "sub folder").mkdir(parents=True)
    (d / "notes.txt").write_text("Lecture 3 of AB1234, inheritance and interfaces.\n")
    (d / "sub folder" / "Main.java").write_text("// AB1234 take home\nclass Main {}\n")
    make_pdf(d / "slides.pdf", "Welcome to CD5678 week one")
    document = docx.Document()
    document.add_paragraph("Lab report for EF9012")
    document.save(d / "report.docx")
    deck = pptx.Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[0])
    slide.shapes.title.text = "AB1234 Chapter 4"
    deck.save(d / "lecture.pptx")
    wb = openpyxl.Workbook()
    wb.active["A1"] = "CD5678 marks"
    wb.save(d / "marks.xlsx")
    nb = {"cells": [{"cell_type": "markdown", "source": ["# EF9012 Week 7 lab"]}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    (d / "lab.ipynb").write_text(json.dumps(nb))
    (d / "photo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    return d


TRUTH = {
    "notes.txt": "AB1234", "sub folder/Main.java": "AB1234", "slides.pdf": "CD5678", "report.docx": "EF9012",
    "lecture.pptx": "AB1234", "marks.xlsx": "CD5678", "lab.ipynb": "EF9012", "photo.png": "Unsorted",
}


class RegexLLM:
    """Stand in model for plumbing tests only. Finds a module code with a regex and files by it."""

    CODE = re.compile(r"\b[A-Z]{2}\d{4}\b")

    def __init__(self, files):
        self.files, self.gen = files, None

    def _plan(self):
        for f in self.files:
            out = yield ("read_file", {"path": f"Downloads/{f}"})
            m = self.CODE.search(out or "")
            folder = m.group(0) if m else "Unsorted"
            yield ("make_dir", {"path": f"Organised/{folder}"})
            yield ("move_file", {"src": f"Downloads/{f}", "dst": f"Organised/{folder}/{PurePosixPath(f).name}"})
        yield ("finish", {"summary": "done"})

    def chat(self, messages, tools):
        last = next((m["content"] for m in reversed(messages) if m["role"] == "tool"), None)
        try:
            if self.gen is None:
                self.gen = self._plan()
                name, args = next(self.gen)
            else:
                name, args = self.gen.send(last)
        except StopIteration:
            return LLMReply("done", [])
        return LLMReply("", [ToolCall(f"c{len(messages)}", name, json.dumps(args))])


# ---- readers ----------------------------------------------------------
@pytest.mark.parametrize("name,expect", [
    ("notes.txt", "AB1234"), ("sub folder/Main.java", "AB1234"), ("slides.pdf", "CD5678"), ("report.docx", "EF9012"),
    ("lecture.pptx", "AB1234"), ("marks.xlsx", "CD5678"), ("lab.ipynb", "EF9012"),
])
def test_readers_extract_text(folder, name, expect):
    assert expect in extract_text(folder / name)


def test_image_has_no_text_and_says_so(folder):
    assert "no readable text" in extract_text(folder / "photo.png")


def test_corrupt_file_does_not_crash(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"not really a pdf")
    assert extract_text(bad).startswith("NOTE")


def test_read_file_respects_max_chars(tmp_path):
    (tmp_path / "long.txt").write_text("x" * 5000)
    sb = Sandbox(tmp_path)
    assert len(sb.read_file("long.txt", 300)) < 400
    assert len(sb.read_file("long.txt", 99999)) < 6100  # clamped


# ---- real mode --------------------------------------------------------
def test_end_to_end_organises_into_new_folder_and_leaves_original_alone(folder, tmp_path):
    out = tmp_path / "result"
    before = sorted(p.name for p in folder.rglob("*"))
    summary = run_organise(folder, out, RegexLLM(sorted(TRUTH)), max_steps=100)
    assert sorted(p.name for p in folder.rglob("*")) == before  # original untouched
    assert summary["n_files"] == 8 and summary["left_behind"] == 0 and summary["missing"] == 0
    assert summary["status"] == "complete"
    assert (out / "Organised" / "AB1234" / "Main.java").is_file()
    assert (out / "Organised" / "CD5678" / "slides.pdf").is_file()
    assert (out / "Organised" / "Unsorted" / "photo.png").is_file()
    assert len(list(out.glob("moves_report_*.csv"))) == 1 and (out / "_run" / "transcript.json").is_file()
    assert summary["per_module"] == {"AB1234": 3, "CD5678": 2, "EF9012": 2, "Unsorted": 1}


def test_labels_scoring_and_template(folder, tmp_path):
    template = tmp_path / "labels.csv"
    assert make_labels_template(folder, template) == 8
    rows = list(csv.DictReader(open(template, newline="")))
    assert {r["file"] for r in rows} == set(TRUTH) and all(r["module"] == "" for r in rows)

    labels_csv = tmp_path / "answers.csv"
    wrong = dict(TRUTH, **{"marks.xlsx": "EF9012"})  # one deliberately wrong label
    with open(labels_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "module"])
        w.writerows(wrong.items())
    summary = run_organise(folder, tmp_path / "r2", RegexLLM(sorted(TRUTH)), 100, load_labels(labels_csv))
    s = summary["score"]
    assert s["n_labelled"] == 8 and s["n_correct"] == 7
    assert s["wrong"][0]["file"] == "marks.xlsx" and s["wrong"][0]["got"] == "CD5678"


def test_unmoved_files_are_reported_as_left_behind(folder, tmp_path):
    summary = run_organise(folder, tmp_path / "r3", RegexLLM(sorted(TRUTH)[:3]), 100)  # only handles 3 files
    assert summary["left_behind"] == 5
    assert len(list((tmp_path / "r3" / "_left_behind").rglob("*.*"))) == 5


def test_safety_checks_on_folders(folder, tmp_path):
    with pytest.raises(ValueError):
        run_organise(folder, folder / "inside", RegexLLM([]), 5)
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "x.txt").write_text("hi")
    with pytest.raises(FileExistsError):
        run_organise(folder, busy, RegexLLM([]), 5)
    with pytest.raises(FileNotFoundError):
        run_organise(tmp_path / "nope", tmp_path / "o", RegexLLM([]), 5)


def test_prompt_contains_no_hard_coded_modules():
    from harness.real import REAL_PROMPT
    for code in ["CS2030", "ESP3201", "EE2026", "MA1522", "ES2631", "PC1101", "GEC1044"]:
        assert code not in REAL_PROMPT


# ---- timetable and module list ----------------------------------------
from harness.real import REAL_PROMPT, build_prompt, extract_registry, save_registry  # noqa: E402


class ExtractLLM:
    """Plain question answerer, returns a fixed reply and remembers what it was asked."""

    def __init__(self, reply):
        self.reply, self.asked = reply, []

    def chat(self, messages, tools):
        self.asked.append((messages, tools))
        return LLMReply(self.reply, [])


@pytest.fixture
def timetable(tmp_path):
    p = tmp_path / "timetable.txt"
    p.write_text("AB1234 Programming Methodology  Lecture Mon 10:00\nCD5678 Signals and Systems  Tutorial Tue 14:00\n"
                 "AB1234 Programming Methodology  Lab Wed 09:00\nEF9012 Energy Systems  Lecture Thu 16:00\n")
    return p


def test_registry_from_model_is_checked_against_the_timetable(timetable):
    reply = '```json\n[{"code":"AB1234","name":"Programming Methodology"},{"code":"cd 5678","name":"Signals and Systems"},' \
            '{"code":"ZZ9999","name":"Invented"},{"code":"AB1234","name":"dup"}]\n```'
    llm = ExtractLLM(reply)
    reg = extract_registry(timetable, llm)
    assert [m["code"] for m in reg["modules"]] == ["AB1234", "CD5678"]  # invented one dropped, duplicate merged
    assert reg["method"] == "model"
    assert llm.asked[0][1] == []  # asked as a plain question, no tools sent
    assert "Energy Systems" in llm.asked[0][0][0]["content"]  # the timetable text reached the model


def test_registry_falls_back_to_code_shape_when_the_answer_is_unusable(timetable):
    reg = extract_registry(timetable, ExtractLLM("sorry I cannot do that"))
    assert [m["code"] for m in reg["modules"]] == ["AB1234", "CD5678", "EF9012"]
    assert "fallback" in reg["method"]


def test_registry_save_and_reuse_costs_no_model_call(timetable, tmp_path):
    reg = extract_registry(timetable, ExtractLLM('[{"code":"AB1234","name":"Prog"}]'))
    saved = tmp_path / "module_registry.json"
    save_registry(reg, saved)
    llm = ExtractLLM("should not be used")
    again = extract_registry(saved, llm)
    assert again["modules"] == reg["modules"] and again["method"] == "saved registry" and llm.asked == []


def test_image_timetable_gives_a_clear_error(tmp_path):
    img = tmp_path / "timetable.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
    with pytest.raises(ValueError, match="text based export"):
        extract_registry(img, ExtractLLM("[]"))
    with pytest.raises(FileNotFoundError):
        extract_registry(tmp_path / "missing.pdf", ExtractLLM("[]"))


def test_ics_timetable_is_readable(tmp_path):
    ics = tmp_path / "tt.ics"
    ics.write_text("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:AB1234 Lecture\nEND:VEVENT\nEND:VCALENDAR\n")
    reg = extract_registry(ics, ExtractLLM("not json"))
    assert [m["code"] for m in reg["modules"]] == ["AB1234"]


def test_prompt_with_modules_lists_only_what_was_given():
    assert build_prompt(None) == REAL_PROMPT and build_prompt([]) == REAL_PROMPT
    p = build_prompt([{"code": "AB1234", "name": "Programming"}, {"code": "CD5678", "name": ""}])
    assert "- AB1234 (Programming)" in p and "- CD5678\n" in p
    assert "EF9012" not in p
    assert "stated" in p and "inferred" in p  # the log must say how each module was chosen


def test_run_with_timetable_reports_unsorted_and_unlisted_folders(folder, tmp_path):
    seen = []

    class Spy(RegexLLM):
        def chat(self, messages, tools):
            seen.append(messages[1]["content"])
            return super().chat(messages, tools)

    mods = [{"code": "AB1234", "name": "Programming"}, {"code": "CD5678", "name": ""}]
    s = run_organise(folder, tmp_path / "t1", Spy(sorted(TRUTH)), 100, modules=mods)
    assert "- AB1234 (Programming)" in seen[0]  # the agent really was told about the timetable
    assert s["timetable_modules"] == ["AB1234", "CD5678"]
    assert s["unlisted_folders"] == ["EF9012"] and s["unsorted"] == 1
    plain = run_organise(folder, tmp_path / "t2", RegexLLM(sorted(TRUTH)), 100)
    assert plain["unlisted_folders"] == [] and plain["timetable_modules"] == []  # nothing to compare against


# ---- busy models ------------------------------------------------------
import harness.llm as llm_mod  # noqa: E402
from harness.llm import LLMUnavailable, OpenRouterLLM  # noqa: E402


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, json.dumps(body)

    def json(self):
        return self._body


def make_llm(monkeypatch, responses, retries=3):
    calls = []

    def fake_post(url, headers, json, timeout):
        calls.append(json)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)
    return OpenRouterLLM("m", api_key="k", max_retries=retries), calls


OVERLOADED = FakeResponse(200, {"error": {"message": "Service temporarily overloaded", "code": 503}})
OK = FakeResponse(200, {"choices": [{"message": {"content": "hi"}}], "usage": {}})


def test_busy_model_is_retried_then_succeeds(monkeypatch):
    llm, calls = make_llm(monkeypatch, [OVERLOADED, OVERLOADED, OK])
    assert llm.chat([{"role": "user", "content": "x"}], []).content == "hi" and len(calls) == 3
    assert "tools" not in calls[0]  # no empty tools list


def test_backoff_is_capped_at_60_seconds(monkeypatch):
    waits = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: waits.append(s))
    llm = OpenRouterLLM("m", api_key="k")
    for attempt in range(10):
        llm._backoff(attempt, "busy")
    assert waits[:6] == [3, 6, 12, 24, 48, 60] and max(waits) == 60


def test_model_that_stays_busy_raises_a_clear_error(monkeypatch):
    llm, calls = make_llm(monkeypatch, [OVERLOADED], retries=4)
    with pytest.raises(LLMUnavailable, match="Wait a few minutes"):
        llm.chat([], [])
    assert len(calls) == 4


def test_real_problems_in_a_200_body_are_not_retried(monkeypatch):
    llm, calls = make_llm(monkeypatch, [FakeResponse(200, {"error": {"message": "No auth credentials", "code": 401}})])
    with pytest.raises(RuntimeError, match="No auth credentials") as e:
        llm.chat([], [])
    assert not isinstance(e.value, LLMUnavailable) and len(calls) == 1


def test_run_keeps_partial_work_when_the_model_goes_away(folder, tmp_path):
    class Dies(RegexLLM):
        n = 0

        def chat(self, messages, tools):
            Dies.n += 1
            if Dies.n > 7:
                raise LLMUnavailable("busy")
            return super().chat(messages, tools)

    s = run_organise(folder, tmp_path / "partial", Dies(sorted(TRUTH)), 100)
    assert s["stop_reason"] == "llm_unavailable"
    assert s["status"] == "partial"
    assert 0 < s["left_behind"] < 8 and len(list((tmp_path / "partial").glob("moves_report_*.csv"))) == 1


def test_interrupted_run_writes_a_partial_report_and_re_raises(folder, tmp_path):
    class Interrupted(RegexLLM):
        n = 0

        def chat(self, messages, tools):
            Interrupted.n += 1
            if Interrupted.n > 5:  # the first move has completed, the second file is not handled yet
                raise KeyboardInterrupt
            return super().chat(messages, tools)

    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):  # the error is not swallowed
        run_organise(folder, out, Interrupted(sorted(TRUTH)), 100)

    reports = list(out.glob("moves_report_*.csv"))
    assert len(reports) == 1
    with open(reports[0], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 8 and {r["status"] for r in rows} == {"partial"}
    # only the move that really completed is reported outside Downloads, and it was exported
    placed = [r for r in rows if r["final_location"].startswith("Organised/")]
    assert len(placed) == 1 and (out / placed[0]["final_location"]).is_file()
    assert sum(r["final_location"].startswith("Downloads/") for r in rows) == 7


# ---- folders ----------------------------------------------------------
class ScriptLLM:
    """Plays a fixed list of tool calls, then stops. For plumbing tests only."""

    def __init__(self, calls):
        self.calls, self.i = list(calls), 0

    def chat(self, messages, tools):
        if self.i >= len(self.calls):
            return LLMReply("done", [])
        name, args = self.calls[self.i]
        self.i += 1
        return LLMReply("", [ToolCall(f"c{self.i}", name, json.dumps(args))])


def test_move_folder_keeps_inside_structure_and_records_each_file(tmp_path):
    (tmp_path / "Downloads" / "Lab 5" / "starter").mkdir(parents=True)
    (tmp_path / "Downloads" / "Lab 5" / "starter" / "a.java").write_text("class A {}")
    (tmp_path / "Downloads" / "Lab 5" / "notes.txt").write_text("notes")
    (tmp_path / "Organised" / "AB1234").mkdir(parents=True)
    sb = Sandbox(tmp_path)
    assert sb.move_folder("Downloads/Lab 5", "Organised/AB1234/Lab 5").startswith("OK")
    assert (tmp_path / "Organised/AB1234/Lab 5/starter/a.java").is_file()
    assert not (tmp_path / "Downloads" / "Lab 5").exists()
    assert sorted(e["dst"] for e in sb.events) == ["Organised/AB1234/Lab 5/notes.txt", "Organised/AB1234/Lab 5/starter/a.java"]


def test_move_folder_refuses_unsafe_moves(tmp_path):
    (tmp_path / "Downloads" / "x").mkdir(parents=True)
    (tmp_path / "Organised" / "M").mkdir(parents=True)
    (tmp_path / "Organised" / "M" / "x").mkdir()
    sb = Sandbox(tmp_path)
    assert sb.move_folder("Downloads/x", "Organised/M/x").startswith("ERROR")        # destination exists
    assert sb.move_folder("Downloads/x", "Downloads/x/inner").startswith("ERROR")    # into itself
    assert sb.move_folder("Organised", "Downloads/Org").startswith("ERROR")          # not the Organised folder
    assert sb.move_folder("Downloads/x", "Organised/Nope/x").startswith("ERROR")     # parent missing
    assert sb.move_folder("Downloads/none", "Organised/M/none").startswith("ERROR")  # source missing
    assert sb.move_folder("Downloads/x", "../x").startswith("ERROR")                 # escapes the workspace


def test_flattening_a_subfolder_is_reported(folder, tmp_path):
    summary = run_organise(folder, tmp_path / "flat", RegexLLM(sorted(TRUTH)), max_steps=100)
    st = summary["folder_structure"]
    assert st["checked"] == 8 and st["kept"] == 7
    assert st["changed"][0]["file"] == "sub folder/Main.java"
    assert st["changed"][0]["original_folder"] == "sub folder" and st["changed"][0]["ended_up_in"] == "Organised/AB1234"


def test_moving_the_whole_folder_keeps_the_structure(folder, tmp_path):
    calls = [("make_dir", {"path": "Organised/AB1234"}),
             ("move_folder", {"src": "Downloads/sub folder", "dst": "Organised/AB1234/sub folder"}), ("finish", {"summary": "x"})]
    summary = run_organise(folder, tmp_path / "kept", ScriptLLM(calls), max_steps=20, agent_type="improved")
    st = summary["folder_structure"]
    assert st["checked"] == 1 and st["kept"] == 1 and st["changed"] == []
    assert (tmp_path / "kept/Organised/AB1234/sub folder/Main.java").is_file()


def test_improved_agent_is_stopped_from_turning_a_subfolder_into_a_module_folder(folder, tmp_path):
    calls = [("make_dir", {"path": "Organised"}),
             ("move_folder", {"src": "Downloads/sub folder", "dst": "Organised/sub folder"}), ("finish", {"summary": "x"})]
    summary = run_organise(folder, tmp_path / "wrong", ScriptLLM(calls), max_steps=20, agent_type="improved")
    assert summary["tool_errors"] == 1 and summary["folder_structure"]["changed"] == []
    assert (tmp_path / "wrong/_left_behind/sub folder/Main.java").is_file()  # the bad move never happened


def test_renamed_file_after_a_name_clash_still_counts_as_kept():
    from harness.real import folder_kept
    assert folder_kept("Lab/a.txt", ["Organised/M/Lab/a_2.txt"]) == (True, "Lab")
    assert folder_kept("a.txt", ["Organised/M/a.txt"]) == (True, "")
    assert folder_kept("a.txt", ["Downloads/a.txt"]) is None


def test_only_the_improved_agent_gets_folder_tool_and_rules():
    from harness.tools import TOOL_SPECS, IMPROVED_TOOL_SPECS
    names = lambda specs: [t["function"]["name"] for t in specs]
    assert "move_folder" not in names(TOOL_SPECS) and "move_folder" in names(IMPROVED_TOOL_SPECS)
    mods = [{"code": "AB1234", "name": ""}]
    for baseline in (build_prompt(), build_prompt(mods)):
        assert "move_folder" not in baseline and "Never flatten" not in baseline
    for improved in (build_prompt(folders=True), build_prompt(mods, folders=True)):
        assert "move_folder" in improved and "Never flatten" in improved


def test_baseline_cannot_use_move_folder_even_if_the_model_asks(folder, tmp_path):
    calls = [("make_dir", {"path": "Organised/AB1234"}),
             ("move_folder", {"src": "Downloads/sub folder", "dst": "Organised/AB1234/sub folder"}), ("finish", {"summary": "x"})]
    summary = run_organise(folder, tmp_path / "base", ScriptLLM(calls), max_steps=20)  # default agent is the baseline
    assert summary["agent"] == "baseline" and summary["tool_errors"] == 1
    assert (tmp_path / "base/_left_behind/sub folder/Main.java").is_file()


def test_unknown_agent_name_is_rejected(folder, tmp_path):
    with pytest.raises(ValueError):
        run_organise(folder, tmp_path / "x", ScriptLLM([]), agent_type="nope")


# ---- combined improved agent ------------------------------------------
def test_improved_agent_verifies_a_move_and_keeps_checkpointing(folder, tmp_path):
    out = tmp_path / "combined"
    calls = [("make_dir", {"path": "Organised/AB1234"}),
             ("move_file", {"src": "Downloads/notes.txt", "dst": "Organised/AB1234/notes.txt"}),
             ("finish", {"summary": "done"})]
    summary = run_organise(folder, out, ScriptLLM(calls), max_steps=20, agent_type="improved")

    assert summary["agent"] == "improved" and summary["tool_errors"] == 0 and not summary["resumable"]
    transcript = json.loads((out / "_run" / "transcript.json").read_text())
    move_call = next(c for step in transcript["steps"] for c in step["tool_calls"] if c["name"] == "move_file")
    assert move_call["verification"]["ok"] is True
    assert "VERIFIED:" in move_call["result"]
    checkpoint = json.loads((out / "_run" / "checkpoint.json").read_text())
    assert checkpoint["complete"] is True


def test_improved_agent_verifies_a_whole_folder_move(folder, tmp_path):
    out = tmp_path / "combined-folder"
    calls = [("make_dir", {"path": "Organised/AB1234"}),
             ("move_folder", {"src": "Downloads/sub folder", "dst": "Organised/AB1234/sub folder"}),
             ("finish", {"summary": "done"})]
    summary = run_organise(folder, out, ScriptLLM(calls), max_steps=20, agent_type="improved")

    assert summary["tool_errors"] == 0 and summary["folder_structure"]["kept"] == 1
    transcript = json.loads((out / "_run" / "transcript.json").read_text())
    move_call = next(c for step in transcript["steps"] for c in step["tool_calls"] if c["name"] == "move_folder")
    assert move_call["verification"]["ok"] is True
    assert "1 files preserved" in move_call["verification"]["detail"]