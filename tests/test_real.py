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
    assert (out / "Organised" / "AB1234" / "Main.java").is_file()
    assert (out / "Organised" / "CD5678" / "slides.pdf").is_file()
    assert (out / "Organised" / "Unsorted" / "photo.png").is_file()
    assert (out / "moves_report.csv").is_file() and (out / "_run" / "transcript.json").is_file()
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
