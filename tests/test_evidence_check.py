from harness.agent import BaselineAgent
from harness.evidence_check import EvidenceCheck, evidence_specs
from harness.improved import ImprovedAgent
from harness.sandbox import Sandbox
from harness.sibling_check import SiblingCheck
from harness.tools import IMPROVED_TOOL_SPECS, TOOL_SPECS

TEXT = ("CS2030DE Programming Methodology II\nLecture 7 notes on streams and lambda expressions in Java.\n"
        "Taught by Aaron Tan.")


def setup(tmp_path, files=None):
    for name, text in (files or {"n.txt": TEXT}).items():
        p = tmp_path / "Downloads" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return Sandbox(tmp_path)


def mv(module, evidence=None, name="n.txt"):
    a = {"src": f"Downloads/{name}", "dst": f"Organised/{module}/{name}"}
    if evidence is not None:
        a["evidence"] = evidence
    return "move_file", a


def test_missing_evidence_is_refused(tmp_path):
    msg = EvidenceCheck(setup(tmp_path)).check(*mv("CS2030DE"))
    assert msg.startswith("ERROR: evidence check") and "No evidence" in msg


def test_a_quote_that_is_not_in_the_file_is_refused(tmp_path):
    msg = EvidenceCheck(setup(tmp_path)).check(*mv("CS2030DE", "CS2030DE Object Oriented Design"))
    assert "does not appear" in msg


def test_a_real_quote_with_the_right_code_passes(tmp_path):
    ev = EvidenceCheck(setup(tmp_path))
    assert ev.check(*mv("CS2030DE", "cs2030de programming methodology ii")) is None  # case and spacing do not matter
    assert ev.last == "stated code" and ev.evidence_log[0]["status"] == "stated code"


def test_a_quote_naming_another_code_is_refused(tmp_path):
    msg = EvidenceCheck(setup(tmp_path)).check(*mv("ES1103", "CS2030DE Programming Methodology II"))
    assert "names CS2030DE" in msg and "going to ES1103" in msg


def test_a_quote_without_a_code_needs_enough_words(tmp_path):
    ev = EvidenceCheck(setup(tmp_path))
    assert ev.check(*mv("CS2030DE", "streams and lambda expressions")) is None and ev.last == "quote"
    assert "too short" in EvidenceCheck(setup(tmp_path)).check(*mv("CS2030DE", "Java"))


def test_a_whole_page_is_not_evidence(tmp_path):
    assert "too long" in EvidenceCheck(setup(tmp_path, {"n.txt": "word " * 200})).check(*mv("M", "word " * 100))


def test_it_lets_the_move_through_after_two_refusals(tmp_path):
    ev = EvidenceCheck(setup(tmp_path))
    assert ev.check(*mv("CS2030DE", "made up phrase here")) and ev.check(*mv("CS2030DE", "made up phrase here"))
    assert ev.check(*mv("CS2030DE", "made up phrase here")) is None
    assert ev.last == "unverified" and ev.evidence_log[-1]["status"] == "unverified"


def test_unsorted_images_and_folders_are_not_gated(tmp_path):
    sb = setup(tmp_path)
    (tmp_path / "Downloads" / "pic.png").write_bytes(b"\x89PNG")
    ev = EvidenceCheck(sb)
    assert ev.check(*mv("Unsorted")) is None
    assert ev.check(*mv("ME2121", name="pic.png")) is None and ev.last == "no text"
    assert ev.check("move_folder", {"src": "Downloads/L", "dst": "Organised/M/L"}) is None
    assert ev.check("read_file", {"path": "x"}) is None


def test_only_the_improved_agent_asks_for_evidence():
    def props(specs):
        return next(t for t in specs if t["function"]["name"] == "move_file")["function"]["parameters"]["properties"]
    assert "evidence" not in props(TOOL_SPECS) and "evidence" not in props(IMPROVED_TOOL_SPECS)
    assert "evidence" in props(ImprovedAgent.tool_specs) and "evidence" not in props(BaselineAgent.tool_specs)
    assert "evidence" in props(evidence_specs(TOOL_SPECS))
    assert next(t for t in ImprovedAgent.tool_specs if t["function"]["name"] == "move_folder")  # nothing was lost


def test_a_file_that_states_its_code_skips_the_sibling_check(tmp_path, monkeypatch):
    # the parent class's own checks need a run in progress, so stand them aside: only our two checks are tested here
    monkeypatch.setattr(ImprovedAgent.__mro__[1], "before_call", lambda self, name, args: None)
    sb = setup(tmp_path)
    (tmp_path / "Downloads" / "m.txt").write_text("ME2121 note. Aaron Tan streams lambdas.")
    for i in range(3):
        d = tmp_path / "Organised" / "CS2030DE"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"l{i}.txt").write_text("Aaron Tan lecture on streams and lambdas in Java " + str(i))
    a = ImprovedAgent(None)
    a.folder_check = __import__("harness.folder_check", fromlist=["FolderCheck"]).FolderCheck()
    a.evidence, a.sibling = EvidenceCheck(sb), SiblingCheck(sb)
    # states ME2121 and goes to ME2121: the sibling check would have objected, but the stated code wins
    assert a.before_call(*mv("ME2121", "ME2121 note", "m.txt")) is None
    # a quote with no code does not get that protection
    assert a.before_call(*mv("ME2121", "Aaron Tan streams lambdas", "m.txt")).startswith("ERROR: sibling check")