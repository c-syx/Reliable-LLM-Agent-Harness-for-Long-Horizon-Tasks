from harness.improved import ImprovedAgent
from harness.agent import BaselineAgent
from harness.sandbox import Sandbox
from harness.sibling_check import SiblingCheck

STREAMS = ("Professor Aaron Tan lecture on streams, lambda expressions and functional programming in Java. "
           "Streams, lambdas and Optional are covered by Aaron Tan. ")
THERMO = "Heat engines, entropy and the second law of thermodynamics. Carnot cycle and enthalpy. "


def make(tmp_path, sorted_files, incoming):
    """sorted_files: {Organised-relative path: text}. incoming: {Downloads-relative path: text}."""
    for rel, text in sorted_files.items():
        p = tmp_path / "Organised" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    for rel, text in incoming.items():
        p = tmp_path / "Downloads" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    return Sandbox(tmp_path), tmp_path


def base(tmp_path, incoming):
    return make(tmp_path, {
        "CS2030DE/l1.txt": STREAMS + "Lecture one on generics.",
        "CS2030DE/l2.txt": STREAMS + "Lecture two on immutability and records.",
        "CS2030DE/l3.txt": STREAMS + "Lecture three on monads and functors.",
        "ME2121/t1.txt": THERMO + "Tutorial one on boilers.",
        "ME2121/t2.txt": THERMO + "Tutorial two on refrigeration.",
        "Unsorted/x.txt": "random shopping list eggs milk",
    }, incoming)


def move(src, module, name):
    return "move_file", {"src": f"Downloads/{name}", "dst": f"Organised/{module}/{name}"}


def test_flags_a_file_that_looks_like_another_module(tmp_path):
    sb, _ = base(tmp_path, {"L07 Notes.txt": "Aaron Tan explains streams and lambdas with new examples."})
    sc = SiblingCheck(sb)
    msg = sc.check(*move(None, "ME2121", "L07 Notes.txt"))
    assert msg and msg.startswith("ERROR: sibling check") and "CS2030DE" in msg and "aaron" in msg
    assert sc.sibling_log[0]["suggested"] == "CS2030DE"


def test_repeating_the_same_move_goes_through(tmp_path):
    sb, _ = base(tmp_path, {"n.txt": "Aaron Tan explains streams and lambdas."})
    sc = SiblingCheck(sb)
    assert sc.check(*move(None, "ME2121", "n.txt"))
    assert sc.check(*move(None, "ME2121", "n.txt")) is None
    sc.record(*move(None, "ME2121", "n.txt"), "OK: moved")
    assert sc.sibling_log[0]["outcome"] == "kept own choice"


def test_following_the_suggestion_is_recorded(tmp_path):
    sb, _ = base(tmp_path, {"n.txt": "Aaron Tan explains streams and lambdas."})
    sc = SiblingCheck(sb)
    sc.check(*move(None, "ME2121", "n.txt"))
    assert sc.check(*move(None, "CS2030DE", "n.txt")) is None
    sc.record(*move(None, "CS2030DE", "n.txt"), "OK: moved")
    assert sc.sibling_log[0]["outcome"] == "moved to suggested"


def test_a_file_that_fits_its_own_module_is_not_flagged(tmp_path):
    sb, _ = base(tmp_path, {"h.txt": THERMO + "Homework on the Carnot cycle."})
    assert SiblingCheck(sb).check(*move(None, "ME2121", "h.txt")) is None


def test_unrelated_file_is_not_flagged(tmp_path):
    sb, _ = base(tmp_path, {"cv.txt": "Curriculum vitae of a student, hobbies swimming and travelling."})
    assert SiblingCheck(sb).check(*move(None, "Unsorted", "cv.txt")) is None


def test_other_module_needs_enough_files(tmp_path):
    sb, _ = make(tmp_path, {"CS2030DE/l1.txt": STREAMS, "ME2121/t1.txt": THERMO, "ME2121/t2.txt": THERMO},
                 {"n.txt": "Aaron Tan explains streams and lambdas."})
    assert SiblingCheck(sb).check(*move(None, "ME2121", "n.txt")) is None  # only one CS2030DE file


def test_flags_a_file_sent_to_unsorted(tmp_path):
    sb, _ = base(tmp_path, {"n.txt": "Aaron Tan explains streams and lambdas."})
    assert SiblingCheck(sb).check(*move(None, "Unsorted", "n.txt"))


def test_unreadable_file_is_left_alone(tmp_path):
    sb, _ = base(tmp_path, {})
    (tmp_path / "Downloads").mkdir(exist_ok=True)
    (tmp_path / "Downloads" / "pic.png").write_bytes(b"\x89PNG\r\n")
    assert SiblingCheck(sb).check("move_file", {"src": "Downloads/pic.png", "dst": "Organised/ME2121/pic.png"}) is None


def test_ignores_other_tools_and_bad_arguments(tmp_path):
    sb, _ = base(tmp_path, {})
    sc = SiblingCheck(sb)
    assert sc.check("read_file", {"path": "x"}) is None
    assert sc.check("move_file", {"src": 3, "dst": None}) is None
    assert sc.check("move_file", {"src": "Downloads/missing.txt", "dst": "Organised/ME2121/missing.txt"}) is None
    assert sc.check("move_file", {"src": "Downloads/a", "dst": "Organised/organise_log.csv"}) is None


def test_folder_is_judged_as_a_whole(tmp_path):
    sb, _ = base(tmp_path, {"Lab/a.txt": "Aaron Tan streams lambdas lab sheet.", "Lab/b.txt": "More streams and lambdas by Aaron Tan."})
    assert SiblingCheck(sb).check("move_folder", {"src": "Downloads/Lab", "dst": "Organised/ME2121/Lab"})


def test_only_the_improved_agent_has_it(tmp_path):
    sb, _ = base(tmp_path, {"n.txt": "Aaron Tan explains streams and lambdas."})
    assert BaselineAgent(None).before_call(*move(None, "ME2121", "n.txt")) is None
    a = ImprovedAgent(None)
    a.folder_check = __import__("harness.folder_check", fromlist=["FolderCheck"]).FolderCheck()
    a.sibling = SiblingCheck(sb)
    assert a.before_call(*move(None, "ME2121", "n.txt")).startswith("ERROR: sibling check")