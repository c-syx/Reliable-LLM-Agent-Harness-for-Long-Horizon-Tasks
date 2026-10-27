import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.mock import OracleLLM
from harness.sandbox import Sandbox
from harness.tasks import generate_task
from harness.verified_agent import VerifiedAgent
from run_experiment import run_once


def test_verified_agent_passes_for_correct_move_sequence():
    spec = generate_task(11)
    result = run_once(VerifiedAgent, OracleLLM(spec), spec, max_steps=80)
    assert result["metrics"]["success"]
    assert result["metrics"]["file_accuracy"] == 1.0
    move_calls = [c for step in result["steps"] for c in step["tool_calls"] if c["name"] == "move_file"]
    assert move_calls
    assert all(c["verification"]["ok"] for c in move_calls)


def test_verifier_flags_move_tool_error():
    spec = generate_task(11)
    work = __import__("tempfile").mkdtemp()
    try:
        spec.materialise(Path(work))
        sb = Sandbox(Path(work), spec.truth())
        from harness.verifier import ActionVerifier
        v = ActionVerifier(sb)
        ok, detail = v.verify_move("Downloads/nope.txt", "Organised/X/nope.txt", "ERROR: source missing")
        assert not ok
        assert "tool reported failure" in detail
    finally:
        import shutil
        shutil.rmtree(work, ignore_errors=True)


def test_verified_agent_preserves_semantic_wrong_move_for_evaluator_to_score():
    # Execution verification checks whether the requested move happened; it does not
    # inspect the answer key and therefore cannot by itself prove the module was right.
    spec = generate_task(11)
    result = run_once(VerifiedAgent, OracleLLM(spec, wrong_move_index=2), spec, max_steps=80)
    assert not result["metrics"]["success"]
    assert result["metrics"]["n_wrong_moves"] == 1
    move_calls = [c for step in result["steps"] for c in step["tool_calls"] if c["name"] == "move_file"]
    assert all(c["verification"]["ok"] for c in move_calls)


def test_verifier_compares_destination_to_pre_move_hash(tmp_path):
    from harness.verifier import ActionVerifier

    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Organised" / "AB1234").mkdir(parents=True)
    source = tmp_path / "Downloads" / "lecture.txt"
    source.write_text("original contents")
    sb = Sandbox(tmp_path)
    verifier = ActionVerifier(sb)
    snapshot = verifier.before_move("Downloads/lecture.txt", "Organised/AB1234/lecture.txt")

    source.rename(tmp_path / "Organised" / "AB1234" / "lecture.txt")
    ok, detail = verifier.verify_move(
        "Downloads/lecture.txt", "Organised/AB1234/lecture.txt",
        "OK: moved", snapshot,
    )
    assert ok, detail


def test_verifier_rejects_destination_with_changed_content(tmp_path):
    from harness.verifier import ActionVerifier

    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Organised" / "AB1234").mkdir(parents=True)
    source = tmp_path / "Downloads" / "lecture.txt"
    source.write_text("original contents")
    sb = Sandbox(tmp_path)
    verifier = ActionVerifier(sb)
    snapshot = verifier.before_move("Downloads/lecture.txt", "Organised/AB1234/lecture.txt")

    destination = tmp_path / "Organised" / "AB1234" / "lecture.txt"
    destination.write_text("different contents")
    source.unlink()
    ok, detail = verifier.verify_move(
        "Downloads/lecture.txt", "Organised/AB1234/lecture.txt",
        "OK: moved", snapshot,
    )
    assert not ok
    assert "hash" in detail


def test_verifier_prevents_overwrite_before_move(tmp_path):
    from harness.verifier import ActionVerifier

    (tmp_path / "Downloads").mkdir()
    (tmp_path / "Organised" / "AB1234").mkdir(parents=True)
    (tmp_path / "Downloads" / "lecture.txt").write_text("source")
    (tmp_path / "Organised" / "AB1234" / "lecture.txt").write_text("keep me")
    verifier = ActionVerifier(Sandbox(tmp_path))

    import pytest
    with pytest.raises(FileExistsError):
        verifier.before_move("Downloads/lecture.txt", "Organised/AB1234/lecture.txt")
