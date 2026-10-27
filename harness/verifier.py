"""Deterministic verification for file-management tool actions.

Verification checks whether the requested filesystem operation occurred and whether
file contents were preserved. It intentionally does not use the benchmark truth or
answer key, so it cannot judge whether the LLM chose the semantically correct module.
"""
from __future__ import annotations

from pathlib import Path

from .sandbox import Sandbox, sha


class ActionVerifier:
    """Verify file moves and basic write/directory postconditions in a sandbox."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def before_move(self, src: str, dst: str) -> dict:
        """Capture source identity before a move; reject invalid preconditions.

        The snapshot is independent of the move tool's event log. The destination
        must not already exist, matching Sandbox.move_file's no-overwrite contract.
        Raises ValueError/FileNotFoundError/FileExistsError when preconditions fail.
        """
        source = self.sandbox._resolve(src)
        destination = self.sandbox._resolve(dst)

        if not source.is_file():
            raise FileNotFoundError(f"source file does not exist: {src}")
        if destination.exists():
            raise FileExistsError(f"destination already exists: {dst}")

        return {
            "src": source.relative_to(self.sandbox.root).as_posix(),
            "dst": destination.relative_to(self.sandbox.root).as_posix(),
            "source_hash": sha(source.read_bytes()),
            "source_size": source.stat().st_size,
        }

    def verify_move(
        self,
        src: str,
        dst: str,
        tool_result: str,
        before: dict | None = None,
    ) -> tuple[bool, str]:
        """Return (verified, explanation) after a move_file attempt.

        A pre-move snapshot is required to verify file identity. This avoids relying
        on the sandbox event log, which is written by the same executor being checked.
        """
        if tool_result.startswith("ERROR"):
            return False, f"tool reported failure: {tool_result}"
        if before is None:
            return False, "pre-move snapshot is missing; file identity cannot be verified"

        try:
            source = self.sandbox._resolve(src)
            destination = self.sandbox._resolve(dst)
        except Exception as exc:
            return False, f"path validation failed: {exc}"

        expected_src = source.relative_to(self.sandbox.root).as_posix()
        expected_dst = destination.relative_to(self.sandbox.root).as_posix()
        if before.get("src") != expected_src or before.get("dst") != expected_dst:
            return False, "the requested move paths differ from the pre-move snapshot"
        if source.exists():
            return False, f"source still exists after move: {src}"
        if not destination.is_file():
            return False, f"destination file is missing after move: {dst}"

        actual_hash = sha(destination.read_bytes())
        if actual_hash != before.get("source_hash"):
            return False, "destination content hash differs from the pre-move source hash"
        if destination.stat().st_size != before.get("source_size"):
            return False, "destination size differs from the pre-move source size"

        return True, f"verified: {src} -> {dst} (pre/post content hash {actual_hash})"

    def verify_file_exists(self, path: str, tool_result: str) -> tuple[bool, str]:
        """Verify that a requested write_file action left a file at its path."""
        if tool_result.startswith("ERROR"):
            return False, f"tool reported failure: {tool_result}"
        try:
            target = self.sandbox._resolve(path)
        except Exception as exc:
            return False, f"path validation failed: {exc}"
        if not target.is_file():
            return False, f"file missing after write: {path}"
        return True, f"verified: file exists at {path}"

    def verify_directory_exists(self, path: str, tool_result: str) -> tuple[bool, str]:
        """Verify that a requested make_dir action left a directory at its path."""
        if tool_result.startswith("ERROR"):
            return False, f"tool reported failure: {tool_result}"
        try:
            target = self.sandbox._resolve(path)
        except Exception as exc:
            return False, f"path validation failed: {exc}"
        if not target.is_dir():
            return False, f"directory missing after creation: {path}"
        return True, f"verified: directory exists at {path}"
