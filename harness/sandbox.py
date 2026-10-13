"""Path confined workspace that the agent acts on.

Every method returns a plain string. Failures start with "ERROR:" so the agent
sees them as tool output instead of the run crashing.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from .readers import extract_text


class SandboxError(Exception):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


class Sandbox:
    def __init__(self, root: Path, truth: dict | None = None):
        # truth maps content hash to {"module": str, "expected": str} and is only
        # used to label move events for analysis. The agent never sees it.
        self.root = Path(root).resolve()
        self.truth = truth or {}
        self.events: list[dict] = []
        self.current_step = 0

    # ---- helpers -------------------------------------------------------
    def _resolve(self, rel: str) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise SandboxError("path must be a non empty string")
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise SandboxError(f"path escapes the workspace: {rel}")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    # ---- tools ---------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        try:
            p = self._resolve(path)
            if not p.is_dir():
                return f"ERROR: not a directory: {path}"
            rows = []
            for child in sorted(p.iterdir(), key=lambda c: (c.is_file(), c.name.lower())):
                rows.append(f"{child.name}/" if child.is_dir() else f"{child.name}  ({child.stat().st_size} bytes)")
            return "\n".join(rows) if rows else "(empty directory)"
        except (SandboxError, OSError) as e:
            return f"ERROR: {e}"

    def read_file(self, path: str, max_chars: int = 1500) -> str:
        try:
            p = self._resolve(path)
            if not p.is_file():
                return f"ERROR: not a file: {path}"
            try:
                max_chars = max(200, min(int(max_chars), 6000))
            except (TypeError, ValueError):
                max_chars = 1500
            return extract_text(p, max_chars)
        except (SandboxError, OSError) as e:
            return f"ERROR: {e}"

    def make_dir(self, path: str) -> str:
        try:
            p = self._resolve(path)
            p.mkdir(parents=True, exist_ok=True)
            return f"OK: directory ready at {self._rel(p)}"
        except (SandboxError, OSError) as e:
            return f"ERROR: {e}"

    def move_file(self, src: str, dst: str) -> str:
        try:
            s, d = self._resolve(src), self._resolve(dst)
            if not s.is_file():
                return f"ERROR: source file does not exist: {src}"
            if d.exists():
                return f"ERROR: destination already exists, nothing was moved: {dst}"
            if not d.parent.is_dir():
                return f"ERROR: destination folder does not exist: {self._rel(d.parent)}"
            h = sha(s.read_bytes())
            shutil.move(str(s), str(d))
            self._record_move(h, self._rel(s), self._rel(d))
            return f"OK: moved {self._rel(s)} to {self._rel(d)}"
        except (SandboxError, OSError) as e:
            return f"ERROR: {e}"

    def write_file(self, path: str, content: str) -> str:
        try:
            p = self._resolve(path)
            if not p.parent.is_dir():
                return f"ERROR: folder does not exist: {self._rel(p.parent)}"
            p.write_text(content)
            return f"OK: wrote {len(content)} chars to {self._rel(p)}"
        except (SandboxError, OSError) as e:
            return f"ERROR: {e}"

    # ---- analysis only -------------------------------------------------
    def _record_move(self, h: str, src: str, dst: str) -> None:
        t = self.truth.get(h)
        module_ok = None
        if t is not None and dst.startswith("Organised/"):
            module_ok = dst.split("/")[1] == t["module"]
        self.events.append({"step": self.current_step, "src": src, "dst": dst, "module_ok": module_ok})
