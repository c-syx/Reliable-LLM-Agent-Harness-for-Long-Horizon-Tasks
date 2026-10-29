"""Folder structure check for the improved agent.

A file or folder must keep its original subfolder path inside its module folder. For example
Downloads/Lab 5/starter/a.java must go to Organised/<module>/Lab 5/starter/a.java. The check runs
before a move, so a bad move never happens and nothing has to be undone. The agent is told the
path it should have used, and can retry in the same run.

It does not use the answer key and cannot tell whether the module is right. It only compares the
destination with where the file came from. The file name may change, for a name clash, but the
folder path may not.
"""
from __future__ import annotations

import posixpath

ROOT = "Downloads"


def _norm(path) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return None
    p = posixpath.normpath(path.strip().replace("\\", "/")).lstrip("/")
    return None if p.startswith("..") or p == "." else p


def _parent(path: str) -> str:
    return posixpath.dirname(path)


class FolderCheck:
    def __init__(self) -> None:
        self.origin: dict[str, str] = {}  # current path -> original path inside Downloads

    # ---- where did this come from ------------------------------------
    def original_of(self, path: str) -> str | None:
        """Original path relative to Downloads, or None when it is not known."""
        if path in self.origin:
            return self.origin[path]
        parts = path.split("/")
        for i in range(len(parts) - 1, 0, -1):  # longest known ancestor first
            anc = "/".join(parts[:i])
            if anc in self.origin:
                return posixpath.join(self.origin[anc], *parts[i:])
        if parts[0] == ROOT and len(parts) > 1:
            return "/".join(parts[1:])
        return None

    # ---- the check ----------------------------------------------------
    def check(self, name: str, args: dict) -> str | None:
        """Return an ERROR message if this move breaks the folder structure, otherwise None."""
        if name not in ("move_file", "move_folder") or not isinstance(args, dict):
            return None
        src, dst = _norm(args.get("src")), _norm(args.get("dst"))
        if src is None or dst is None:
            return None
        orig = self.original_of(src)
        parts = dst.split("/")
        if orig is None or parts[0] != "Organised":
            return None  # not something we can judge, or not a move into the organised folder

        if name == "move_folder":
            if len(parts) == 2:
                return (f"ERROR: folder check failed. The folder {orig} must not become a module folder. "
                        f"Put it inside a module folder, for example Organised/<module>/{orig}.")
            if len(parts) < 3:
                return None
            if "/".join(parts[2:]) != orig:
                return (f"ERROR: folder check failed. The folder {orig} must keep its name and its place. "
                        f"Move it to Organised/{parts[1]}/{orig} instead.")
            return None

        if len(parts) < 3:
            return None  # a file placed straight into Organised has no module folder to judge
        want = _parent(orig)
        if "/".join(parts[2:-1]) != want:
            right = "/".join(["Organised", parts[1]] + ([want] if want else []) + [parts[-1]])
            where = f'"{want}"' if want else "the top level"
            return (f"ERROR: folder check failed. This file was in {where} and must keep that folder path "
                    f"inside its module folder. Create the folder if needed, then move it to {right} instead.")
        return None

    def record(self, name: str, args: dict, result: str) -> None:
        """Remember where things came from after a successful move."""
        if name not in ("move_file", "move_folder") or not result.startswith("OK"):
            return
        src, dst = _norm(args.get("src")), _norm(args.get("dst"))
        if src is None or dst is None:
            return
        orig = self.original_of(src)
        if orig is not None:
            self.origin[dst] = orig