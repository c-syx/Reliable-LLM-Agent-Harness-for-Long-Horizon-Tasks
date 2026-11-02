"""Evidence gate for the improved agent.

Moving a file into a module folder now needs evidence: a short exact phrase copied from the
file that shows which module it belongs to. The code checks the phrase itself, with no model
involved:
- the phrase must really appear in the file's text, so an invented quote is refused
- if the phrase names a module code, it must be the code of the folder the file is going to
- a phrase with no code must still be long enough to say something

A refused move is sent back as an error and the agent can try again. After two refusals for the
same move it goes through and is logged as unverified, so a bad quote can never trap a run.
Moves to Unsorted, moves of images and other files with no readable text, and folder moves are
not gated.

This checks that the stated reason is true. It cannot tell whether the reason is a good one.
"""
from __future__ import annotations

import copy
import re
from collections import Counter

from .readers import extract_text

CODE = re.compile(r"\b[A-Za-z]{2,4}\d{4}[A-Za-z]{0,2}\b")
NO_TEXT = ("(no readable text", "(no extractable text", "NOTE:", "(no ")
READ_CHARS = 30000
MIN_CHARS = 12   # a phrase with no module code must have at least this many letters and digits
MAX_CHARS = 300  # longer than this is a pasted page, not a phrase
MAX_BLOCKS = 2

EVIDENCE_PARAM = {"type": "string", "description": (
    "A short exact phrase copied from the file's own text (about 4 to 30 words) that shows which module it "
    "belongs to, for example the line that contains the module code or the course title. Required when the "
    "destination is a module folder. Leave it out only for Organised/Unsorted or for images with no readable text.")}


def evidence_specs(specs: list[dict]) -> list[dict]:
    """A copy of the tool list in which move_file also takes an evidence argument."""
    out = copy.deepcopy(specs)
    for t in out:
        if t["function"]["name"] == "move_file":
            t["function"]["parameters"]["properties"]["evidence"] = EVIDENCE_PARAM
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


class EvidenceCheck:
    def __init__(self, sandbox, max_blocks: int = MAX_BLOCKS) -> None:
        self.sb = sandbox
        self.max_blocks = max_blocks
        self.blocks: Counter = Counter()
        self.last = "skipped"  # how the most recent move was judged
        self.evidence_log: list[dict] = []

    def check(self, name: str, args: dict) -> str | None:
        self.last = "skipped"
        if name != "move_file" or not isinstance(args, dict):
            return None
        src, dst = args.get("src"), args.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            return None
        parts = dst.strip().replace("\\", "/").strip("/").split("/")
        if len(parts) < 3 or parts[0] != "Organised" or parts[1].lower() == "unsorted":
            return None
        module, src = parts[1], src.strip().replace("\\", "/").strip("/")
        try:
            p = self.sb._resolve(src)
            text = extract_text(p, READ_CHARS) if p.is_file() else ""
        except Exception:
            return None
        if not text or text.startswith(NO_TEXT):
            self.last = "no text"
            return None

        quote = args.get("evidence")
        quote = quote.strip() if isinstance(quote, str) else ""
        problem = self._problem(quote, module, _norm(text))
        if problem is None:
            codes = {c.upper() for c in CODE.findall(quote)}
            self.last = "stated code" if module.upper() in codes else "quote"
            self._log(src, module, quote, self.last)
            return None
        key = (src, module)
        self.blocks[key] += 1
        if self.blocks[key] > self.max_blocks:
            self.last = "unverified"
            self._log(src, module, quote, "unverified")
            return None
        return (f"ERROR: evidence check. {problem} Read the file, then move it again with an evidence "
                f"argument: an exact phrase copied from the file that shows it belongs to a module.")

    def _problem(self, quote: str, module: str, text_norm: str) -> str | None:
        if not quote:
            return "No evidence was given for this move."
        if len(quote) > MAX_CHARS:
            return "The evidence is too long. Copy one short phrase, not a whole passage."
        q = _norm(quote)
        if q not in text_norm:
            return "That phrase does not appear in the file."
        codes = {c.upper() for c in CODE.findall(quote)}
        if codes and module.upper() not in codes:
            return (f"The phrase names {', '.join(sorted(codes))} but the file is going to {module}.")
        if not codes and len(q.replace(" ", "")) < MIN_CHARS:
            return "The phrase is too short to show which module the file belongs to."
        return None

    def _log(self, src: str, module: str, quote: str, status: str) -> None:
        self.evidence_log.append({"file": src, "module": module, "evidence": quote, "status": status})