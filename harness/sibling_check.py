"""Sibling check for the improved agent.

Before a file is moved into a module folder, compare it with the files already sorted into the
other module folders. If it looks much more like the files of module B than the files of the
module the agent chose, the move is sent back once so the agent re-reads the file.

Only the text of the files is used. Distinctive words that two documents share, such as a
lecturer's name or a course title, count the most, because common words appear everywhere and
get a low weight. Nothing here knows any module code, name or topic, and the answer key is never
used. It costs no model requests.

It cannot be sure, so it never forces a move. The agent may repeat the same move and it will go
through. Each flag and what the agent did afterwards is kept in sibling_log for the report.

A weakness to keep in mind: it learns from the files already sorted, so if the first few files
in a folder were put in the wrong place it can agree with the mistake instead of fixing it.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from .readers import extract_text

WORD = re.compile(r"[A-Za-z][A-Za-z0-9']{2,}")
STOP = set("""the and for are but not you all any can had her was one our out has have his how its may new now old see
two way who did get use that this with from they will would there their what when which while about after again
also been before being between both could does done each even every first here into just like make many more most
much must only other over same should some such than then these those through under very were where your page
slide slides lecture notes week figure table example examples using used chapter section university national
singapore nus semester student students""".split())
SKIP_TEXT = ("(no readable text", "(no extractable text", "NOTE:", "(no ")

# Starting values. Tune them on a development split of the files, not on the files you report on.
MIN_SIM = 0.10       # the other module must be at least this similar
RATIO = 1.5          # and this many times more similar than the chosen module
MIN_GAP = 0.05       # and ahead by at least this much
MIN_SIBLINGS = 2     # the other module needs at least this many files to be a fair comparison
MIN_EVIDENCE = 2     # distinctive words the file shares with the other module and not the chosen one
READ_CHARS = 8000


def _tokens(text: str) -> Counter:
    return Counter(w for w in (m.lower() for m in WORD.findall(text)) if w not in STOP)


class SiblingCheck:
    def __init__(self, sandbox, min_sim=MIN_SIM, ratio=RATIO, min_gap=MIN_GAP,
                 min_siblings=MIN_SIBLINGS, min_evidence=MIN_EVIDENCE) -> None:
        self.sb = sandbox
        self.min_sim, self.ratio, self.min_gap = min_sim, ratio, min_gap
        self.min_siblings, self.min_evidence = min_siblings, min_evidence
        self.cache: dict[str, Counter] = {}  # content hash -> word counts, so each file is read once
        self.challenged: set[tuple[str, str]] = set()  # (source path, module) already sent back once
        self.sibling_log: list[dict] = []

    # ---- reading ---------------------------------------------------------
    def _count(self, path) -> Counter:
        try:
            key = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return Counter()
        if key not in self.cache:
            text = extract_text(path, READ_CHARS)
            self.cache[key] = Counter() if text.startswith(SKIP_TEXT) else _tokens(text)
        return self.cache[key]

    def _candidate(self, rel: str) -> Counter | None:
        try:
            p = self.sb._resolve(rel)
        except Exception:
            return None
        if p.is_file():
            c = self._count(p)
        elif p.is_dir():
            c = Counter()
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    c.update(self._count(f))
        else:
            return None
        return c or None

    def _sorted_docs(self) -> dict[str, list[Counter]]:
        """Word counts of every readable file now sitting in a module folder."""
        docs: dict[str, list[Counter]] = {}
        org = self.sb.root / "Organised"
        if not org.is_dir():
            return docs
        for mod in sorted(org.iterdir()):
            if not mod.is_dir() or mod.name.lower() == "unsorted":
                continue
            for f in sorted(mod.rglob("*")):
                if f.is_file():
                    c = self._count(f)
                    if c:
                        docs.setdefault(mod.name, []).append(c)
        return docs

    # ---- scoring ---------------------------------------------------------
    @staticmethod
    def _vec(c: Counter, idf: dict[str, float]) -> dict[str, float]:
        v = {w: (1 + math.log(n)) * idf.get(w, 1.0) for w, n in c.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    @staticmethod
    def _cos(a: dict, b: dict) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b.get(w, 0.0) for w, x in a.items())

    def _module_sim(self, cand: dict, docs: list[Counter], idf: dict) -> float:
        sims = sorted((self._cos(cand, self._vec(d, idf)) for d in docs), reverse=True)[:2]
        return sum(sims) / len(sims) if sims else 0.0

    def assess(self, cand_counts: Counter, chosen: str, docs: dict[str, list[Counter]]) -> dict | None:
        """Return the best competing module and the evidence, or None when the choice looks fine."""
        everyone = [d for ds in docs.values() for d in ds] + [cand_counts]
        df = Counter(w for d in everyone for w in d)
        n = len(everyone)
        idf = {w: math.log((n + 1) / (k + 1)) + 1.0 for w, k in df.items()}
        cand = self._vec(cand_counts, idf)
        own_docs = docs.get(chosen, [])
        own_sim = self._module_sim(cand, own_docs, idf)
        own_words = {w for d in own_docs for w in d}
        best = None
        for mod, ds in docs.items():
            if mod == chosen or len(ds) < self.min_siblings:
                continue
            sim = self._module_sim(cand, ds, idf)
            if sim < self.min_sim or sim < self.ratio * own_sim or sim - own_sim < self.min_gap:
                continue
            in_two = Counter(w for d in ds for w in d if w in cand_counts)
            evidence = sorted((w for w, k in in_two.items() if k >= 2 and w not in own_words),
                              key=lambda w: (-idf[w], w))
            if len(evidence) < self.min_evidence:
                continue
            if best is None or sim > best["sim"]:
                best = {"module": mod, "sim": round(sim, 3), "own_sim": round(own_sim, 3),
                        "evidence": evidence[:5], "siblings": len(ds)}
        return best

    # ---- hooks -------------------------------------------------------------
    @staticmethod
    def _target(args: dict):
        dst = args.get("dst") if isinstance(args, dict) else None
        src = args.get("src") if isinstance(args, dict) else None
        if not isinstance(dst, str) or not isinstance(src, str):
            return None
        parts = dst.strip().replace("\\", "/").strip("/").split("/")
        if len(parts) < 2 or parts[0] != "Organised":
            return None
        if len(parts) == 2:
            return None  # a file straight into Organised has no module folder to judge
        return src.strip().replace("\\", "/").strip("/"), parts[1]

    def check(self, name: str, args: dict) -> str | None:
        """Return an ERROR message to send the move back, or None to let it through."""
        if name not in ("move_file", "move_folder"):
            return None
        t = self._target(args)
        if t is None:
            return None
        src, chosen = t
        if (src, chosen) in self.challenged:
            return None  # already questioned once, the agent has had its second look
        cand = self._candidate(src)
        if not cand:
            return None
        found = self.assess(cand, chosen, self._sorted_docs())
        if found is None:
            return None
        self.challenged.add((src, chosen))
        self.sibling_log.append({"file": src, "chosen": chosen, "suggested": found["module"],
                                 "sim_suggested": found["sim"], "sim_chosen": found["own_sim"],
                                 "evidence": found["evidence"], "outcome": "pending"})
        words = ", ".join(f"'{w}'" for w in found["evidence"])
        return (f"ERROR: sibling check. {src} looks more like the {found['siblings']} files already in "
                f"Organised/{found['module']} than the ones in Organised/{chosen}. It shares {words} with "
                f"Organised/{found['module']}, and none of those words appear in Organised/{chosen}. "
                f"Read the file again. If it belongs to {found['module']}, move it to Organised/{found['module']} "
                f"instead. If you are sure about {chosen}, repeat the same move and it will go through.")

    def record(self, name: str, args: dict, result: str) -> None:
        """After a successful move, note whether a flagged file went where the check suggested."""
        if name not in ("move_file", "move_folder") or not result.startswith("OK"):
            return
        t = self._target(args)
        if t is None:
            return
        src, module = t
        for e in self.sibling_log:
            if e["file"] == src and e["outcome"] == "pending":
                e["outcome"] = "moved to suggested" if module == e["suggested"] else "kept own choice"