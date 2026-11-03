"""Compare a moves report with your answer key and list every file that went to the wrong folder.

This is a separate tool for you. The agent never sees the answer key.

Usage
    python check_moves.py results/cohere_north_mini_code answer.csv
    python check_moves.py results/cohere_north_mini_code/moves_report_20261026_12.10pm.csv answer.csv

The first argument can be a run folder (the newest moves_report_*.csv in it is used) or a report file.
The answer key has the columns file,module. Rows with a blank module are skipped.
A folder counts as one item in the totals; files inside it are not counted separately.
An optional third column, also_accept, may hold one extra module code that is also acceptable
for that row. A blank value means none. It does not change the strict result. For example:

    file,module,also_accept
    Downloads/notes.txt,CS2030DE,
    Downloads/slides.pdf,CS1010E,CS1010S

A comparison file called check_<report name> is written next to the report. The strict result
column ignores also_accept; a separate with_aliases column shows the score with it.
"""
import csv
import sys
from collections import Counter
from pathlib import Path


def find_report(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        reports = sorted(p.glob("moves_report_*.csv"), key=lambda x: x.stat().st_mtime)
        if not reports:
            sys.exit(f"No moves_report_*.csv found in {p}")
        return reports[-1]
    if not p.is_file():
        sys.exit(f"Not found, {p}")
    return p


def norm(path: str) -> str:
    return path.strip().replace("\\", "/")


def item_of(path: str) -> str:
    """Top-level item: a whole folder counts as one, not each file inside it."""
    return path.split("/", 1)[0]


def structure_of(original: str, final_location: str):
    """True if the file kept its subfolder path inside the module folder, False if not, None if it is not in a module folder.
    Only the folder part is compared, so a rename after a name clash still counts."""
    want = "/".join(original.split("/")[:-1])
    placed = [l.split("/") for l in final_location.split(" | ") if l.startswith("Organised/") and len(l.split("/")) >= 3]
    if not placed:
        return None, ""
    for parts in placed:
        if "/".join(parts[2:-1]) == want:
            return True, want
    return False, "Organised/" + "/".join(placed[0][1:-1])


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    report = find_report(sys.argv[1])
    answers_path = Path(sys.argv[2])
    if not answers_path.is_file():
        sys.exit(f"Not found, {answers_path}")

    with open(report, newline="", encoding="utf-8-sig") as fh:
        moved = {norm(r["original"]): r for r in csv.DictReader(fh)}
    with open(answers_path, newline="", encoding="utf-8-sig") as fh:
        answers = {}
        for r in csv.DictReader(fh):
            f, m = norm(r.get("file") or ""), (r.get("module") or "").strip()
            if f and m:
                also = ""
                for name, value in r.items():  # tolerate a header with stray spaces, such as " also_accept"
                    if name and name.strip().lower() == "also_accept":
                        also = (value or "").strip()
                        break
                answers[f] = (m, also)

    by_name = {}  # fallback when the answer key only has a bare file name
    for k in moved:
        by_name.setdefault(k.rsplit("/", 1)[-1], []).append(k)

    def report_key(f: str) -> str | None:
        if f in moved:
            return f
        # A newer moves report may collapse a whole folder into one row.
        top = f.split("/", 1)[0] if "/" in f else None
        if top and top in moved:
            return top
        return by_name.get(f, [None])[0] if len(by_name.get(f, [])) == 1 else None

    rows, not_in_report = [], []
    for f, (want, also) in answers.items():
        key = report_key(f)
        if key is None:
            not_in_report.append(f)
            continue
        got = moved[key]["module_folder"].strip()
        ok = got.upper() == want.upper()
        with_aliases = ok or (bool(also) and got.upper() == also.upper())
        rows.append({"file": key, "expected": want, "got": got, "result": "correct" if ok else "WRONG",
                     "with_aliases": "correct" if with_aliases else "WRONG",
                     "final_location": moved[key]["final_location"]})

    wrong = [r for r in rows if r["result"] == "WRONG"]
    n = len(rows)
    right = n - len(wrong)
    right_aliases = sum(r["with_aliases"] == "correct" for r in rows)
    has_aliases = any(also for _, also in answers.values())
    items = {}
    for k, r in moved.items():
        items.setdefault(item_of(k), []).append((k, r))
    scored_items = {item_of(r["file"]) for r in rows}
    total, with_answer = len(items), len(scored_items)
    print(f"Report  {report}")
    print(f"Answers {answers_path}  ({len(answers)} labelled)")
    print("\n" + "=" * 50)
    print(f"  OVERALL  {right} of {n} scored files sorted correctly"
          + (f"  ({right / n:.0%})" if n else ""))
    if has_aliases:
        print(f"  With also_accept codes:  {right_aliases} of {n} correct"
              + (f"  ({right_aliases / n:.0%})" if n else "")
              + "   <- the strict score above is the main result")
    print(f"  The report has {total} items, {with_answer} of them have an answer, "
          f"{total - with_answer} do not")
    print(f"  Out of all {total} items in the report, {right} are confirmed correct")
    print("=" * 50)
    per = {}
    if n:
        for r in rows:
            c = per.setdefault(r["expected"], [0, 0])
            c[1] += 1
            c[0] += r["result"] == "correct"
        print("\nBy module (correct of expected)")
        for m, (c, t) in sorted(per.items()):
            print(f"  {m:<12} {c} of {t}")
    if wrong:
        print("\nWRONG")
        w = max(len(r["file"]) for r in wrong)
        for r in sorted(wrong, key=lambda r: (r["expected"], r["file"])):
            note = "   [accepted with also_accept]" if r["with_aliases"] == "correct" else ""
            print(f"  {r['file']:<{w}}  expected {r['expected']:<10} got {r['got']}{note}")
        print("\nMistakes by type (expected -> got)")
        for (a, b), c in Counter((r["expected"], r["got"]) for r in wrong).most_common():
            print(f"  {a} -> {b}  x{c}")
    # folder structure, checked once per top-level item, with or without an answer
    kept, changed = 0, []
    for item, members in items.items():
        checks = []
        for k, r in members:
            ok, got = structure_of(k, r["final_location"])
            if ok is not None:
                checks.append((ok, got))
        if not checks:
            continue
        if all(ok for ok, _ in checks):
            kept += 1
        else:
            now = next(got for ok, got in checks if not ok)
            changed.append((item, "/".join(item.split("/")[:-1]) or "(top level)", now))
    print("\n" + "-" * 50)
    print(f"  FOLDER STRUCTURE  kept for {kept} of {kept + len(changed)} items placed in a module folder")
    print("-" * 50)
    for k, was, now in changed:
        print(f"  changed  {k}\n           was in {was}, now in {now}")

    unscored = [item for item in items if item not in scored_items]
    if unscored:
        print(f"\nNot scored, no answer given ({len(unscored)})")
        for item in sorted(unscored):
            mods = sorted({m.strip() for _, r in items[item]
                           for m in r["module_folder"].split(",") if m.strip()})
            print(f"  {item}  went to {', '.join(mods)}")
    if not_in_report:
        print(f"\nIn your answer key but not in the report ({len(not_in_report)}), check the spelling")
        for k in not_in_report:
            print(f"  {k}")

    out = report.with_name("check_" + report.name)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "expected", "got", "result", "with_aliases", "final_location"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["result"] != "WRONG", r["file"])))
        # A summary under the file rows. The text sits in the result column.
        def line(label, text):
            w.writerow({"file": label, "expected": "", "got": "", "result": text, "with_aliases": "",
                        "final_location": ""})
        w.writerow({})
        line("OVERALL", f"{right} of {n} scored files sorted correctly" + (f" ({right / n:.0%})" if n else ""))
        if has_aliases:
            line("WITH ALSO_ACCEPT", f"{right_aliases} of {n}" + (f" ({right_aliases / n:.0%})" if n else ""))
        line("FILES IN REPORT", f"{total} items, {with_answer} with an answer, {total - with_answer} without")
        line("FOLDER STRUCTURE", f"kept for {kept} of {kept + len(changed)} items placed in a module folder")
        for m, (c, t) in sorted(per.items()) if n else []:
            line(f"BY MODULE {m}", f"{c} of {t} correct")
    print(f"\nFull comparison saved to {out}")


if __name__ == "__main__":
    main()