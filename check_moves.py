"""Compare a moves report with your answer key and list every file that went to the wrong folder.

This is a separate tool for you. The agent never sees the answer key.

Usage
    python check_moves.py results/cohere_north_mini_code answer.csv
    python check_moves.py results/cohere_north_mini_code/moves_report_20261026_12.10pm.csv answer.csv

The first argument can be a run folder (the newest moves_report_*.csv in it is used) or a report file.
The answer key has the columns file,module. Rows with a blank module are skipped.
A comparison file called check_<report name> is written next to the report.
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
                answers[f] = m

    by_name = {}  # fallback when the answer key only has a bare file name
    for k in moved:
        by_name.setdefault(k.rsplit("/", 1)[-1], []).append(k)

    rows, not_in_report = [], []
    for f, want in answers.items():
        key = f if f in moved else (by_name.get(f, [None])[0] if len(by_name.get(f, [])) == 1 else None)
        if key is None:
            not_in_report.append(f)
            continue
        got = moved[key]["module_folder"].strip()
        ok = got.upper() == want.upper()
        rows.append({"file": key, "expected": want, "got": got, "result": "correct" if ok else "WRONG",
                     "final_location": moved[key]["final_location"]})

    wrong = [r for r in rows if r["result"] == "WRONG"]
    n = len(rows)
    right = n - len(wrong)
    total = len(moved)
    print(f"Report  {report}")
    print(f"Answers {answers_path}  ({len(answers)} labelled)")
    print("\n" + "=" * 50)
    print(f"  OVERALL  {right} of {n} scored files sorted correctly"
          + (f"  ({right / n:.0%})" if n else ""))
    print(f"  The report has {total} files, {n} of them have an answer, {total - n} do not")
    print(f"  Out of all {total} files in the report, {right} are confirmed correct")
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
            print(f"  {r['file']:<{w}}  expected {r['expected']:<10} got {r['got']}")
        print("\nMistakes by type (expected -> got)")
        for (a, b), c in Counter((r["expected"], r["got"]) for r in wrong).most_common():
            print(f"  {a} -> {b}  x{c}")
    # folder structure, checked for every file in the report, with or without an answer
    kept, changed = 0, []
    for k, r in moved.items():
        ok, got = structure_of(k, r["final_location"])
        if ok is None:
            continue
        if ok:
            kept += 1
        else:
            changed.append((k, "/".join(k.split("/")[:-1]) or "(top level)", got))
    print("\n" + "-" * 50)
    print(f"  FOLDER STRUCTURE  kept for {kept} of {kept + len(changed)} files placed in a module folder")
    print("-" * 50)
    for k, was, now in changed:
        print(f"  changed  {k}\n           was in {was}, now in {now}")

    unscored = [k for k in moved if k not in {r["file"] for r in rows}]
    if unscored:
        print(f"\nNot scored, no answer given ({len(unscored)})")
        for k in unscored:
            print(f"  {k}  went to {moved[k]['module_folder']}")
    if not_in_report:
        print(f"\nIn your answer key but not in the report ({len(not_in_report)}), check the spelling")
        for k in not_in_report:
            print(f"  {k}")

    out = report.with_name("check_" + report.name)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "expected", "got", "result", "final_location"])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["result"] != "WRONG", r["file"])))
        # A summary under the file rows. The text sits in the result column.
        def line(label, text):
            w.writerow({"file": label, "expected": "", "got": "", "result": text, "final_location": ""})
        w.writerow({})
        line("OVERALL", f"{right} of {n} scored files sorted correctly" + (f" ({right / n:.0%})" if n else ""))
        line("FILES IN REPORT", f"{total} files, {n} with an answer, {total - n} without")
        line("FOLDER STRUCTURE", f"kept for {kept} of {kept + len(changed)} files placed in a module folder")
        for m, (c, t) in sorted(per.items()) if n else []:
            line(f"BY MODULE {m}", f"{c} of {t} correct")
    print(f"\nFull comparison saved to {out}")


if __name__ == "__main__":
    main()