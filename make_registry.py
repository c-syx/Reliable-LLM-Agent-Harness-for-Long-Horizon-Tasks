#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from harness.readers import extract_text

# Matches NUS style module codes e.g. CS2030DE, EE2022, ES1103, ESP3201A
CODE_RE = re.compile(r"\b([A-Z]{2,4})\s*(\d{4}[A-Z]{0,2})\b")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build module_registry.json by scanning one or more timetables (ICS/PDF/TXT/...) with regex. No API calls."
    )
    ap.add_argument("files", nargs="+", help="Timetable files from different semesters")
    ap.add_argument("--out", default="module_registry.json", help="Output registry JSON")
    args = ap.parse_args()

    seen = set()
    modules = []

    for f in args.files:
        p = Path(f)
        if not p.is_file():
            raise SystemExit(f"File not found: {p}")
        print(f"Reading {p} ...")
        try:
            text = extract_text(p, 50000)
        except Exception as e:
            raise SystemExit(f"Failed to extract text from {p}: {e}")

        if text.startswith(("NOTE", "(no ")):
            print(f"  WARNING: {text}")
            continue

        upper = text.upper()
        for m in CODE_RE.finditer(upper):
            code = m.group(1) + m.group(2)
            if code not in seen:
                seen.add(code)
                modules.append({"code": code, "name": ""})

    modules.sort(key=lambda x: x["code"])
    reg = {
        "modules": modules,
        "method": "regex-merge",
        "files": [str(Path(f).name) for f in args.files],
    }

    Path(args.out).write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(f"\nSaved {len(modules)} unique modules to {args.out}")
    for m in modules:
        print(f"  {m['code']}")


if __name__ == "__main__":
    main()
