"""Organise a real folder of files by module, using an LLM that reads the files.

1. Optional, make an answer key template to fill in by hand
    python organise.py --input test_files --make-labels labels.csv

2. Run the agent
    python organise.py --input test_files --output organised_output --model qwen/qwen3.8-27b:free

3. Run it and score it against your labels
    python organise.py --input test_files --output organised_output --model qwen/qwen3.8-27b:free --labels labels.csv

The API key is read from a .env file or the OPENROUTER_API_KEY environment variable.
Your original files are never changed. The agent works on a copy.
"""
from __future__ import annotations

import argparse
import json

from harness.llm import OpenRouterLLM
from harness.real import extract_registry, load_labels, make_labels_template, run_organise, save_registry


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="folder of files to organise")
    ap.add_argument("--output", default="organised_output", help="new folder for the result, must be empty or missing")
    ap.add_argument("--model", help="OpenRouter model id")
    ap.add_argument("--labels", help="labels.csv with columns file,module, used only for scoring")
    ap.add_argument("--timetable", help="timetable file (PDF, Word, Excel, ICS or text) or a saved module_registry.json. "
                                        "Keep it outside the input folder.")
    ap.add_argument("--registry-out", default="module_registry.json", help="where the module list from the timetable is saved")
    ap.add_argument("--make-labels", metavar="CSV", help="write a labels template for the input folder and stop")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.3)
    args = ap.parse_args()

    if args.make_labels:
        n = make_labels_template(args.input, args.make_labels)
        print(f"Wrote {n} rows to {args.make_labels}. Fill in the module column (use Unsorted for non coursework), "
              "then pass the file with --labels. Blank rows are not scored.")
        return
    if not args.model:
        ap.error("pass --model, copy the id from the model's page on OpenRouter")

    def show(rec: dict) -> None:
        calls = rec["tool_calls"]
        if not calls:
            print(f"step {rec['step']:>3}  (no tool call) {rec['content'][:80]!r}")
        for c in calls:
            arg = c["arguments"].replace("\n", " ")[:70]
            res = c["result"].replace("\n", " ")[:60]
            print(f"step {rec['step']:>3}  {c['name']} {arg}  ->  {res}")

    try:
        llm = OpenRouterLLM(args.model, temperature=args.temperature)
        labels = load_labels(args.labels) if args.labels else None
        modules = None
        if args.timetable:
            reg = extract_registry(args.timetable, llm)
            modules = reg["modules"]
            if not modules:
                raise ValueError("No module codes were found in the timetable.")
            if reg["method"] != "saved registry":
                save_registry(reg, args.registry_out)
            print(f"Timetable read ({reg['method']}), {len(modules)} modules")
            for m in modules:
                print(f"  {m['code']}  {m['name']}")
            if reg["method"] != "saved registry":
                print(f"Saved to {args.registry_out}. Check it is right, then reuse it next time with "
                      f"--timetable {args.registry_out} to skip this step.\n")
        summary = run_organise(args.input, args.output, llm, args.max_steps, labels, on_step=show, modules=modules)
    except (RuntimeError, FileNotFoundError, FileExistsError, ValueError) as e:
        raise SystemExit(f"\nError, {e}")

    print("\nDone.")
    print(f"  files            {summary['n_files']}")
    print(f"  by module folder {json.dumps(summary['per_module'])}")
    print(f"  unsorted         {summary['unsorted']}")
    if summary["timetable_modules"]:
        print(f"  folders not on the timetable  {summary['unlisted_folders'] or 'none'}")
    print(f"  left behind      {summary['left_behind']}   missing {summary['missing']}")
    print(f"  steps used       {summary['steps_used']}   tool errors {summary['tool_errors']}   stopped by {summary['stop_reason']}")
    if "score" in summary:
        s = summary["score"]
        print(f"  accuracy         {s['n_correct']}/{s['n_labelled']} = {s['accuracy']:.0%}")
        for w in s["wrong"][:15]:
            print(f"    wrong  {w['file']}  expected {w['expected']}  got {w['got']}")
    print(f"\nResult is in {summary['output']}")
    print("  Organised/        the files sorted by module, plus the agent's own log")
    print("  moves_report.csv  where every original file ended up")
    print("  _left_behind/     anything the agent did not move")
    print("  _run/             full transcript for your report")


if __name__ == "__main__":
    main()