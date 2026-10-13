"""Run the baseline agent over a fixed task set and summarise the results.

Dry run, no API key needed
    python run_experiment.py --mock --repeats 2

Real run
    set OPENROUTER_API_KEY=your_key
    python run_experiment.py --model <model id from the OpenRouter model page>

Runs that already have a result file are skipped, so you can stop and restart freely
when you hit the free tier daily limit.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics as st
import tempfile
from pathlib import Path

from harness.agent import BaselineAgent
from harness.llm import OpenRouterLLM
from harness.mock import OracleLLM
from harness.sandbox import Sandbox
from harness.scoring import score
from harness.tasks import make_task_set


def run_once(agent_cls, llm, spec, max_steps: int) -> dict:
    work = Path(tempfile.mkdtemp(prefix=f"{spec.task_id}_"))
    try:
        spec.materialise(work)
        sb = Sandbox(work, spec.truth())
        result = agent_cls(llm, max_steps=max_steps).run(spec, sb)
        metrics = score(spec, work, result.finished)
        wrong = [e for e in sb.events if e["module_ok"] is False]
        first_err = min((e["step"] for e in wrong), default=None)
        metrics.update({
            "steps_used": len(result.steps),
            "stop_reason": result.stop_reason,
            "n_tool_errors": result.n_tool_errors,
            "n_wrong_moves": len(wrong),
            "first_error_step": first_err,
            "moves_after_first_error": 0 if first_err is None else sum(1 for e in sb.events if e["step"] > first_err),
            "total_tokens": sum(s["usage"].get("total_tokens", 0) for s in result.steps),
        })
        return {"metrics": metrics, "steps": result.steps, "move_events": sb.events}
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarise(out_dir: Path, agent_name: str) -> None:
    rows = []
    for f in sorted((out_dir / agent_name).glob("*.json")):
        d = json.loads(f.read_text())
        rows.append({"task_id": d["task_id"], "repeat": d["repeat"], **d["metrics"]})
    if not rows:
        print("No results yet.")
        return
    keys = list(rows[0].keys())
    with open(out_dir / f"{agent_name}_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    n = len(rows)
    succ = [r["success"] for r in rows]
    print(f"\n{agent_name}, {n} runs")
    print(f"  success rate        {sum(succ) / n:.0%}")
    print(f"  file accuracy       {st.mean(r['file_accuracy'] for r in rows):.1%} "
          f"(sd {st.pstdev(r['file_accuracy'] for r in rows):.1%})")
    print(f"  steps used          {st.mean(r['steps_used'] for r in rows):.1f}")
    print(f"  false success rate  {sum(r['false_success'] for r in rows) / n:.0%}")
    print(f"  tool errors per run {st.mean(r['n_tool_errors'] for r in rows):.1f}")
    print(f"  wrong moves per run {st.mean(r['n_wrong_moves'] for r in rows):.1f}")
    print(f"  summary written to  {out_dir / (agent_name + '_summary.csv')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="OpenRouter model id")
    ap.add_argument("--mock", action="store_true", help="use the scripted oracle instead of a real model")
    ap.add_argument("--n-tasks", type=int, default=10)
    ap.add_argument("--n-files", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--base-seed", type=int, default=1000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--summarise-only", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    name = BaselineAgent.name
    (out / name).mkdir(parents=True, exist_ok=True)
    if args.summarise_only:
        summarise(out, name)
        return
    if not args.mock and not args.model:
        ap.error("pass --model, or use --mock for a dry run")

    llm_real = None if args.mock else OpenRouterLLM(args.model, temperature=args.temperature)
    for spec in make_task_set(args.n_tasks, args.n_files, args.base_seed):
        for k in range(args.repeats):
            path = out / name / f"{spec.task_id}_r{k}.json"
            if path.exists():
                continue
            # In mock mode every odd repeat sends one file to the wrong folder, so the dry run shows failures too.
            llm = llm_real or OracleLLM(spec, wrong_move_index=2 if k % 2 else None)
            try:
                res = run_once(BaselineAgent, llm, spec, args.max_steps)
            except RuntimeError as e:
                print(f"{spec.task_id} r{k} skipped, {e}")
                continue
            res.update({"task_id": spec.task_id, "repeat": k, "agent": name,
                        "model": "mock" if args.mock else args.model})
            path.write_text(json.dumps(res, indent=1))
            m = res["metrics"]
            print(f"{spec.task_id} r{k} success={m['success']} acc={m['file_accuracy']:.0%} steps={m['steps_used']}")
    summarise(out, name)


if __name__ == "__main__":
    main()
