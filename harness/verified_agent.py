"""Baseline-compatible agent with deterministic action verification.

The LLM protocol and tool schemas are unchanged. Before move_file executes, the
verifier snapshots the source file's identity; afterward it checks the source is gone,
the destination exists, and the destination hash matches the original. Verification
failures are returned as tool results so the model can inspect the workspace.
"""
from __future__ import annotations

from .agent import RunResult, SYSTEM
from .llm import LLMUnavailable, assistant_message
from .sandbox import Sandbox
from .tasks import TaskSpec
from .tools import TOOL_SPECS, execute
from .verifier import ActionVerifier


class VerifiedAgent:
    name = "verified"

    def __init__(self, llm, max_steps: int = 80, on_step=None):
        self.llm = llm
        self.max_steps = max_steps
        self.on_step = on_step

    def run(self, spec: TaskSpec, sandbox: Sandbox) -> RunResult:
        messages = [
            {
                "role": "system",
                "content": SYSTEM + " After each file operation, pay attention to verification feedback; if verification fails, inspect the workspace and correct the action.",
            },
            {"role": "user", "content": spec.prompt},
        ]
        steps: list[dict] = []
        n_errors = 0
        verifier = ActionVerifier(sandbox)

        for step in range(1, self.max_steps + 1):
            sandbox.current_step = step
            try:
                reply = self.llm.chat(messages, TOOL_SPECS)
            except LLMUnavailable:
                return RunResult(steps, False, "llm_unavailable", n_errors)

            messages.append(assistant_message(reply))
            rec = {
                "step": step,
                "content": reply.content,
                "usage": reply.usage,
                "latency": round(reply.latency, 2),
                "tool_calls": [],
            }
            steps.append(rec)

            if not reply.tool_calls:
                if self.on_step:
                    self.on_step(rec)
                return RunResult(steps, True, "no_tool_call", n_errors)

            done = False
            for call in reply.tool_calls:
                args = call.arguments
                verification = None
                before = None

                if args is None:
                    result = "ERROR: arguments were not a valid JSON object"
                elif call.name == "move_file":
                    src, dst = args.get("src"), args.get("dst")
                    if not isinstance(src, str) or not isinstance(dst, str):
                        result = "ERROR: move_file requires string src and dst arguments"
                        verification = {"ok": False, "detail": result}
                    else:
                        # Capture the source before executing the operation. Invalid
                        # preconditions stop the move rather than being treated as verified.
                        try:
                            before = verifier.before_move(src, dst)
                        except Exception as exc:
                            result = f"ERROR: action precondition failed: {exc}"
                            verification = {"ok": False, "detail": result}
                        else:
                            result = execute(sandbox, call.name, args)
                            ok, detail = verifier.verify_move(src, dst, result, before)
                            verification = {"ok": ok, "detail": detail}
                            if not ok:
                                result = f"ERROR: action verification failed: {detail}. Inspect the current workspace before retrying."
                            else:
                                result = f"{result}\nVERIFIED: {detail}"
                else:
                    result = execute(sandbox, call.name, args)
                    if call.name == "make_dir" and isinstance(args.get("path"), str):
                        ok, detail = verifier.verify_directory_exists(args["path"], result)
                        verification = {"ok": ok, "detail": detail}
                        if not ok:
                            result = f"ERROR: action verification failed: {detail}"
                        else:
                            result = f"{result}\nVERIFIED: {detail}"
                    elif call.name == "write_file" and isinstance(args.get("path"), str):
                        ok, detail = verifier.verify_file_exists(args["path"], result)
                        verification = {"ok": ok, "detail": detail}
                        if not ok:
                            result = f"ERROR: action verification failed: {detail}"
                        else:
                            result = f"{result}\nVERIFIED: {detail}"

                if result.startswith("ERROR"):
                    n_errors += 1
                call_record = {
                    "name": call.name,
                    "arguments": call.raw_arguments,
                    "result": result,
                }
                if verification is not None:
                    call_record["verification"] = verification
                rec["tool_calls"].append(call_record)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if call.name == "finish" and not result.startswith("ERROR"):
                    done = True
                    break

            if self.on_step:
                self.on_step(rec)
            if done:
                return RunResult(steps, True, "finish", n_errors)

        return RunResult(steps, False, "max_steps", n_errors)
