"""Baseline agent. A plain tool loop with raw history, no checks, no recovery."""
from __future__ import annotations

from dataclasses import dataclass

from .llm import LLMUnavailable, assistant_message
from .sandbox import Sandbox
from .tasks import TaskSpec
from .tools import TOOL_SPECS, execute

SYSTEM = (
    "You are a careful file management agent. Use the tools to complete the task. "
    "Call one tool at a time and wait for its result before deciding what to do next."
)


@dataclass
class RunResult:
    steps: list[dict]
    finished: bool  # the agent claimed it was done, by calling finish or by replying with plain text
    stop_reason: str  # finish, no_tool_call, max_steps or llm_unavailable
    n_tool_errors: int


class BaselineAgent:
    name = "baseline"

    def __init__(self, llm, max_steps: int = 80, on_step=None):
        self.llm = llm
        self.max_steps = max_steps
        self.on_step = on_step  # optional callback, called with each step record for live progress

    def run(self, spec: TaskSpec, sandbox: Sandbox) -> RunResult:
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec.prompt}]
        steps: list[dict] = []
        n_errors = 0
        for step in range(1, self.max_steps + 1):
            sandbox.current_step = step
            try:
                reply = self.llm.chat(messages, TOOL_SPECS)
            except LLMUnavailable:
                # Keep what was done so far instead of losing the whole run. It is scored as not finished.
                return RunResult(steps, False, "llm_unavailable", n_errors)
            messages.append(assistant_message(reply))
            rec = {"step": step, "content": reply.content, "usage": reply.usage,
                   "latency": round(reply.latency, 2), "tool_calls": []}
            steps.append(rec)

            if not reply.tool_calls:
                if self.on_step:
                    self.on_step(rec)
                return RunResult(steps, True, "no_tool_call", n_errors)

            done = False
            for call in reply.tool_calls:
                args = call.arguments
                if args is None:
                    result = "ERROR: arguments were not a valid JSON object"
                else:
                    result = execute(sandbox, call.name, args)
                if result.startswith("ERROR"):
                    n_errors += 1
                rec["tool_calls"].append({"name": call.name, "arguments": call.raw_arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if call.name == "finish" and not result.startswith("ERROR"):
                    done = True
                    break
            if self.on_step:
                self.on_step(rec)
            if done:
                return RunResult(steps, True, "finish", n_errors)
        return RunResult(steps, False, "max_steps", n_errors)