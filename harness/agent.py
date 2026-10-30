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
    stop_reason: str  # finish, no_tool_call, max_steps, llm_unavailable or llm_error
    n_tool_errors: int
    note: str = ""  # why the model call failed, when it did


class BaselineAgent:
    name = "baseline"
    tool_specs = TOOL_SPECS  # the improved agent overrides this with a longer list
    stop_on = (LLMUnavailable,)  # model errors that end the run with the work so far kept
    supports_checkpoint = False  # only the improved agent can be saved and resumed
    checkpoint_cb = None

    def __init__(self, llm, max_steps: int = 80, on_step=None):
        self.llm = llm
        self.max_steps = max_steps
        self.allowed = {t["function"]["name"] for t in self.tool_specs}
        self.on_step = on_step  # optional callback, called with each step record for live progress

    # Hooks for the improved agent. The baseline does nothing here, so it behaves exactly as before.
    def before_call(self, name: str, args: dict) -> str | None:
        """Return an ERROR string to stop the call before it runs, or None to let it run."""
        return None

    def after_call(self, name: str, args: dict, result: str) -> None:
        """Called after a tool ran. Not called when before_call stopped it."""

    def after_step(self, step: int, messages: list, steps: list, n_errors: int) -> None:
        """Called when a step is complete, and once before the first step. The baseline does nothing."""

    def run(self, spec: TaskSpec, sandbox: Sandbox, resume: dict | None = None) -> RunResult:
        if resume:  # carry on from a saved state
            messages, steps = resume["messages"], resume["steps"]
            n_errors, first = resume["n_errors"], resume["step"] + 1
        else:
            messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec.prompt}]
            steps, n_errors, first = [], 0, 1
        self.after_step(first - 1, messages, steps, n_errors)
        for step in range(first, self.max_steps + 1):
            sandbox.current_step = step
            try:
                reply = self.llm.chat(messages, self.tool_specs)
            except self.stop_on as e:
                # Keep what was done so far instead of losing the whole run. It is scored as not finished.
                reason = "llm_unavailable" if isinstance(e, LLMUnavailable) else "llm_error"
                return RunResult(steps, False, reason, n_errors, str(e))
            messages.append(assistant_message(reply))
            rec = {"step": step, "content": reply.content, "usage": reply.usage,
                   "latency": round(reply.latency, 2), "tool_calls": []}
            steps.append(rec)

            if not reply.tool_calls:
                if self.on_step:
                    self.on_step(rec)
                self.after_step(step, messages, steps, n_errors)
                return RunResult(steps, True, "no_tool_call", n_errors)

            done = False
            for call in reply.tool_calls:
                args = call.arguments
                if args is None:
                    result = "ERROR: arguments were not a valid JSON object"
                else:
                    blocked = self.before_call(call.name, args)
                    if blocked is not None:  # a check stopped the call, so it never ran
                        result = blocked
                    else:
                        result = execute(sandbox, call.name, args, self.allowed)
                        self.after_call(call.name, args, result)
                if result.startswith("ERROR"):
                    n_errors += 1
                rec["tool_calls"].append({"name": call.name, "arguments": call.raw_arguments, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
                if call.name == "finish" and not result.startswith("ERROR"):
                    done = True
                    break
            if self.on_step:
                self.on_step(rec)
            self.after_step(step, messages, steps, n_errors)
            if done:
                return RunResult(steps, True, "finish", n_errors)
        return RunResult(steps, False, "max_steps", n_errors)