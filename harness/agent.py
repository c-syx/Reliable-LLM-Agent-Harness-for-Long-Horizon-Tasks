"""Shared tool loop. The baseline uses it directly; verified and improved add hooks."""
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


@dataclass
class ToolOutcome:
    """A tool result, optionally with verification metadata for the transcript."""
    result: str
    verification: dict | None = None


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

    # Hooks for the improved and verified agents. The baseline does nothing here, so it behaves exactly as before.
    def system_prompt(self) -> str:
        """The system message that opens the conversation."""
        return SYSTEM

    def before_call(self, name: str, args: dict) -> str | ToolOutcome | None:
        """Return an ERROR string or a ToolOutcome to stop the call before it runs, or None to let it run."""
        return None

    def after_call(self, name: str, args: dict, result: str) -> str | ToolOutcome | None:
        """Called after a tool ran. Return a replacement result or None to keep the tool's result."""

    def after_step(self, step: int, messages: list, steps: list, n_errors: int) -> None:
        """Called when a step is complete, and once before the first step. The baseline does nothing."""

    def run(self, spec: TaskSpec, sandbox: Sandbox, resume: dict | None = None) -> RunResult:
        if resume:  # carry on from a saved state
            messages, steps = resume["messages"], resume["steps"]
            n_errors, first = resume["n_errors"], resume["step"] + 1
        else:
            messages = [{"role": "system", "content": self.system_prompt()}, {"role": "user", "content": spec.prompt}]
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
                verification = None
                if args is None:
                    result = "ERROR: arguments were not a valid JSON object"
                else:
                    blocked = self.before_call(call.name, args)
                    if isinstance(blocked, ToolOutcome):  # a check stopped the call with structured feedback
                        result, verification = blocked.result, blocked.verification
                    elif blocked is not None:  # a check stopped the call, so it never ran
                        result = blocked
                    else:
                        result = execute(sandbox, call.name, args, self.allowed)
                        outcome = self.after_call(call.name, args, result)
                        if isinstance(outcome, ToolOutcome):
                            result, verification = outcome.result, outcome.verification
                        elif outcome is not None:
                            result = outcome
                if result.startswith("ERROR"):
                    n_errors += 1
                call_record = {"name": call.name, "arguments": call.raw_arguments, "result": result}
                if verification is not None:
                    call_record["verification"] = verification
                rec["tool_calls"].append(call_record)
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