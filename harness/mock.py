"""Scripted stand in for a model, used for tests and for dry runs with no API key."""
from __future__ import annotations

import json
from pathlib import PurePosixPath

from .llm import LLMReply, ToolCall
from .tasks import TaskSpec


class OracleLLM:
    """Does the task correctly. It ignores tool results, so it is a plumbing test, not a model.

    wrong_move_index sends that file to the wrong module folder while the log still
    claims the correct place, which is the cascading failure we want to detect.
    skip_log leaves out the CSV log.
    """

    def __init__(self, spec: TaskSpec, wrong_move_index: int | None = None, skip_log: bool = False):
        folders = sorted({PurePosixPath(f.expected_path).parent.name for f in spec.files})
        script: list[tuple[str, dict]] = [("list_dir", {"path": "Downloads"})]
        script += [("make_dir", {"path": f"Organised/{m}"}) for m in folders]
        for i, f in enumerate(spec.files):
            dst = f.expected_path
            if wrong_move_index == i:
                other = next(m for m in folders if m != PurePosixPath(dst).parent.name)
                dst = f"Organised/{other}/{PurePosixPath(dst).name}"
            script.append(("read_file", {"path": f.orig_path}))
            script.append(("move_file", {"src": f.orig_path, "dst": dst}))
        if not skip_log:
            rows = ["original_path,new_path"] + [f"{f.orig_path},{f.expected_path}" for f in spec.files]
            script.append(("write_file", {"path": "Organised/organise_log.csv", "content": "\n".join(rows) + "\n"}))
        script.append(("finish", {"summary": "Sorted every file and wrote the log."}))
        self.script = script

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        idx = sum(1 for m in messages if m["role"] == "assistant")
        if idx >= len(self.script):
            return LLMReply("All done.", [])
        name, args = self.script[idx]
        return LLMReply("", [ToolCall(f"call_{idx}", name, json.dumps(args))])


class LoopingLLM:
    """Repeats the same call forever, to test the step limit."""

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        return LLMReply("", [ToolCall("loop", "list_dir", json.dumps({"path": "Downloads"}))])


class BrokenArgsLLM:
    """Sends invalid JSON once, then finishes, to test error handling."""

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMReply:
        idx = sum(1 for m in messages if m["role"] == "assistant")
        if idx == 0:
            return LLMReply("", [ToolCall("bad", "move_file", "{not json")])
        return LLMReply("", [ToolCall("fin", "finish", json.dumps({"summary": "gave up"}))])
