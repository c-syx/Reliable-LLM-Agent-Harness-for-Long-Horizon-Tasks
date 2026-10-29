"""The improved agent. It starts as the baseline loop plus folder handling and checkpointing.

Folder handling is the first improvement. This agent gets a move_folder tool, the folder rules
in its prompt, and a check that stops any move that would flatten a subfolder.

Checkpointing is the second. After every step the run state is handed to a callback, which saves
it. If the model stops answering (a rate limit, a daily quota, an outage), the run ends cleanly
with its work kept, and can be resumed later. The baseline agent gets none of this. Rollback and
other verifier checks will be added here so they stay out of the baseline.
"""
from __future__ import annotations

from .agent import BaselineAgent
from .folder_check import FolderCheck
from .tools import IMPROVED_TOOL_SPECS


class ImprovedAgent(BaselineAgent):
    name = "improved"
    tool_specs = IMPROVED_TOOL_SPECS
    uses_folder_rules = True  # real.build_prompt adds the folder section for this agent
    supports_checkpoint = True
    stop_on = (RuntimeError,)  # includes LLMUnavailable, and also errors that waiting cannot fix, such as no credit

    def run(self, spec, sandbox, resume=None):
        self.folder_check = FolderCheck()  # fresh memory of where files came from for every run
        if resume:
            self.folder_check.origin = dict(resume.get("folder_origin", {}))
        return super().run(spec, sandbox, resume)

    def before_call(self, name, args):
        return self.folder_check.check(name, args)

    def after_call(self, name, args, result):
        self.folder_check.record(name, args, result)

    def after_step(self, step, messages, steps, n_errors):
        if self.checkpoint_cb:
            self.checkpoint_cb({"step": step, "messages": messages, "steps": steps, "n_errors": n_errors,
                                "folder_origin": self.folder_check.origin})