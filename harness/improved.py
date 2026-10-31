"""The improved agent: action verification, folder handling, a sibling check and checkpointing in one loop.

Action verification comes from VerifiedAgent. Moves are snapshotted before they run and
checked afterwards, and make_dir/write_file are checked for their postcondition. A failed
check is returned as a tool error so the model can correct it.

Folder handling adds the move_folder tool, the folder rules in the prompt, and a check
that stops any move that would flatten a subfolder before it happens.

The sibling check compares a file with the files already sorted into each module folder. If it
looks much more like another module's files than the one the agent chose, the move is sent back
once for a second look. See sibling_check.py.

Checkpointing saves the run state after every step. If the model stops answering (a rate
limit, a daily quota, an outage), the run ends cleanly with its work kept and can resume.
The baseline agent gets none of these.
"""
from __future__ import annotations

from .agent import ToolOutcome
from .folder_check import FolderCheck
from .sibling_check import SiblingCheck
from .tools import IMPROVED_TOOL_SPECS
from .verified_agent import VerifiedAgent


class ImprovedAgent(VerifiedAgent):
    name = "improved"
    tool_specs = IMPROVED_TOOL_SPECS
    uses_folder_rules = True  # real.build_prompt adds the folder section for this agent
    supports_checkpoint = True
    stop_on = (RuntimeError,)  # includes LLMUnavailable, and also errors that waiting cannot fix, such as no credit

    def run(self, spec, sandbox, resume=None):
        self.folder_check = FolderCheck()  # fresh memory of where files came from for every run
        if resume:
            self.folder_check.origin = dict(resume.get("folder_origin", {}))
        # It rebuilds what it knows from the workspace, so a resumed run needs no saved state for it.
        self.sibling = SiblingCheck(sandbox)
        self.sibling_log = self.sibling.sibling_log
        return super().run(spec, sandbox, resume)

    def before_call(self, name: str, args: dict) -> str | ToolOutcome | None:
        blocked = self.folder_check.check(name, args)
        if blocked is not None:  # the folder check runs before the action is executed
            return blocked
        blocked = self.sibling.check(name, args)
        if blocked is not None:  # sent back once for a second look
            return blocked
        return super().before_call(name, args)

    def after_call(self, name: str, args: dict, result: str) -> str | ToolOutcome | None:
        self.folder_check.record(name, args, result)
        self.sibling.record(name, args, result)
        return super().after_call(name, args, result)

    def after_step(self, step: int, messages: list, steps: list, n_errors: int) -> None:
        if self.checkpoint_cb:
            self.checkpoint_cb({"step": step, "messages": messages, "steps": steps, "n_errors": n_errors,
                                "folder_origin": self.folder_check.origin})