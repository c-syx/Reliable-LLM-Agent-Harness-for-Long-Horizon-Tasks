"""The improved agent. It starts as the baseline loop plus folder handling.

Folder handling is the first improvement. This agent gets a move_folder tool, the folder rules
in its prompt, and a check that stops any move that would flatten a subfolder. The baseline agent
gets none of these. Checkpointing, rollback and other verifier checks will be added here so they
stay out of the baseline.
"""
from __future__ import annotations

from .agent import BaselineAgent
from .folder_check import FolderCheck
from .tools import IMPROVED_TOOL_SPECS


class ImprovedAgent(BaselineAgent):
    name = "improved"
    tool_specs = IMPROVED_TOOL_SPECS
    uses_folder_rules = True  # real.build_prompt adds the folder section for this agent

    def run(self, spec, sandbox):
        self.folder_check = FolderCheck()  # fresh memory of where files came from for every run
        return super().run(spec, sandbox)

    def before_call(self, name, args):
        return self.folder_check.check(name, args)

    def after_call(self, name, args, result):
        self.folder_check.record(name, args, result)