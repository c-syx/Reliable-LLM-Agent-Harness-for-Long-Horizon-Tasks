"""The improved agent. It starts as the baseline loop plus folder handling.

Folder handling is the first improvement. This agent gets a move_folder tool and the folder
rules in its prompt, so it can keep a subfolder together instead of flattening it. The
baseline agent gets neither. Checkpointing, rollback and verifier checks will be added here
so they stay out of the baseline.
"""
from __future__ import annotations

from .agent import BaselineAgent
from .tools import IMPROVED_TOOL_SPECS


class ImprovedAgent(BaselineAgent):
    name = "improved"
    tool_specs = IMPROVED_TOOL_SPECS
    uses_folder_rules = True  # real.build_prompt adds the folder section for this agent