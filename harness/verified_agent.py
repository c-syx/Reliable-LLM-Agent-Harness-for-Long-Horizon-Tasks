"""Action-verifying agent built on the shared baseline loop.

Before a move executes, the verifier snapshots the source identity; afterward it checks
the source is gone, the destination exists, and the destination content matches. Writes
and directory creation are checked for their postcondition. Verification failures are
returned as tool results so the model can inspect the workspace and correct the action.
"""
from __future__ import annotations

from .agent import SYSTEM, BaselineAgent, ToolOutcome
from .sandbox import Sandbox
from .tasks import TaskSpec
from .verifier import ActionVerifier

VERIFICATION_NOTE = (
    " After each file operation, pay attention to verification feedback; "
    "if verification fails, inspect the workspace and correct the action."
)


class VerifiedAgent(BaselineAgent):
    name = "verified"

    def __init__(self, llm, max_steps: int = 80, on_step=None):
        super().__init__(llm, max_steps=max_steps, on_step=on_step)
        self.verifier: ActionVerifier | None = None
        self._move_snapshot: dict | None = None

    def system_prompt(self) -> str:
        return SYSTEM + VERIFICATION_NOTE

    def run(self, spec: TaskSpec, sandbox: Sandbox, resume: dict | None = None):
        self.verifier = ActionVerifier(sandbox)
        self._move_snapshot = None
        return super().run(spec, sandbox, resume)

    # ---- hooks used by the shared loop --------------------------------
    def before_call(self, name: str, args: dict) -> str | ToolOutcome | None:
        if name not in ("move_file", "move_folder"):
            return None

        src, dst = args.get("src"), args.get("dst")
        if not isinstance(src, str) or not isinstance(dst, str):
            detail = f"ERROR: {name} requires string src and dst arguments"
            return ToolOutcome(detail, {"ok": False, "detail": detail})

        self._move_snapshot = None
        try:
            # Capture the source before executing the operation. Invalid preconditions
            # stop the move rather than being treated as verified.
            if name == "move_file":
                self._move_snapshot = self.verifier.before_move(src, dst)
            else:
                self._move_snapshot = self.verifier.before_move_folder(src, dst)
        except Exception as exc:
            detail = f"ERROR: action precondition failed: {exc}"
            return ToolOutcome(detail, {"ok": False, "detail": detail})
        return None

    def after_call(self, name: str, args: dict, result: str) -> str | ToolOutcome | None:
        if name in ("move_file", "move_folder"):
            src, dst = args.get("src"), args.get("dst")
            if not isinstance(src, str) or not isinstance(dst, str):
                return None
            if name == "move_file":
                ok, detail = self.verifier.verify_move(src, dst, result, self._move_snapshot)
            else:
                ok, detail = self.verifier.verify_move_folder(src, dst, result, self._move_snapshot)
            self._move_snapshot = None
            return self._verification_outcome(ok, detail, result, retry_hint=True)

        if name == "make_dir" and isinstance(args.get("path"), str):
            ok, detail = self.verifier.verify_directory_exists(args["path"], result)
            return self._verification_outcome(ok, detail, result)

        if name == "write_file" and isinstance(args.get("path"), str):
            ok, detail = self.verifier.verify_file_exists(args["path"], result)
            return self._verification_outcome(ok, detail, result)

        return None

    @staticmethod
    def _verification_outcome(ok: bool, detail: str, result: str, retry_hint: bool = False) -> ToolOutcome:
        verification = {"ok": ok, "detail": detail}
        if not ok:
            message = f"ERROR: action verification failed: {detail}"
            if retry_hint:
                message += ". Inspect the current workspace before retrying."
            return ToolOutcome(message, verification)
        return ToolOutcome(f"{result}\nVERIFIED: {detail}", verification)
