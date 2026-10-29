"""Tool schemas (OpenAI function calling format) and the executor."""
from __future__ import annotations

from .sandbox import Sandbox


def _fn(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


S = {"type": "string"}

TOOL_SPECS = [
    _fn("list_dir", "List the files and folders directly inside a folder. Folders end with a slash.",
        {"path": {**S, "description": "Folder path relative to the workspace root. Use . for the root."}}, ["path"]),
    _fn("read_file", "Read the start of a file as text. Works on text, code, PDF, Word, PowerPoint, Excel and notebook files. "
        "Images and video have no readable text.",
        {"path": {**S, "description": "File path relative to the workspace root."},
         "max_chars": {"type": "integer", "description": "Optional. How many characters to return. Default 1500, maximum 6000."}},
        ["path"]),
    _fn("make_dir", "Create a folder, including any missing parent folders.",
        {"path": {**S, "description": "Folder path relative to the workspace root."}}, ["path"]),
    _fn("move_file", "Move or rename a file. Fails if the destination already exists or its folder is missing.",
        {"src": {**S, "description": "Existing file path."}, "dst": {**S, "description": "New file path including the file name."}},
        ["src", "dst"]),
    _fn("write_file", "Create or overwrite a text file.",
        {"path": {**S, "description": "File path relative to the workspace root."}, "content": S}, ["path", "content"]),
    _fn("finish", "Call this once when the whole task is complete.",
        {"summary": {**S, "description": "One or two sentences on what was done."}}, ["summary"]),
]


# Only the improved agent gets this tool. The baseline agent does not know it exists.
MOVE_FOLDER_SPEC = _fn("move_folder", "Move a whole folder with everything inside it, keeping its inner structure unchanged. "
        "Fails if the destination already exists or its parent folder is missing.",
        {"src": {**S, "description": "Existing folder path."},
         "dst": {**S, "description": "New folder path including the folder name, for example Organised/ABC1234/Lab 5."}},
        ["src", "dst"])

IMPROVED_TOOL_SPECS = TOOL_SPECS + [MOVE_FOLDER_SPEC]


def execute(sb: Sandbox, name: str, args: dict, allowed: set[str] | None = None) -> str:
    """Run one tool call. Bad names or arguments come back as ERROR strings.

    allowed is the set of tool names the calling agent was given. A tool outside it is refused
    even if the model asks for it, so the baseline cannot use tools it was not offered.
    """
    if allowed is not None and name not in allowed:
        return f"ERROR: unknown tool {name}"
    try:
        if name == "list_dir":
            return sb.list_dir(args.get("path", "."))
        if name == "read_file":
            return sb.read_file(args["path"], args.get("max_chars", 1500))
        if name == "make_dir":
            return sb.make_dir(args["path"])
        if name == "move_file":
            return sb.move_file(args["src"], args["dst"])
        if name == "move_folder":
            return sb.move_folder(args["src"], args["dst"])
        if name == "write_file":
            return sb.write_file(args["path"], args["content"])
        if name == "finish":
            return "OK: finished"
        return f"ERROR: unknown tool {name}"
    except KeyError as e:
        return f"ERROR: missing argument {e}"
    except TypeError as e:
        return f"ERROR: bad arguments ({e})"