"""Seeded generator for messy student folders, with a known correct answer."""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .sandbox import sha

MODULES = {
    "CS2030DE": ["Java Basics", "Data Encapsulation", "Inheritance", "Polymorphism", "Interfaces", "Generics", "Lambdas"],
    "ESP3201A": ["Search and Planning", "Kalman Filters", "Q Learning", "Neural Networks", "Edge AI", "World Models", "Agentic AI"],
    "EE2026": ["Number Systems", "Boolean Algebra", "Finite State Machines", "Timing Analysis", "Memory Design"],
    "MA1522": ["Matrix Algebra", "Determinants", "Eigenvalues", "Vector Spaces", "Linear Maps"],
    "ES2631": ["Energy Balances", "Heat Transfer", "Solar Cells", "Grid Storage", "Wind Power"],
    "PC1101": ["Kinematics", "Newton Laws", "Work and Energy", "Waves", "Thermodynamics"],
}

JUNK_NAMES = [
    "boarding_pass_notes.txt", "Untitled.txt", "recipe_ideas.txt", "flat_viewing_checklist.txt",
    "club_poster_text.md", "random_download (3).txt", "gym_plan.txt", "birthday_gift_list.md",
]

SUBFOLDERS = ["", "", "old stuff", "misc", "week downloads"]

FILLER = [
    "Key definitions are listed below.", "Worked example attempted but not checked.",
    "Remember to revise this before the quiz.", "Tutorial question 3 is the tricky one.",
    "Lecturer said this is examinable.", "Summary copied from the slides.",
    "Need to ask the tutor about the second proof.", "Diagram on the board was hard to read.",
]

PROMPT = """You are tidying a student's messy Downloads folder using the tools provided. All paths are relative to the workspace root.

Rules
1. Every file inside Downloads, including files in its subfolders, belongs to one module. The module code is on the line starting "Module:" inside the file. File names are often misleading, so read the file to decide.
2. Move each module file into Organised/<MODULE>/ and rename it to <MODULE>_W<week as two digits>_<Title with spaces replaced by underscores><original extension>. Week and Title come from the "Week:" and "Title:" lines inside the file. Example: Organised/CS2030DE/CS2030DE_W03_Data_Encapsulation.txt
3. Files whose Module line says NONE are not coursework. Move them to Organised/Unsorted/ and keep their original file names.
4. Write a CSV log to Organised/organise_log.csv with the header original_path,new_path and one row per file. Use workspace relative paths such as Downloads/misc/notes_4.txt.
5. Do not delete files and do not leave any file behind in Downloads.
6. Call finish once every file is handled and you have checked your work."""


@dataclass
class FileSpec:
    orig_path: str
    content: str
    module: str  # "NONE" for junk
    expected_path: str

    @property
    def hash(self) -> str:
        return sha(self.content.encode())


@dataclass
class TaskSpec:
    task_id: str
    seed: int
    files: list[FileSpec]
    prompt: str = PROMPT

    def truth(self) -> dict:
        return {
            f.hash: {"module": "Unsorted" if f.module == "NONE" else f.module, "expected": f.expected_path}
            for f in self.files
        }

    def materialise(self, root: Path) -> None:
        root = Path(root)
        for f in self.files:
            p = root / f.orig_path
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(f.content.encode())  # bytes so hashes match on Windows too


def generate_task(seed: int, n_files: int = 10) -> TaskSpec:
    rng = random.Random(seed)
    n_junk = max(1, n_files // 5)
    n_course = n_files - n_junk
    mods = rng.sample(sorted(MODULES), 4)
    used_titles: set[tuple[str, str]] = set()
    used_paths: set[str] = set()
    files: list[FileSpec] = []

    def unique_path(name: str) -> str:
        folder = rng.choice(SUBFOLDERS)
        base = f"Downloads/{folder}/{name}" if folder else f"Downloads/{name}"
        stem, dot, ext = name.rpartition(".")
        k = 2
        while base in used_paths:
            alt = f"{stem} ({k}).{ext}"
            base = f"Downloads/{folder}/{alt}" if folder else f"Downloads/{alt}"
            k += 1
        used_paths.add(base)
        return base

    for _ in range(n_course):
        m = rng.choice(mods)
        free = [t for t in MODULES[m] if (m, t) not in used_titles]
        if not free:
            m = next(x for x in mods if any((x, t) not in used_titles for t in MODULES[x]))
            free = [t for t in MODULES[m] if (m, t) not in used_titles]
        title = rng.choice(free)
        used_titles.add((m, title))
        week = rng.randint(1, 13)
        ext = rng.choice([".txt", ".txt", ".md"])
        # File names deliberately mislead some of the time.
        if rng.random() < 0.3:
            other = rng.choice([x for x in MODULES if x != m])
            name_title = rng.choice(MODULES[other])
        else:
            name_title = title
        name = rng.choice([
            f"{name_title}{ext}", f"{name_title} v2{ext}", f"Lecture {week}{ext}",
            f"notes_{rng.randint(1, 99)}{ext}", f"{name_title} (1){ext}", f"final_{rng.randint(1, 9)}{ext}",
        ])
        body = [f"Module: {m}", f"Title: {title}", f"Week: {week}", "", rng.choice(FILLER)]
        if rng.random() < 0.4:  # distractor mention of another module
            body.append(f"See also {rng.choice([x for x in mods if x != m])} for related reading.")
        body.append(f"Ref: {rng.getrandbits(32):08x}")
        content = "\n".join(body) + "\n"
        expected = f"Organised/{m}/{m}_W{week:02d}_{title.replace(' ', '_')}{ext}"
        files.append(FileSpec(unique_path(name), content, m, expected))

    for name in rng.sample(JUNK_NAMES, n_junk):
        content = "\n".join(["Module: NONE", f"Not coursework. {rng.choice(FILLER)}", f"Ref: {rng.getrandbits(32):08x}"]) + "\n"
        path = unique_path(name)
        files.append(FileSpec(path, content, "NONE", f"Organised/Unsorted/{Path(name).name}"))

    rng.shuffle(files)
    return TaskSpec(task_id=f"task{seed}", seed=seed, files=files)


def make_task_set(n_tasks: int = 10, n_files: int = 10, base_seed: int = 1000) -> list[TaskSpec]:
    return [generate_task(base_seed + i, n_files) for i in range(n_tasks)]
