"""The commands the technical report shows name scripts and flags that exist.

Chapter 8 and appendix M tell a reader what to type. A command naming a script that is not in the
repository, or a flag the script does not take, fails on the reader's machine and nowhere else, so
this test takes every command out of the report and looks each one up in the repository.

Where it looks: every fenced code block of the report, and the inline code of appendix M (the
appendix is a table; its command cells are inline code). Backslash continuations are joined, a
``#`` comment is dropped, and a line is split at ``&&``, ``||``, ``;`` and ``|`` so that a flag
belongs to the command it follows.

What it extracts: an interpreter (a word whose last path part is ``python``, ``python3`` or
``python.sh`` -- ``python``, ``.venv/bin/python``, ``<isaac-sim>/python.sh``), the interpreter's
own options (``-I``, ``-B``, ...), then either a script ending in ``.py`` or ``-m <module>``.

What it checks:
  - the script exists in the repository; for ``-m <module>`` the module's file does
    (``a.b`` -> ``a/b.py`` or ``a/b/__main__.py``), except ``-m pytest``, whose ``.py`` arguments
    must exist instead;
  - every ``--flag`` after the script on that command is written in the script's source as a quoted
    string (``"--flag"`` or ``'--flag'``) -- so a flag named only in a comment or a help sentence does
    not count, and ``--episode`` does not pass on the strength of ``--episode-id``. A shell script
    (not ``.py``) only needs the flag as a whole word;
  - every argument of a shell command (the lines of a ``bash`` or ``sh`` block, and appendix M's
    inline code) that names a repository path -- it begins with ``generation/``, ``validator/``,
    ``scripts/``, ``tests/`` or ``meta/`` and holds no placeholder (``<``, ``*``, ``{``) -- is tracked by
    git: a tracked file, or a directory that holds one. A path made by a ``mkdir -p`` earlier in the
    same block counts, with its parents (the build directories are not tracked). LeRobot's own
    files under ``meta/`` (``info.json``, ``stats.json``, ``tasks.parquet``, ``episodes/``) are the
    release's and never the repository's, so a token naming one of them is skipped -- by that list,
    written out below, not by asking whether the file happens to exist;
  - the interpreter is one a reader has: ``python`` (an environment the text names), the
    repository's ``.venv/bin/python`` from ``make venv``, or ``<isaac-sim>/python.sh``;
  - chapter 8 and appendix M each show at least one command, so an emptied chapter is not a pass.

What it does not check (blind spots):
  1. Commands that are not Python scripts (``hf``, ``lerobot-train``, ``cp``, ``make``): their
     flags are not looked up anywhere.
  2. Positional arguments that are not repository paths, and whether a flag takes the value it is
     given.
  3. A flag quoted somewhere else in the script (a list of names, say) still passes -- the check is
     "the quoted flag is in the file", not "argparse defines it".
  6. Paths in a ``python`` block or in a tree drawing (a block with no language) -- those are not
     shell arguments; a release-relative path inside Python code reads like a repository path.
  4. Python code in a fenced block is not run; only command lines are read.
  5. Commands written in prose outside code blocks and outside appendix M.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / "docs" / "tech-report" / "dataset-technical-report-v0.1.md"

INTERPRETER_NAMES = ("python", "python3", "python.sh")
PUBLIC_INTERPRETERS = ("python", ".venv/bin/python", "<isaac-sim>/python.sh")
EXTERNAL_MODULES = ("pytest",)

FENCE = re.compile(r"^\s*```")
SPAN = re.compile(r"`([^`\n]+)`")
SEPARATOR = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
COMMENT = re.compile(r"(?:^|\s)#.*$")
FLAG = re.compile(r"^--[a-z][\w-]*")
SHELL_BLOCKS = ("bash", "sh")
REPO_PATH_PREFIXES = ("generation/", "validator/", "scripts/", "tests/", "meta/")
# LeRobot's files under meta/ (LeRobotDataset v3.0): they are in every release and in no repository.
RELEASE_META = ("meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/episodes/")
PLACEHOLDER = ("<", "*", "{")


@dataclass(frozen=True)
class Command:
    line: str           # the command as the report shows it (continuations joined)
    interpreter: str
    script: str         # a repository path, or "-m <module>"
    flags: tuple[str, ...]


def section(text: str, heading: str) -> str:
    """The text from the line starting with ``heading`` up to the next '## ' heading."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(heading))
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    return "\n".join(lines[start:end])


def fenced_blocks(text: str) -> list[tuple[str, list[str]]]:
    """(language, lines) of every fenced code block, with backslash continuations joined."""
    blocks: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    pending = ""
    for line in text.splitlines():
        if FENCE.match(line):
            if current is None:
                current = []
                blocks.append((line.strip()[3:].strip(), current))
            else:
                current = None
            pending = ""
            continue
        if current is None:
            continue
        if line.rstrip().endswith("\\"):
            pending += line.rstrip()[:-1] + " "
            continue
        current.append(pending + line)
        pending = ""
    return blocks


def fenced_lines(text: str) -> list[str]:
    """Lines inside fenced code blocks, with backslash continuations joined."""
    return [line for _, lines in fenced_blocks(text) for line in lines]


def inline_spans(text: str) -> list[str]:
    """Inline code outside fenced blocks."""
    spans: list[str] = []
    inside = False
    for line in text.splitlines():
        if FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            spans += SPAN.findall(line)
    return spans


def _words(segment: str) -> list[str]:
    try:
        return shlex.split(segment)
    except ValueError:      # an unbalanced quote in a line of Python code
        return segment.split()


def commands(lines: list[str]) -> list[Command]:
    """Every interpreter-plus-script command in ``lines``."""
    found: list[Command] = []
    for line in lines:
        for segment in SEPARATOR.split(COMMENT.sub("", line)):
            words = _words(segment)
            at = next((i for i, w in enumerate(words)
                       if w.rsplit("/", 1)[-1] in INTERPRETER_NAMES), None)
            if at is None:
                continue
            i = at + 1
            script = None
            while i < len(words) and words[i].startswith("-"):
                if words[i] == "-m" and i + 1 < len(words):
                    script = "-m " + words[i + 1]
                    i += 2
                    break
                if words[i] == "-c":
                    break
                i += 1
            if script is None:
                if i >= len(words) or not words[i].endswith(".py"):
                    continue
                script = words[i]
                i += 1
            rest = words[i:]
            flags = tuple(m.group(0) for w in rest if (m := FLAG.match(w)))
            if script == "-m pytest":
                flags = tuple(w.split("::")[0] for w in rest if w.split("::")[0].endswith(".py"))
            found.append(Command(segment.strip(), words[at], script, flags))
    return found


def report_commands(text: str) -> list[Command]:
    return commands(fenced_lines(text) + inline_spans(section(text, "## 부록 M.")))


def shell_groups(text: str) -> list[list[str]]:
    """The lines of each bash or sh block, and each inline-code span of appendix M on its own."""
    groups = [lines for language, lines in fenced_blocks(text) if language in SHELL_BLOCKS]
    return groups + [[span] for span in inline_spans(section(text, "## 부록 M."))]


@lru_cache(maxsize=None)
def tracked(repo: Path) -> frozenset[str] | None:
    """Every tracked file of ``repo`` and every directory that holds one ("a/b/"), or None
    when git cannot say (then the filesystem is asked instead)."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"], capture_output=True,
                             check=True).stdout.decode("utf-8")
    except (OSError, subprocess.CalledProcessError):
        return None
    names: set[str] = set()
    for f in filter(None, out.split("\0")):
        names.add(f)
        parts = f.split("/")
        names.update("/".join(parts[:i]) + "/" for i in range(1, len(parts)))
    return frozenset(names)


def _is_repo_path(word: str) -> bool:
    return (word.startswith(REPO_PATH_PREFIXES) and not any(c in word for c in PLACEHOLDER)
            and not word.startswith(RELEASE_META))


def path_problems(groups: list[list[str]], repo: Path = REPO) -> list[str]:
    """Shell arguments that name a repository path git does not track."""
    known = tracked(repo)
    problems: list[str] = []
    for lines in groups:
        made: set[str] = set()
        for line in lines:
            for segment in SEPARATOR.split(COMMENT.sub("", line)):
                words = _words(segment)
                paths = [w for w in words if _is_repo_path(w)]
                if words[:1] == ["mkdir"]:
                    for w in paths:
                        parts = w.rstrip("/").split("/")
                        made.update("/".join(parts[:i]) + "/" for i in range(1, len(parts) + 1))
                    continue
                for w in paths:
                    name = w.split("::")[0]
                    as_dir = name.rstrip("/") + "/"
                    if as_dir in made:
                        continue
                    if known is None:
                        ok = (repo / name).exists()
                    else:
                        ok = name in known or as_dir in known
                    if not ok:
                        problems.append(f"{segment.strip()[:70]}: {name} is not tracked in the repository")
    return problems


def command_problems(found: list[Command], repo: Path = REPO) -> list[str]:
    problems: list[str] = []
    for c in found:
        def add(what: str) -> None:
            problems.append(f"{c.line[:70]}: {what}")

        if c.interpreter not in PUBLIC_INTERPRETERS:
            add(f"interpreter {c.interpreter!r} is not one a reader has {PUBLIC_INTERPRETERS}")
        if c.script == "-m pytest":
            for test_file in c.flags:
                if not (repo / test_file).is_file():
                    add(f"test file {test_file} is not in the repository")
            continue
        if c.script.startswith("-m "):
            module = c.script[3:]
            if module in EXTERNAL_MODULES:
                continue
            base = repo / module.replace(".", "/")
            source = next((p for p in (base.with_suffix(".py"), base / "__main__.py") if p.is_file()),
                          None)
            if source is None:
                add(f"module {module} is not in the repository")
                continue
        else:
            source = repo / c.script
            if c.script.startswith(("<", "/", "~")) or not source.is_file():
                add(f"script {c.script} is not in the repository")
                continue
        text = source.read_text(encoding="utf-8")
        for flag in c.flags:
            if source.suffix == ".py":
                ok = f'"{flag}"' in text or f"'{flag}'" in text
            else:
                ok = re.search(r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])", text) is not None
            if not ok:
                add(f"flag {flag} does not appear in {source.relative_to(repo)}")
    return problems


def test_the_forms_the_report_writes_are_extracted() -> None:
    text = "\n".join([
        "```bash",
        ".venv/bin/python -I scripts/make_train_copy.py <a> <b> --copy-videos",
        "<isaac-sim>/python.sh generation/envs/run_s1a.py --scenario S1f --seq 132 \\",
        "    --max-steps 1000 --out <d>",
        "cd x && python generation/scorecard.py --dataset <d>  # --not-a-flag",
        "PYTHONPATH= .venv/bin/python -m pytest -q tests/generation/test_assets_canon.py",
        "python -m generation.conditions --scenario S1e --out <f>",
        "hf download KETI-IRRC/keti-humanoid-hri-safety --revision v26.11.0",
        "```",
        "",
        "## 부록 M. 명령 요약",
        "",
        "| 검증 | `.venv/bin/python validator/validate_dataset.py <d> --json <f>` | `.venv/bin/python` |",
    ])
    got = [(c.interpreter, c.script, c.flags) for c in report_commands(text)]
    assert got == [
        (".venv/bin/python", "scripts/make_train_copy.py", ("--copy-videos",)),
        ("<isaac-sim>/python.sh", "generation/envs/run_s1a.py",
         ("--scenario", "--seq", "--max-steps", "--out")),
        ("python", "generation/scorecard.py", ("--dataset",)),
        (".venv/bin/python", "-m pytest", ("tests/generation/test_assets_canon.py",)),
        ("python", "-m generation.conditions", ("--scenario", "--out")),
        (".venv/bin/python", "validator/validate_dataset.py", ("--json",)),
    ]
    assert command_problems(report_commands(text)) == []


def test_a_missing_script_a_missing_flag_and_a_private_interpreter_are_reported() -> None:
    found = commands([
        ".venv/bin/python generation/no_such_tool.py --dataset <d>",
        ".venv/bin/python generation/scorecard.py --dataset <d> --no-such-flag",
        "python generation/compare_published.py --episode <id>",
        ".venv-lerobot/bin/python generation/compare_published.py --seed 1",
        "python -m generation.no_such_module",
        ".venv/bin/python -m pytest tests/generation/test_no_such_file.py",
    ])
    problems = command_problems(found)
    assert len(problems) == 6, problems
    assert "script generation/no_such_tool.py" in problems[0]
    assert "flag --no-such-flag" in problems[1]
    assert "flag --episode does not appear" in problems[2]      # only --episode-id is defined
    assert "interpreter '.venv-lerobot/bin/python'" in problems[3]
    assert "module generation.no_such_module" in problems[4]
    assert "test file tests/generation/test_no_such_file.py" in problems[5]


def test_shell_arguments_that_name_repository_paths_must_be_tracked() -> None:
    text = "\n".join([
        "```bash",
        "mkdir -p generation/scene/build/textures",
        "cp <d>/assets/a.usd generation/scene/build/",
        "cp <d>/assets/b.usd generation/no_such_dir/",
        ".venv/bin/python validator/validate_dataset.py <d> --repro-hook generation/rerun_seed.sh",
        ".venv/bin/python validator/validate_dataset.py <d> --repro-hook generation/no_such_hook.sh",
        "cat meta/info.json meta/episodes/chunk-000/file-000.parquet meta/rules.json meta/no_such.json",
        "```",
        "",
        "```python",
        'pd.read_parquet("generation/not_a_shell_argument.parquet")',
        "```",
        "",
        "## 부록 M. 명령 요약",
        "",
        "| 대조 | `PYTHONPATH= .venv/bin/python -m pytest -q tests/generation/test_no_such.py` |",
    ])
    problems = path_problems(shell_groups(text))
    assert [p.rsplit(": ", 1)[1].split(" ")[0] for p in problems] == [
        "generation/no_such_dir/", "generation/no_such_hook.sh", "meta/no_such.json",
        "tests/generation/test_no_such.py",
    ], problems


def test_a_flag_counts_only_where_the_script_quotes_it(tmp_path: Path) -> None:
    (tmp_path / "tool.py").write_text(
        '"""Run with --mentioned to ..."""\nap.add_argument("--real")\nap.add_argument(\'--single\')\n',
        encoding="utf-8")
    (tmp_path / "hook.sh").write_text("# takes --word\n", encoding="utf-8")
    found = commands([
        "python tool.py --real --single --mentioned",
        "python -m pytest x.py",
    ])
    found = [c for c in found if c.script == "tool.py"]
    found.append(Command("hook", "python", "hook.sh", ("--word", "--absent")))
    problems = command_problems(found, repo=tmp_path)
    assert len(problems) == 2, problems
    assert "flag --mentioned" in problems[0]
    assert "flag --absent" in problems[1]


def test_chapter_8_and_appendix_m_show_commands() -> None:
    text = REPORT.read_text(encoding="utf-8")
    chapter = section(text, "## 8. ")
    appendix = section(text, "## 부록 M.")
    assert commands(fenced_lines(chapter)), "chapter 8 shows no command"
    assert commands(inline_spans(appendix) + fenced_lines(appendix)), "appendix M shows no command"


def test_every_command_in_the_report_names_a_real_script_and_its_flags() -> None:
    found = report_commands(REPORT.read_text(encoding="utf-8"))
    assert found
    assert command_problems(found) == []


def test_every_repository_path_a_shell_command_in_the_report_names_is_tracked() -> None:
    groups = shell_groups(REPORT.read_text(encoding="utf-8"))
    assert sum(_is_repo_path(w) for lines in groups for line in lines for w in _words(line)) > 0
    assert path_problems(groups) == []
