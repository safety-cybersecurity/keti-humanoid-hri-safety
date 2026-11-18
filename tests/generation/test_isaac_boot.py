"""A failed Isaac Sim boot must not report success. No Isaac Sim needed.

Measured on 2026-08-20 with CUDA_VISIBLE_DEVICES="" on a machine that has a
GPU: `AppLauncher({"headless": True})` raised RuntimeError("No CUDA GPUs are
available"), the traceback was printed in full, and the process still exited
0. Kit boots far enough to install its own atexit handler, that handler calls
SimulationApp.close() with the default exit_code=0 and hard-exits, and the
process's real exit code is replaced on the way out.
"""

import ast
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_a_launcher_that_raises_exits_nonzero(tmp_path):
    # A stand-in for isaaclab.app that fails the way a GPU-less machine does,
    # so the helper's failure path runs without booting anything.
    script = tmp_path / "boot.py"
    script.write_text(
        dedent(
            f"""
            import sys, types
            sys.path.insert(0, {str(REPO_ROOT)!r})

            app_mod = types.ModuleType("isaaclab.app")

            def AppLauncher(args):
                raise RuntimeError("No CUDA GPUs are available")

            app_mod.AppLauncher = AppLauncher
            pkg = types.ModuleType("isaaclab")
            pkg.app = app_mod
            sys.modules["isaaclab"] = pkg
            sys.modules["isaaclab.app"] = app_mod

            from generation.isaac_boot import launch_app

            launch_app()
            print("KEPT GOING")
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0
    assert "KEPT GOING" not in result.stdout
    # The reason has to survive too -- an exit code with no traceback tells
    # whoever ran the batch nothing about what to fix.
    assert "No CUDA GPUs are available" in result.stderr


def test_a_successful_launch_returns_the_app(tmp_path):
    script = tmp_path / "boot_ok.py"
    script.write_text(
        dedent(
            f"""
            import sys, types
            sys.path.insert(0, {str(REPO_ROOT)!r})

            app_mod = types.ModuleType("isaaclab.app")

            class AppLauncher:
                def __init__(self, args):
                    self.app = "simulation-app"

            app_mod.AppLauncher = AppLauncher
            pkg = types.ModuleType("isaaclab")
            pkg.app = app_mod
            sys.modules["isaaclab"] = pkg
            sys.modules["isaaclab.app"] = app_mod

            from generation.isaac_boot import launch_app

            launcher, app = launch_app()
            assert app == "simulation-app"
            print("LAUNCHED")
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode == 0, result.stderr
    assert "LAUNCHED" in result.stdout


BOOT_NAMES = {"AppLauncher", "SimulationApp"}


def boot_calls(source: str) -> list[int]:
    """Line numbers where `source` constructs Isaac's app itself.

    Parsed, not grepped. A one-line regex over the raw text missed `from
    isaaclab.app import AppLauncher as Launcher; Launcher(...)` and any call
    whose name and open-paren fell on different lines, so the claim that CI
    blocks direct boots did not hold for either.
    """
    tree = ast.parse(source)

    # Names bound to one of the launchers, including under an alias.
    bound = set(BOOT_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name in BOOT_NAMES:
                    bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # `Launcher = AppLauncher`, and the annotated spelling of it --
            # `Boot: object = Launcher` is an AnnAssign, not an Assign, and
            # following only the latter let an aliased boot through.
            value = node.value
            name = value.id if isinstance(value, ast.Name) else (
                value.attr if isinstance(value, ast.Attribute) else None
            )
            if name in bound:
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        bound.add(target.id)
        elif isinstance(node, ast.ClassDef):
            # `class Boot(AppLauncher): ...` -- calling Boot() boots Isaac
            # just as surely, through a name the assignment cases never see.
            for base in node.bases:
                base_name = base.id if isinstance(base, ast.Name) else (
                    base.attr if isinstance(base, ast.Attribute) else None
                )
                if base_name in bound:
                    bound.add(node.name)

    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        called = func.id if isinstance(func, ast.Name) else (
            func.attr if isinstance(func, ast.Attribute) else None
        )
        if called in bound:
            hits.append(node.lineno)
    return sorted(hits)


def test_the_scan_catches_a_launcher_imported_under_an_alias():
    source = (
        "from isaaclab.app import AppLauncher as Launcher\n"
        "app = Launcher({'headless': True})\n"
    )
    assert boot_calls(source) == [2]


def test_the_scan_catches_a_call_split_across_lines():
    source = "from isaaclab.app import AppLauncher\napp = AppLauncher(\n    {'headless': True}\n)\n"
    assert boot_calls(source) == [2]


def test_the_scan_catches_a_module_qualified_call():
    source = "import isaaclab.app as a\napp = a.AppLauncher({'headless': True})\n"
    assert boot_calls(source) == [2]


def test_the_scan_does_not_flag_the_import_alone():
    assert boot_calls("from isaaclab.app import AppLauncher\n") == []


def test_no_generation_script_boots_isaac_outside_the_helper():
    # The exit-code contract lives in one place; a script that constructs the
    # launcher itself silently opts out of it, which is how all twelve of them
    # came to exit 0 on a boot failure.
    offenders = []
    for path in sorted((REPO_ROOT / "generation").rglob("*.py")):
        if path.name == "isaac_boot.py":
            continue
        for lineno in boot_calls(path.read_text(encoding="utf-8")):
            offenders.append(f"{path.relative_to(REPO_ROOT)}:{lineno}")

    assert offenders == [], "\n".join(offenders)


def test_a_boot_failure_exits_nonzero_even_when_the_streams_are_broken(tmp_path):
    # os._exit(1) was reached only if the two flushes before it returned. A
    # closed stdout raises BrokenPipeError there, the exit never runs, and
    # Kit's atexit handler -- already registered by the time the boot failed
    # -- closes the process with its own 0. That is the exact failure this
    # module exists to prevent, arriving by a different door.
    script = tmp_path / "broken_streams.py"
    script.write_text(
        dedent(
            f"""
            import atexit, os, sys, types
            sys.path.insert(0, {str(REPO_ROOT)!r})

            # Stand-in for Kit's own shutdown hook: exit_code=0, hard exit.
            atexit.register(lambda: os._exit(0))

            class BrokenStream:
                def write(self, *a, **k):
                    return 0

                def flush(self):
                    raise BrokenPipeError("closed pipe")

            sys.stdout = BrokenStream()
            sys.stderr = BrokenStream()

            app_mod = types.ModuleType("isaaclab.app")

            def AppLauncher(args):
                raise RuntimeError("No CUDA GPUs are available")

            app_mod.AppLauncher = AppLauncher
            pkg = types.ModuleType("isaaclab")
            pkg.app = app_mod
            sys.modules["isaaclab"] = pkg
            sys.modules["isaaclab.app"] = app_mod

            from generation.isaac_boot import launch_app

            launch_app()
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=60
    )

    assert result.returncode != 0


def test_the_scan_catches_an_annotated_rebinding():
    # `Boot: object = Launcher` is an AnnAssign, which the first version of
    # this scan did not walk, so the call through Boot was invisible.
    source = (
        "from isaaclab.app import AppLauncher as Launcher\n"
        "Boot: object = Launcher\n"
        "app = Boot({'headless': True})\n"
    )
    assert boot_calls(source) == [3]


def test_the_scan_catches_a_subclassed_launcher():
    source = (
        "from isaaclab.app import AppLauncher\n"
        "class Boot(AppLauncher):\n"
        "    pass\n"
        "app = Boot({'headless': True})\n"
    )
    assert boot_calls(source) == [4]
