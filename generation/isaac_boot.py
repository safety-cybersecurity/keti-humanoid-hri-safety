"""Boot Isaac Sim so that a failed boot cannot report success.

Every script here already wraps its *runtime* in try/finally and passes an
explicit exit code to ``SimulationApp.close(exit_code=...)``, because Kit's
fast-shutdown path would otherwise replace a nonzero code with its own 0.
The boot itself sat outside that guard in all twelve of them.

Measured 2026-08-20, ``CUDA_VISIBLE_DEVICES="" python.sh check_ssm.py`` on a
machine that does have a GPU: ``AppLauncher({"headless": True})`` raised
``RuntimeError: No CUDA GPUs are available``, the traceback printed in full,
Kit logged "SimulationApp.close() was not called explicitly. Shutting down
automatically" -- and the process exited **0**. Kit gets far enough to install
an atexit handler before the failure, and that handler closes with the default
exit_code=0 and hard-exits, discarding the interpreter's own code 1.

``os._exit`` is the only exit that runs no atexit handler, which is precisely
why the failure path uses it: nothing Kit registered can overwrite the code
afterwards. Output is flushed by hand first, since ``os._exit`` will not do it.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Any, NoReturn


def launch_app(launcher_args: dict[str, Any] | None = None) -> tuple[Any, Any]:
    """Start Isaac Lab's AppLauncher. Returns ``(app_launcher, simulation_app)``.

    On any failure the process ends with exit code 1 and never returns, so a
    caller can arm its own try/finally on the line after this one and trust
    that ``simulation_app`` exists.
    """
    try:
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(launcher_args or {"headless": True})
        return app_launcher, app_launcher.app
    except BaseException:  # noqa: BLE001 -- see module docstring
        _die("Isaac Lab AppLauncher 부팅 실패")


def launch_simulation_app(config: dict[str, Any] | None = None) -> Any:
    """Same contract for the scripts that construct ``SimulationApp`` directly."""
    try:
        from isaacsim import SimulationApp

        return SimulationApp(config or {"headless": True})
    except BaseException:  # noqa: BLE001 -- see module docstring
        _die("Isaac Sim SimulationApp 부팅 실패")


def _die(what: str) -> NoReturn:
    # BaseException, not Exception: a KeyboardInterrupt during a two-minute
    # boot would otherwise unwind into Kit's atexit handler and come back out
    # as 0, which is the one outcome this module exists to prevent.
    #
    # try/finally, so that the exit happens whether or not the reporting does.
    # Reporting first and exiting after meant a closed stdout -- flush() raising
    # BrokenPipeError -- skipped the exit entirely and handed the process back
    # to Kit's atexit handler, which closes with 0. Measured: exit code 0 again,
    # by a different door than the one this module was written to shut.
    try:
        print(f"{what}:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(1)
