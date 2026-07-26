"""tests/test_breakglass_smoke.py - break-glass, proven end to end.

The in-flight Ctrl+Q teardown ends in ``os._exit`` and kills the run's child
processes, so it cannot be exercised inside the pytest process (it would take
pytest down with it). This drives the real ``CrewAIPipelineTUI`` in a *child*
interpreter: a run is left blocked in ``kickoff()`` with a live child
subprocess standing in for a scanner/MCP server, then Ctrl+Q is pressed. The
process must exit fast (no 300s worker-join hang) and the child must be dead
(not orphaned and left running against the host's targets).

Coverage of these lines is not counted here (they run in the child), which is
why the corresponding source is marked ``pragma: no cover``; this test is the
behavioural guarantee instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

# Runs in a child interpreter. Blocks the run, spawns a child, then Ctrl+Q.
_DRIVER = """
import asyncio, subprocess, sys, threading
from crewui.app import CrewAIPipelineTUI

child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
print("CHILD=%d" % child.pid, flush=True)

_never = threading.Event()

class _BlockingCrew:
    def __init__(self):
        self.tasks = []
        self.step_callback = None
    def kickoff(self):
        _never.wait()  # a run that never returns - the stuck-run case

async def main():
    app = CrewAIPipelineTUI(crew=_BlockingCrew(), dry_run=False)
    async with app.run_test() as pilot:
        await pilot.pause()
        await asyncio.sleep(0.4)   # let the worker start and block
        await pilot.press("ctrl+q")
        await asyncio.sleep(3.0)   # unreachable: break-glass os._exit fires first

asyncio.run(main())
print("REACHED_END", flush=True)  # must NOT print
"""


def _child_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_breakglass_exits_fast_and_kills_children() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", _DRIVER],
        capture_output=True,
        text=True,
        timeout=20,  # a hung teardown would blow this; break-glass returns in <1s
    )

    # Hard-exit(0) from the panic path: clean, fast, and never reached the end.
    assert proc.returncode == 0, f"stdout={proc.stdout!r} stderr={proc.stderr[-2000:]!r}"
    assert "REACHED_END" not in proc.stdout, "app did not hard-exit at the gate"

    child_pids = [
        int(line.split("=", 1)[1])
        for line in proc.stdout.splitlines()
        if line.startswith("CHILD=")
    ]
    assert child_pids, f"driver did not report a child pid: {proc.stdout!r}"

    # The child scanner must have been killed by break-glass, not orphaned.
    deadline = time.monotonic() + 5
    pid = child_pids[0]
    while _child_alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    if _child_alive(pid):
        os.kill(pid, 9)  # do not leak the process out of the test run
        pytest.fail(f"child {pid} survived break-glass (orphaned scanner)")
