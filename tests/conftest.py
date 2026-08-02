"""tests/conftest.py - shared fixtures for the crewui suite.

The App tests drive a real ``CrewAIPipelineTUI`` through Textual's pilot
harness, but against a *fake* crew rather than a real ``crewai.Crew``. A fake
keeps the tests deterministic and offline: the App only ever reads ``.tasks``
(each with ``.name`` / ``.agent.role`` / ``.callback``), writes ``.step_callback``,
and calls ``.kickoff()`` - so a small stand-in exercises every path the App
takes without constructing agents or touching an LLM.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


@dataclass
class FakeTask:
    """Duck-types the subset of crewai.Task the App reads."""

    name: str
    role: str
    callback: Callable[[object], None] | None = None
    agent: SimpleNamespace = field(init=False)

    def __post_init__(self) -> None:
        # ``_token_process`` is the real crewai accumulator: the App reads the
        # per-turn subtitle from ``agent._token_process.get_summary()``, so the
        # fake carries the genuine object a live run would tick up.
        from crewai.agents.agent_builder.utilities.base_token_process import TokenProcess

        self.agent = SimpleNamespace(role=self.role, _token_process=TokenProcess())


@dataclass
class FakeUsage:
    prompt_tokens: int = 100
    completion_tokens: int = 40
    total_tokens: int = 140


@dataclass
class FakeResult:
    raw: str = "fake pipeline result"
    token_usage: FakeUsage | None = field(default_factory=FakeUsage)


class FakeCrew:
    """A stand-in for crewai.Crew.

    ``kickoff`` walks the tasks, firing each task callback so the sidebar
    advances, and returns ``result``. ``steps`` are optionally streamed through
    ``step_callback`` first. Raising is opt-in via ``raise_on_kickoff`` so the
    error path can be covered.
    """

    def __init__(
        self,
        tasks: list[FakeTask],
        result: object | None = None,
        steps: list[object] | None = None,
        raise_on_kickoff: bool = False,
        block_until: object | None = None,
    ) -> None:
        self.tasks = tasks
        self.step_callback: Callable[[object], None] | None = None
        self._result = result if result is not None else FakeResult()
        self._steps = steps or []
        self._raise = raise_on_kickoff
        # A threading.Event kickoff parks on before doing any work, so a test can
        # observe the mid-run UI (e.g. the zeroed live metrics) before releasing.
        self._block_until = block_until

    def kickoff(self) -> object:
        if self._block_until is not None:
            self._block_until.wait(timeout=5)
        for step in self._steps:
            if self.step_callback is not None:
                self.step_callback(step)
        if self._raise:
            raise RuntimeError("boom")
        for task in self.tasks:
            if task.callback is not None:
                task.callback("done")
        return self._result


@pytest.fixture
def make_crew() -> Callable[..., FakeCrew]:
    """Factory building a FakeCrew with a default three-phase task list."""

    def _make(
        tasks: list[FakeTask] | None = None,
        result: object | None = None,
        steps: list[object] | None = None,
        raise_on_kickoff: bool = False,
        block_until: object | None = None,
    ) -> FakeCrew:
        if tasks is None:
            tasks = [
                FakeTask("Reconnaissance", "scout"),
                FakeTask("Analysis", "analyst"),
                FakeTask("Report", "scribe"),
            ]
        return FakeCrew(
            tasks=tasks,
            result=result,
            steps=steps,
            raise_on_kickoff=raise_on_kickoff,
            block_until=block_until,
        )

    return _make
