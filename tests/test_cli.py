"""tests/test_cli.py - the command-line surface and the offline demo crew.

The CLI is intentionally thin; these tests pin the contract the release smoke
test depends on: ``--version`` prints the package version through both entry
points, ``demo`` reaches ``run_demo``, and a bare invocation is informational
(exit 0), not an error.
"""

from __future__ import annotations

import pytest

from crewui import __version__
from crewui.cli import main


class TestVersion:
    def test_version_flag_prints_version_and_exits_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # argparse's version action raises SystemExit(0) after printing.
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert __version__ in capsys.readouterr().out


class TestNoCommand:
    def test_bare_invocation_prints_help_and_returns_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main([]) == 0
        assert "usage: crewui" in capsys.readouterr().out


class TestDemoCommand:
    def test_demo_delegates_to_run_demo(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen: dict[str, bool] = {}

        def fake_run_demo(dry_run: bool = False) -> None:
            seen["dry_run"] = dry_run

        # Patch where run_demo is looked up (imported lazily inside main).
        import crewui.demo

        monkeypatch.setattr(crewui.demo, "run_demo", fake_run_demo)
        assert main(["demo", "--dry-run"]) == 0
        assert seen == {"dry_run": True}


class TestDemoCrew:
    def test_build_demo_crew_is_offline_and_scripted(self) -> None:
        # Constructing the demo crew must not need a real API key, and its
        # kickoff must return the canned result without any network call.
        from crewui.demo import build_demo_crew

        crew = build_demo_crew()
        assert [t.name for t in crew.tasks] == ["Destination Research", "Itinerary", "Budget"]

        fired: list[object] = []
        steps: list[object] = []
        crew.step_callback = steps.append
        for task in crew.tasks:
            task.callback = fired.append
        result = crew.kickoff()
        assert len(fired) == 3
        # Two steps per phase (an AgentAction then an AgentFinish).
        assert len(steps) == 6
        assert result.token_usage.total_tokens == 4670
        # Cached is the sum of the per-turn cache hits (0 + 896 + 1408).
        assert result.token_usage.cached_prompt_tokens == 2304
        assert "complete" in result.raw
        # The demo drives the *real* per-turn path: it ticks each agent's live
        # token accumulator (the one a live run's LLM callback feeds), rather
        # than faking usage onto the TaskOutput.
        assert crew.tasks[1].agent._token_process.get_summary().prompt_tokens == 1610
        assert crew.tasks[2].agent._token_process.get_summary().cached_prompt_tokens == 1408

    def test_run_demo_launches_the_tui(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # run_demo should construct the App and call .run(); stub run() so no
        # terminal is driven, and assert the crew was wired in.
        import crewui.demo as demo

        captured: dict[str, object] = {}
        original_init = demo.CrewAIPipelineTUI.__init__

        def spy_init(self: object, *args: object, **kwargs: object) -> None:
            captured.update(kwargs)
            original_init(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(demo.CrewAIPipelineTUI, "__init__", spy_init)
        monkeypatch.setattr(demo.CrewAIPipelineTUI, "run", lambda self: None)
        demo.run_demo(dry_run=True)
        assert captured["dry_run"] is True
        assert captured["record_prefix"] == "crewui.demo"
