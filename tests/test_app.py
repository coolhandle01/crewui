"""tests/test_app.py - the Textual App layer, driven through the pilot harness.

CrewAIPipelineTUI is App + widgets + a worker thread. These tests mount it
against a FakeCrew (see conftest) and use ``App.run_test()`` to exercise the
paths the pure-helper tests cannot reach: sidebar composition, the dry-run
preview, the status transitions a real run drives, the metrics block, the
error path, and the human-review gate.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

from textual.widgets import Label, RichLog, Static

from crewui.app import (
    CrewAIPipelineTUI,
    FeedbackArea,
    _make_tui_human_input_provider,
    _TUILogHandler,
)
from tests.conftest import FakeCrew, FakeResult

MakeCrew = Callable[..., FakeCrew]


def _statuses(app: CrewAIPipelineTUI) -> list[str]:
    return [str(lbl.render()) for lbl in app.query(".task-status").results(Label)]


def _metrics(app: CrewAIPipelineTUI) -> str:
    return str(app.query_one("#metrics", Static).content)


async def _wait_for(pilot: object, predicate: Callable[[], bool], ticks: int = 60) -> bool:
    """Pump the event loop until ``predicate`` holds or ``ticks`` elapse.

    The pipeline runs on a Textual worker thread; call_from_thread updates land
    between pauses, so polling is how a test observes the run advancing.
    """
    for _ in range(ticks):
        if predicate():
            return True
        await pilot.pause(0.05)  # type: ignore[attr-defined]
    return predicate()


class TestCompose:
    async def test_sidebar_lists_one_row_per_task(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), pipeline_name="My Pipeline", dry_run=True)
        async with app.run_test():
            headings = [str(lbl.render()) for lbl in app.query(".phase-heading").results(Label)]
            assert headings == ["Reconnaissance", "Analysis", "Report"]
            names = [str(lbl.render()) for lbl in app.query(".task-name").results(Label)]
            assert names == ["scout", "analyst", "scribe"]
            assert str(app.query_one("#sidebar-title", Label).render()) == "My Pipeline"

    async def test_tasks_without_agent_are_not_listed(self, make_crew: MakeCrew) -> None:
        # A FakeTask always has an agent; drop one to a bare object with agent=None.
        crew = make_crew()
        crew.tasks[1].agent = None  # type: ignore[assignment]
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test():
            names = [str(lbl.render()) for lbl in app.query(".task-name").results(Label)]
            assert names == ["scout", "scribe"]


class TestDryRun:
    async def test_dry_run_does_not_kickoff_and_shows_zeroed_metrics(
        self, make_crew: MakeCrew
    ) -> None:
        called: list[bool] = []
        crew = make_crew()
        crew.tasks[0].callback = lambda _out: called.append(True)
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.1)
            block = _metrics(app)
            assert " Status:  dry run" in block
            assert " Tokens:  0" in block
            # Nothing ran, so no task callback fired and statuses stay Waiting.
            assert called == []
            assert _statuses(app) == ["Waiting", "Waiting", "Waiting"]


class TestRun:
    async def test_successful_run_marks_all_tasks_done(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), get_token_cost=lambda i, o: 0.5)
        async with app.run_test() as pilot:
            done = await _wait_for(
                pilot, lambda: _statuses(app) == ["Done", "Done", "Done"]
            )
            assert done
            block = _metrics(app)
            assert " Tokens:  140" in block
            assert " Cost:    $0.5000" in block
            assert " Status:  done" in block

    async def test_run_without_token_usage_leaves_metrics_untouched(
        self, make_crew: MakeCrew
    ) -> None:
        result = FakeResult(raw="no usage here", token_usage=None)
        app = CrewAIPipelineTUI(crew=make_crew(result=result))
        async with app.run_test() as pilot:
            await _wait_for(pilot, lambda: _statuses(app) == ["Done", "Done", "Done"])
            # No usage -> _on_done returns before touching the metrics widget.
            assert _metrics(app) == ""

    async def test_total_tokens_falls_back_to_prompt_plus_completion(
        self, make_crew: MakeCrew
    ) -> None:
        # A usage object missing total_tokens: the App sums prompt + completion.
        class NoTotal:
            prompt_tokens = 30
            completion_tokens = 12

        app = CrewAIPipelineTUI(crew=make_crew(result=FakeResult(token_usage=NoTotal())))
        async with app.run_test() as pilot:
            await _wait_for(pilot, lambda: " Tokens:  42" in _metrics(app))
            assert " Tokens:  42" in _metrics(app)

    async def test_on_complete_callback_receives_result(self, make_crew: MakeCrew) -> None:
        seen: list[object] = []
        app = CrewAIPipelineTUI(crew=make_crew(), on_complete=seen.append)
        async with app.run_test() as pilot:
            await _wait_for(pilot, lambda: bool(seen))
            assert isinstance(seen[0], FakeResult)

    async def test_on_start_callback_fires_before_kickoff(self, make_crew: MakeCrew) -> None:
        order: list[str] = []
        crew = make_crew()
        crew.tasks[0].callback = lambda _o: order.append("task0")
        app = CrewAIPipelineTUI(crew=crew, on_start=lambda: order.append("start"))
        async with app.run_test() as pilot:
            await _wait_for(pilot, lambda: "task0" in order)
            assert order[0] == "start"

    async def test_on_complete_failure_is_swallowed(self, make_crew: MakeCrew) -> None:
        def boom(_result: object) -> None:
            raise ValueError("cannot save")

        app = CrewAIPipelineTUI(crew=make_crew(), on_complete=boom)
        async with app.run_test() as pilot:
            # A failing on_complete must not crash the run: statuses still finish.
            done = await _wait_for(pilot, lambda: _statuses(app) == ["Done", "Done", "Done"])
            assert done


class TestErrorPath:
    async def test_kickoff_exception_is_reported_not_raised(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(raise_on_kickoff=True))
        async with app.run_test() as pilot:
            def logged() -> bool:
                return len(app.query_one("#agent-log", RichLog).lines) > 0

            assert await _wait_for(pilot, logged)
            # The run raised inside kickoff; the UI stays up and the sidebar
            # never advances past the first task.
            assert _statuses(app)[0] == "Running..."


class TestHumanReview:
    """The gate is a multi-line ``FeedbackArea``: Enter submits, Ctrl+J and
    Shift+Enter insert a newline, and an empty submit accepts as-is. Each path
    is driven end to end through the worker-thread bridge so the value the
    operator actually gets back is asserted, not just that nothing raised.
    """

    async def test_feedback_gate_opens_and_submit_returns_value(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask)
            worker.start()
            # The gate opens on the UI thread; wait for the input to enable.
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"tighten it")
            await pilot.press("enter")
            worker.join(timeout=2)
            assert captured == ["tighten it"]
            # The gate closes again after submission, cleared for the next round.
            assert inp.disabled
            assert inp.text == ""

    async def test_empty_submit_accepts_the_result_as_is(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            # Enter on an empty box means "accept" - an empty value comes back.
            await pilot.press("enter")
            worker.join(timeout=2)
            assert captured == [""]
            assert inp.disabled

    async def test_ctrl_j_inserts_newline_and_full_value_submits(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"line one")
            # Ctrl+J adds a newline instead of submitting - the gate stays open.
            await pilot.press("ctrl+j")
            await pilot.press(*"line two")
            await pilot.pause(0.05)
            assert captured == []
            assert not inp.disabled
            assert inp.text == "line one\nline two"
            # Enter now submits the whole multi-line value.
            await pilot.press("enter")
            worker.join(timeout=2)
            assert captured == ["line one\nline two"]
            assert inp.disabled

    async def test_shift_enter_inserts_newline_without_submitting(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"a")
            await pilot.press("shift+enter")
            await pilot.press(*"b")
            await pilot.pause(0.05)
            # No submit fired; the newline landed between the two characters.
            assert captured == []
            assert not inp.disabled
            assert inp.text == "a\nb"
            # Clean up the parked worker so it does not outlive the test.
            await pilot.press("enter")
            worker.join(timeout=2)

    async def test_open_gate_without_agent_log_still_enables_input(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            # Remove the agent-log pane, then open the gate: the panel write is
            # skipped (NoMatches) but the input must still enable and focus.
            await app.query_one("#agent-log", RichLog).remove()
            await pilot.pause(0.02)
            app._open_feedback_gate()
            inp = app.query_one("#human-input", FeedbackArea)
            assert not inp.disabled

    async def test_submit_without_open_gate_is_ignored(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            inp = app.query_one("#human-input", FeedbackArea)
            inp.text = "stray"
            inp.disabled = False
            inp.focus()
            # No feedback event is pending, so the submit handler returns early
            # and leaves the box as-is.
            await pilot.press("enter")
            await pilot.pause(0.05)
            assert inp.text == "stray"


class TestLogHandler:
    async def test_agent_prefixed_records_go_to_agent_log(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), record_prefix="myapp", dry_run=True)
        async with app.run_test() as pilot:
            # The handler routes records via call_from_thread, which requires a
            # thread other than the app's - just as real log records arrive off
            # the worker thread. Emit from a worker so the routing is exercised.
            def emit() -> None:
                logging.getLogger("myapp.worker").warning("agent line")
                logging.getLogger("urllib3").warning("crew line")

            worker = threading.Thread(target=emit)
            worker.start()
            worker.join(timeout=2)
            await pilot.pause(0.1)
            assert len(app.query_one("#agent-log", RichLog).lines) >= 1
            assert len(app.query_one("#crew-log", RichLog).lines) >= 1


class TestDefensiveBranches:
    """The write helpers and step callback are defensive: a missing widget or a
    step that fails to format must be swallowed, never raised into the run.
    These call the seams directly - no mount needed - so the drop paths are
    covered without contriving a broken UI mid-run.
    """

    async def test_write_helpers_swallow_missing_widget(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            # Remove the log panes, then write: query_one now raises NoMatches
            # (the screen exists, the widget does not) - exactly the teardown
            # race the branch guards - and both writers must swallow it.
            await app.query_one("#agent-log", RichLog).remove()
            await app.query_one("#crew-log", RichLog).remove()
            await pilot.pause(0.02)
            app._write_agent("nowhere")
            app._write_crew("nowhere")

    def test_step_callback_swallows_formatting_error(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        callback = app._make_step_callback()

        class Explosive:
            def __str__(self) -> str:
                raise ValueError("cannot render")

        # format_step_message falls through to str(step); this raises there, and
        # the callback must catch it rather than propagate into the crew's run.
        callback(Explosive())


class TestThemeOwnership:
    def test_owns_absolute_default_stylesheet(self) -> None:
        from pathlib import Path

        css = Path(CrewAIPipelineTUI.CSS_PATH)
        assert css.is_absolute()
        assert css.name == "default.tcss"
        assert css.is_file()


class TestHumanInputProvider:
    """The provider routes CrewAI's feedback prompt to the app's input-box
    bridge (``_await_feedback``) instead of a terminal ``input()``. This pins
    the routing seam, which is the load-bearing contract; the widget/thread
    bridge itself is covered by TestHumanReview.
    """

    def test_prompt_input_delegates_to_app_bridge(self) -> None:
        from unittest.mock import MagicMock

        app = MagicMock()
        app._await_feedback.return_value = "tighten the title"
        provider = _make_tui_human_input_provider(app)

        # crewai calls _prompt_input(crew) and, since 1.15.x, also passes the
        # answer under review as output_to_review. Both shapes must reach the
        # bridge unchanged; the review text is ignored (the step-callback
        # already streamed it into the agent-log pane).
        assert provider._prompt_input(crew=None) == "tighten the title"
        assert (
            provider._prompt_input(crew=None, output_to_review="The sky is blue.")
            == "tighten the title"
        )
        assert app._await_feedback.call_count == 2


class TestLogHandlerDispatch:
    """The log handler must hand every record to the app's thread-aware
    ``_ui_dispatch`` - never call ``call_from_thread`` directly - so a record
    emitted on the UI thread (the human-review gate) does not crash the run.
    """

    def test_agent_record_dispatched_via_ui_dispatch(self) -> None:
        from unittest.mock import MagicMock

        app = MagicMock()
        app._record_prefix = "myapp"
        handler = _TUILogHandler(app)
        record = logging.LogRecord("myapp.scout", logging.INFO, __file__, 1, "hi", None, None)

        handler.emit(record)

        app._ui_dispatch.assert_called_once_with(app._write_agent, "hi")
        app.call_from_thread.assert_not_called()

    def test_crew_record_routed_to_crew_pane(self) -> None:
        from unittest.mock import MagicMock

        app = MagicMock()
        app._record_prefix = "myapp"
        handler = _TUILogHandler(app)
        record = logging.LogRecord(
            "urllib3.connectionpool", logging.INFO, __file__, 1, "noise", None, None
        )

        handler.emit(record)

        app._ui_dispatch.assert_called_once_with(app._write_crew, "noise")
        app.call_from_thread.assert_not_called()


class TestUiDispatch:
    """``_ui_dispatch`` picks the hand-off for the current thread: a caller on
    the UI thread calls the widget method directly (``call_from_thread`` would
    crash there), a worker-thread caller bounces through ``call_from_thread``.
    Exercised against a light stand-in so no Textual event loop is needed.
    """

    def test_direct_call_when_on_ui_thread(self) -> None:
        calls: list[tuple[str, str]] = []
        stub = SimpleNamespace(
            _ui_thread_id=threading.get_ident(),
            call_from_thread=lambda fn, msg: calls.append(("bounced", msg)),
        )

        CrewAIPipelineTUI._ui_dispatch(
            cast("CrewAIPipelineTUI", stub), lambda m: calls.append(("direct", m)), "hi"
        )

        assert calls == [("direct", "hi")]

    def test_bounces_through_call_from_thread_when_off_ui_thread(self) -> None:
        calls: list[tuple[str, str]] = []
        stub = SimpleNamespace(
            _ui_thread_id=-1,  # never matches a real thread id
            call_from_thread=lambda fn, msg: calls.append(("bounced", msg)),
        )

        CrewAIPipelineTUI._ui_dispatch(
            cast("CrewAIPipelineTUI", stub), lambda m: calls.append(("direct", m)), "hi"
        )

        assert calls == [("bounced", "hi")]
