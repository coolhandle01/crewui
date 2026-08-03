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

import psutil
from rich.text import Text
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Label, RichLog, Static

from crewui.app import (
    CrewAIPipelineTUI,
    FeedbackArea,
    _make_tui_human_input_provider,
    _TUILogHandler,
)
from tests.conftest import FakeCrew, FakeResult, FakeTask

MakeCrew = Callable[..., FakeCrew]


def _statuses(app: CrewAIPipelineTUI) -> list[str]:
    return [str(lbl.render()) for lbl in app.query(".task-status").results(Label)]


def _metrics(app: CrewAIPipelineTUI) -> str:
    return str(app.query_one("#metrics", Static).content)


def _crew_log_text(app: CrewAIPipelineTUI) -> str:
    return "\n".join(strip.text for strip in app.query_one("#crew-log", RichLog).lines)


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


class TestLayout:
    async def test_scroll_panes_and_input_share_one_right_edge(self, make_crew: MakeCrew) -> None:
        # The two scroll panes and the input box must end in the same column so
        # their scrollbars line up. A child's own horizontal margin used to
        # shrink its sibling and skew this; the panes now own the inset instead.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test(size=(100, 40)) as pilot:
            await pilot.pause(0.05)
            edges = {
                app.query_one("#agent-session").region.right,
                app.query_one("#crew-log").region.right,
                app.query_one("#human-input").region.right,
            }
            assert len(edges) == 1


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

    async def test_agent_less_task_does_not_misroute_sidebar_rows(
        self, make_crew: MakeCrew
    ) -> None:
        # Sidebar rows exist only for agent-bearing tasks, but callbacks fire per
        # crew-task index. Without a crew-index -> row-index map, running crew
        # task C (index 2) would index the wrong row (or none) and running the
        # agent-less task B (index 1) would mark C's row. Assert the map routes
        # both correctly.
        crew = make_crew(tasks=[FakeTask("A", "a"), FakeTask("B", "b"), FakeTask("C", "c")])
        crew.tasks[1].agent = None  # type: ignore[assignment]
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            assert app._row_of_task == {0: 0, 2: 1}
            c_name, c_status = app._task_widgets[1]  # C's row
            # The agent-less crew task (index 1) must not touch C's row.
            app._set_task_running(1)
            await pilot.pause(0.02)
            assert "running" not in c_name.classes
            # Crew task C (index 2) marks C's row and titles the turn "c".
            app._set_task_running(2)
            await pilot.pause(0.02)
            assert "running" in c_name.classes
            assert str(c_status.render()) == "Running..."
            assert app._turn_box is not None
            assert app._turn_box.border_title == "c"


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
            assert " Total:   0" in block
            # Nothing ran, so no task callback fired and statuses stay Waiting.
            assert called == []
            assert _statuses(app) == ["Waiting", "Waiting", "Waiting"]


class TestLiveMetrics:
    """The sidebar metrics populate from the start of a live run - zeroed, with
    the token rows carried by arrows (up / down / recycle) - and accumulate as
    each turn's spend lands, rather than staying blank until completion."""

    async def test_live_run_shows_zeroed_running_metrics_before_completion(
        self, make_crew: MakeCrew
    ) -> None:
        import threading

        release = threading.Event()
        crew = make_crew(block_until=release)
        app = CrewAIPipelineTUI(crew=crew)  # not dry_run: a real kickoff
        try:
            async with app.run_test() as pilot:
                # kickoff is parked, so the run has started but no turn has
                # finished: the metrics must already read zeros + running.
                await pilot.pause(0.1)
                block = _metrics(app)
                assert " ↑ 0" in block
                assert " ↓ 0" in block
                assert " ↻ 0" in block
                assert " Status:  running" in block
        finally:
            release.set()

    async def test_turn_usage_accumulates_into_the_sidebar_live(self, make_crew: MakeCrew) -> None:
        crew = make_crew()
        app = CrewAIPipelineTUI(crew=crew, dry_run=True, get_token_cost=lambda i, o: 0.0)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            proc = crew.tasks[0].agent._token_process
            proc.sum_prompt_tokens(1000)
            proc.sum_completion_tokens(200)
            app._apply_turn_usage(0)
            # The first turn's spend shows in the sidebar immediately (running).
            block = _metrics(app)
            assert " ↑ 1,000" in block
            assert " ↓ 200" in block
            assert " Status:  running" in block
            # A second agent's turn adds to the running total, not replaces it.
            app._open_agent_turn(1)
            proc2 = crew.tasks[1].agent._token_process
            proc2.sum_prompt_tokens(500)
            proc2.sum_completion_tokens(50)
            app._apply_turn_usage(1)
            block = _metrics(app)
            assert " ↑ 1,500" in block
            assert " ↓ 250" in block


class TestRun:
    async def test_successful_run_marks_all_tasks_done(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), get_token_cost=lambda i, o: 0.5)
        async with app.run_test() as pilot:
            # Wait on the metrics, not the statuses. The last task callback marks
            # the sidebar Done, but the metrics land in a later call_from_thread
            # (_on_done), so waiting for "Done" can observe the run in between
            # and read an empty block. Waiting for the metrics implies both.
            assert await _wait_for(pilot, lambda: " Total:   140" in _metrics(app))
            assert _statuses(app) == ["Done", "Done", "Done"]
            block = _metrics(app)
            assert " ↑ 100" in block
            assert " ↓ 40" in block
            assert " Total:   140" in block
            assert " Cost:    $0.5000" in block
            assert " Status:  done" in block

    async def test_completion_renders_a_labelled_rule_not_a_box(self, make_crew: MakeCrew) -> None:
        # The run's end is a marker, not content: a centred labelled rule, with
        # the deliverable staying in the last agent turn (not repeated here).
        result = FakeResult(raw="the final plan")
        app = CrewAIPipelineTUI(crew=make_crew(result=result))
        async with app.run_test() as pilot:
            assert await _wait_for(pilot, lambda: bool(app.query(".finish-rule")))
            rule = app.query_one(".finish-rule", Static)
            assert "Pipeline Complete" in str(rule.render())
            # The rule carries no box and does not repeat the deliverable.
            assert "the final plan" not in str(rule.render())
            assert not app.query(".done-box")

    async def test_run_without_token_usage_falls_back_to_running_totals_done(
        self, make_crew: MakeCrew
    ) -> None:
        result = FakeResult(raw="no usage here", token_usage=None)
        app = CrewAIPipelineTUI(crew=make_crew(result=result))
        async with app.run_test() as pilot:
            # No authoritative crew usage: the sidebar keeps the running totals
            # (zero for the fake crew) but the run is marked done, not left
            # blank or stuck on "running".
            assert await _wait_for(pilot, lambda: " Status:  done" in _metrics(app))
            assert " ↑ 0" in _metrics(app)

    async def test_total_tokens_falls_back_to_prompt_plus_completion(
        self, make_crew: MakeCrew
    ) -> None:
        # A usage object missing total_tokens: the App sums prompt + completion.
        class NoTotal:
            prompt_tokens = 30
            completion_tokens = 12

        app = CrewAIPipelineTUI(crew=make_crew(result=FakeResult(token_usage=NoTotal())))
        async with app.run_test() as pilot:
            await _wait_for(pilot, lambda: " Total:   42" in _metrics(app))
            assert " Total:   42" in _metrics(app)

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
                return any(
                    "Pipeline error" in str(box.render())
                    for box in app.query(".agent-text").results(Static)
                )

            assert await _wait_for(pilot, logged)
            # The run raised inside kickoff; the UI stays up and the sidebar
            # never advances past the first task.
            assert _statuses(app)[0] == "Running..."

    async def test_kickoff_error_with_bracketed_message_reaches_both_panes(
        self, make_crew: MakeCrew
    ) -> None:
        # The crew-error path writes the exception to both the agent pane and the
        # crew log. A bracketed path in the message must not MarkupError out of
        # either write - previously the agent-pane write raised and the crew-log
        # copy was silently lost. Both panes show the error, path intact.
        crew = make_crew(raise_on_kickoff=True, raise_message="could not read [/etc/hosts]")
        app = CrewAIPipelineTUI(crew=crew)
        async with app.run_test() as pilot:
            assert await _wait_for(
                pilot,
                lambda: any(
                    "/etc/hosts" in str(box.render())
                    for box in app.query(".agent-text").results(Static)
                ),
            )
            assert "/etc/hosts" in _crew_log_text(app)


class TestHumanReview:
    """The gate is a multi-line ``FeedbackArea``: Enter submits, Ctrl+J and
    Shift+Enter insert a newline, and an empty submit accepts as-is. Each path
    is driven end to end through the worker-thread bridge so the value the
    operator actually gets back is asserted, not just that nothing raised.
    """

    async def test_feedback_gate_opens_and_submit_returns_value(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            # The gate opens on the UI thread; wait for the input to enable.
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"tighten it")
            await pilot.press("enter")
            worker.join(timeout=2)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
            assert captured == ["tighten it"]
            # The gate closes again after submission, cleared for the next round.
            assert inp.disabled
            assert inp.text == ""

    async def test_duplicate_submit_does_not_re_resolve_or_echo_twice(
        self, make_crew: MakeCrew
    ) -> None:
        # A double-tapped Enter posts two Submitted messages for one gate. The
        # first resolves it; the queued duplicate must be a no-op - not overwrite
        # the answer (which, once the next gate opens, is how the previous
        # round's text silently answers the next review) and not echo twice.
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            app._feedback_event = threading.Event()
            app._feedback_value = ""
            app._open_feedback_gate()
            await pilot.pause(0.05)
            app.on_feedback_area_submitted(FeedbackArea.Submitted("first"))
            await pilot.pause(0.05)
            assert app._feedback_value == "first"
            assert app._feedback_event is None  # the gate is claimed
            assert len(app.query(".you-box")) == 1
            # The queued duplicate lands next - it must change nothing.
            app.on_feedback_area_submitted(FeedbackArea.Submitted("second"))
            await pilot.pause(0.05)
            assert app._feedback_value == "first"  # not overwritten
            assert len(app.query(".you-box")) == 1  # not echoed twice

    async def test_gate_open_at_teardown_releases_the_worker(self, make_crew: MakeCrew) -> None:
        # Any exit that is not a submit - app.exit(), an unhandled UI error, or a
        # pilot ending with the gate open - must release the parked worker, not
        # leave it on _feedback_event.wait() forever (which wedges teardown on
        # the executor join). The worker returns "accept as-is".
        app = CrewAIPipelineTUI(crew=make_crew())
        captured: list[str] = []

        def ask() -> None:
            captured.append(app._await_feedback())

        worker = threading.Thread(target=ask, daemon=True)
        async with app.run_test() as pilot:
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)  # gate is open
        # The context has exited: the app unmounted with the gate still open.
        worker.join(timeout=2)
        assert not worker.is_alive()  # released, not hung
        assert captured == [""]

    async def test_empty_submit_accepts_the_result_as_is(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            # Enter on an empty box means "accept" - an empty value comes back.
            await pilot.press("enter")
            worker.join(timeout=2)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
            assert captured == [""]
            assert inp.disabled

    async def test_whitespace_only_feedback_strips_to_accept(self, make_crew: MakeCrew) -> None:
        # Ctrl+J then Enter submits a newline-only value. It must strip to "" so
        # CrewAI reads it as "accept", not as real feedback that triggers another
        # review round while the box has already echoed "Accepted".
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press("ctrl+j")  # inserts a newline, does not submit
            await pilot.press("enter")  # submits "\n"
            worker.join(timeout=2)
            assert not worker.is_alive()
            assert captured == [""]  # stripped to accept
            you = list(app.query(".you-box").results(Static))
            assert "Accepted" in str(you[-1].render())

    async def test_ctrl_j_inserts_newline_and_full_value_submits(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
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
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
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

            worker = threading.Thread(target=ask, daemon=True)
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
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite

    async def test_open_gate_without_agent_session_still_enables_input(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            # Remove the session pane, then open the gate: the gate-box mount is
            # skipped (NoMatches) but the input must still enable and focus.
            await app.query_one("#agent-session", VerticalScroll).remove()
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

    async def test_full_review_round_reinvokes_and_renders(self, make_crew: MakeCrew) -> None:
        """The seam neither layer tested alone: crewai's real handle_feedback,
        driven through crewui's provider and a pilot-typed gate, re-invokes the
        agent - and the round is legible (a gate box and an echoed 'you' box).
        """

        class _Answer:
            def __init__(self, output: str) -> None:
                self.output = output

        class _Ctx:
            """The subset of crewai's ExecutorContext SyncHumanInputProvider uses."""

            def __init__(self) -> None:
                self.ask_for_human_input = True
                self.messages: list[dict[str, str]] = []
                self.crew = None
                self.invoke_count = 0

            def _is_training_mode(self) -> bool:
                return False

            def _format_feedback_message(self, feedback: str) -> dict[str, str]:
                return {"role": "user", "content": feedback}

            def _invoke_loop(self) -> _Answer:
                self.invoke_count += 1
                return _Answer(f"revised #{self.invoke_count}")

        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            provider = _make_tui_human_input_provider(app)
            ctx = _Ctx()

            def drive() -> None:
                provider.handle_feedback(_Answer("original"), ctx)

            worker = threading.Thread(target=drive, daemon=True)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"pick basecamp")
            await pilot.press("enter")
            # handle_feedback loops - it prompts again after re-invoking. Submit
            # empty to accept and unwind the loop.
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press("enter")
            worker.join(timeout=5)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite

            assert ctx.invoke_count == 1
            assert any("basecamp" in m["content"] for m in ctx.messages)
            # The round is visible in the session, not just applied in the model.
            assert list(app.query(".you-box"))
            assert list(app.query(".gate-box"))


class TestAgentSession:
    """The agent pane is a scroll of decorated per-turn boxes: a green box per
    agent turn (role/model on the top rail), a centred gate box, and a blue
    'you' box echoing each submitted feedback round.
    """

    def test_agent_label_falls_back_when_task_index_out_of_range(self, make_crew: MakeCrew) -> None:
        # A box opened for a task that is not introspectable (index past the end,
        # or an agent that raises on access) must still get a generic title
        # rather than crash the session.
        app = CrewAIPipelineTUI(crew=make_crew())
        assert app._agent_label(99) == "Agent"

    async def test_write_agent_survives_unbalanced_markup_and_keeps_the_turn(
        self, make_crew: MakeCrew
    ) -> None:
        # Defence in depth: even a caller that hands _write_agent unescaped
        # markup (a stray "[/etc/hosts]") must not raise - and, because the turn
        # buffer accumulates and re-parses, must not poison later writes in the
        # same turn. The turn stays usable and both messages are readable.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._write_agent("danger [/etc/hosts]")  # must not raise
            app._write_agent("still working")  # must not raise, not poisoned
            assert app._turn_text is not None
            rendered = app._turn_text.render()
            plain = rendered.plain if isinstance(rendered, Text) else str(rendered)
            assert "/etc/hosts" in plain
            assert "still working" in plain

    async def test_agent_turn_box_is_titled_with_role(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:

            def has_scout_box() -> bool:
                return any(
                    box.border_title == "scout"
                    for box in app.query(".agent-turn").results(Vertical)
                )

            assert await _wait_for(pilot, has_scout_box)

    async def test_gate_opens_a_titled_gate_box(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_feedback_gate()
            await pilot.pause(0.05)
            boxes = list(app.query(".gate-box").results(Static))
            assert len(boxes) == 1
            assert boxes[0].border_title == "Human Review Requested"

    async def test_feedback_echoes_a_you_box(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"pick basecamp")
            await pilot.press("enter")
            worker.join(timeout=2)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
            await pilot.pause(0.05)
            you = list(app.query(".you-box").results(Static))
            assert len(you) == 1
            assert you[0].border_title == "you"
            assert "pick basecamp" in str(you[0].render())

    async def test_empty_feedback_echoes_an_accepted_you_box(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            captured: list[str] = []

            def ask() -> None:
                captured.append(app._await_feedback())

            worker = threading.Thread(target=ask, daemon=True)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press("enter")  # empty submit = accept as-is
            worker.join(timeout=2)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
            await pilot.pause(0.05)
            # An empty accept still reads as a "you" turn - labelled "Accepted".
            you = list(app.query(".you-box").results(Static))
            assert len(you) == 1
            assert "Accepted" in str(you[0].render())

    async def test_task_usage_stamps_the_turn_subtitle(self, make_crew: MakeCrew) -> None:
        crew = make_crew()
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # Tick the agent's live accumulator as crewai's LLM callback would
            # during task 0, then read the turn's spend back off it.
            proc = crew.tasks[0].agent._token_process
            proc.sum_prompt_tokens(1180)
            proc.sum_completion_tokens(260)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑1.2k · ↓260"

    async def test_cached_tokens_add_a_recycle_rail(self, make_crew: MakeCrew) -> None:
        crew = make_crew()
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            proc = crew.tasks[0].agent._token_process
            proc.sum_prompt_tokens(1610)
            proc.sum_completion_tokens(430)
            proc.sum_cached_prompt_tokens(896)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            # A cached turn appends a recycle rail; an uncached one does not.
            assert app._turn_box.border_subtitle == "↑1.6k · ↓430 · ↻896"

    async def test_second_turn_same_agent_shows_only_the_delta(self, make_crew: MakeCrew) -> None:
        # The accumulator is cumulative across an agent's turns, so a per-agent
        # diff must show the turn's own spend, not the running total.
        crew = make_crew()
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            proc = crew.tasks[0].agent._token_process
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            proc.sum_prompt_tokens(1000)
            proc.sum_completion_tokens(200)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑1.0k · ↓200"
            # Same agent runs again; the accumulator grows to 1500 / 300.
            app._open_agent_turn(0)
            proc.sum_prompt_tokens(500)
            proc.sum_completion_tokens(100)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            # The subtitle shows this turn (500 / 100), not the total (1500 / 300).
            assert app._turn_box.border_subtitle == "↑500 · ↓100"

    async def test_turn_with_no_token_spend_leaves_subtitle_blank(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # No LLM ran for this turn - the accumulator is still zero, so no rail.
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert not app._turn_box.border_subtitle

    async def test_out_of_range_task_index_leaves_subtitle_blank(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # No such task (a re-invoked turn past the list) - swallowed, no rail.
            app._apply_turn_usage(99)
            assert app._turn_box is not None
            assert not app._turn_box.border_subtitle

    async def test_agent_without_accumulator_leaves_subtitle_blank(
        self, make_crew: MakeCrew
    ) -> None:
        crew = make_crew()
        # An agent that exposes no token accumulator (e.g. a custom BaseAgent).
        del crew.tasks[0].agent._token_process
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert not app._turn_box.border_subtitle

    async def test_usage_with_no_open_turn_box_still_snapshots(
        self, make_crew: MakeCrew
    ) -> None:
        # A human-review gate clears _turn_box; the turn's tokens must still be
        # snapshotted (no box to stamp a subtitle on), or they leak into the next
        # turn's delta. Here: record with no box, then a later turn shows only
        # its own delta, not the tokens spent while the box was gone.
        crew = make_crew()
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            assert app._turn_box is None
            proc = crew.tasks[0].agent._token_process
            proc.sum_prompt_tokens(1000)
            proc.sum_completion_tokens(200)
            app._apply_turn_usage(0)
            assert app._turn_box is None  # no crash, no box created
            assert app._usage_snapshots  # but the snapshot was stored
            app._open_agent_turn(0)
            proc.sum_prompt_tokens(50)
            proc.sum_completion_tokens(10)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑50 · ↓10"

    async def test_native_provider_usage_read_off_the_llm_instance(
        self, make_crew: MakeCrew
    ) -> None:
        # A native provider (crewai's Anthropic / Bedrock) ticks the *LLM
        # instance*, not the agent's _token_process, so the subtitle must read
        # the instance's get_token_usage_summary() when the process stays zero.
        crew = make_crew()
        agent = crew.tasks[0].agent
        agent.llm = SimpleNamespace(
            get_token_usage_summary=lambda: SimpleNamespace(
                prompt_tokens=1180, completion_tokens=260, cached_prompt_tokens=0
            )
        )
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑1.2k · ↓260"

    async def test_shared_llm_instance_shows_per_turn_delta(self, make_crew: MakeCrew) -> None:
        # One LLM instance is shared across agents on the native path, and its
        # counters are cumulative, so keying the diff by the instance must show
        # each turn's own spend even as different agents run.
        totals = SimpleNamespace(prompt_tokens=0, completion_tokens=0, cached_prompt_tokens=0)
        shared_llm = SimpleNamespace(get_token_usage_summary=lambda: totals)
        crew = make_crew(tasks=[FakeTask("Recon", "scout"), FakeTask("Report", "scribe")])
        for task in crew.tasks:
            task.agent.llm = shared_llm
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            totals.prompt_tokens, totals.completion_tokens = 1000, 200
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑1.0k · ↓200"
            # Second agent, same instance: cumulative grows to 1500 / 300.
            app._open_agent_turn(1)
            totals.prompt_tokens, totals.completion_tokens = 1500, 300
            app._apply_turn_usage(1)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑500 · ↓100"

    async def test_zeroed_llm_instance_falls_back_to_the_token_process(
        self, make_crew: MakeCrew
    ) -> None:
        # If the LLM instance has recorded nothing (e.g. the scripted demo, or a
        # litellm path that ticks only the process), the subtitle must fall back
        # to the agent's _token_process rather than blank out.
        crew = make_crew()
        agent = crew.tasks[0].agent
        agent.llm = SimpleNamespace(
            get_token_usage_summary=lambda: SimpleNamespace(
                prompt_tokens=0, completion_tokens=0, cached_prompt_tokens=0
            )
        )
        agent._token_process.sum_prompt_tokens(800)
        agent._token_process.sum_completion_tokens(150)
        app = CrewAIPipelineTUI(crew=crew, dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._apply_turn_usage(0)
            assert app._turn_box is not None
            assert app._turn_box.border_subtitle == "↑800 · ↓150"


class TestToolCalls:
    """Tool calls render as collapsed boxes inside the current agent turn, fed
    by CrewAI's tool-usage event bus (which fires on every provider, unlike the
    ReAct step callback). The UI methods and the bus handlers are exercised on
    the UI thread, where ``_on_ui`` dispatches directly - the worker->UI bounce
    is the same seam TestLogHandler covers.
    """

    async def test_started_mounts_a_collapsed_box_with_pending_body(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._tool_started_ui("search_web", {"q": "lisbon"})
            await pilot.pause(0.05)
            coll = app.query_one(".tool-call", Collapsible)
            assert coll.collapsed is True
            assert coll.title == "> search_web(q=lisbon)"
            assert "running" in str(app.query_one(".tool-out", Static).render())

    async def test_finished_fills_body_and_stamps_the_header(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._tool_started_ui("search_web", {"q": "lisbon"})
            app._tool_finished_ui(
                "search_web", {"q": "lisbon"}, "mild ~22C; Alfama", from_cache=True, ok=True
            )
            await pilot.pause(0.05)
            coll = app.query_one(".tool-call", Collapsible)
            # Success tick plus a cache marker; the output fills the body.
            assert "✓" in coll.title
            assert "⚡" in coll.title
            assert "22C" in str(app.query_one(".tool-out", Static).render())

    async def test_error_event_marks_the_header_failed(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._tool_started_ui("recon", "example.com")
            # Through the error bus handler, which pulls name/args/.error off the event.
            app._on_tool_error(
                None, SimpleNamespace(tool_name="recon", tool_args="example.com", error="boom")
            )
            await pilot.pause(0.05)
            coll = app.query_one(".tool-call", Collapsible)
            assert "✗" in coll.title
            assert "boom" in str(app.query_one(".tool-out", Static).render())

    async def test_started_lazily_opens_a_turn_when_none_is_open(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # No turn open yet: a tool starting must open one, then mount into it.
            assert app._turn_box is None
            app._tool_started_ui("recon", "x")
            await pilot.pause(0.05)
            assert app._turn_box is not None
            assert app.query_one(".tool-call", Collapsible) is not None

    async def test_started_without_a_session_is_a_noop(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            await app.query_one("#agent-session", VerticalScroll).remove()
            await pilot.pause(0.05)
            # The session widget is gone; opening a turn fails, so there is
            # nothing to mount into - swallowed, not raised.
            app._tool_started_ui("recon", "x")
            assert not app.query(".tool-call")

    async def test_bus_handlers_extract_event_fields_and_render(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # Fake events with the fields the handlers read; calling on the UI
            # thread dispatches the render directly (no worker bounce).
            app._on_tool_started(None, SimpleNamespace(tool_name="recon", tool_args={"host": "x"}))
            app._on_tool_finished(
                None,
                SimpleNamespace(
                    tool_name="recon",
                    tool_args={"host": "x"},
                    output="found 2 hosts",
                    from_cache=False,
                ),
            )
            await pilot.pause(0.05)
            coll = app.query_one(".tool-call", Collapsible)
            assert coll.title.startswith("> recon(host=x)")
            assert "✓" in coll.title
            assert "found 2 hosts" in str(app.query_one(".tool-out", Static).render())

    async def test_finished_without_a_pending_box_is_a_noop(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # A stray finished event (no start) must not crash or mount anything.
            app._tool_finished_ui("nothing", "x", "orphan", from_cache=False, ok=True)
            await pilot.pause(0.05)
            assert not app.query(".tool-call")

    async def test_parallel_tool_calls_each_fill_their_own_box(self, make_crew: MakeCrew) -> None:
        # The bug this fixes: agents fire tools in parallel and the event bus
        # dispatches concurrently, so both Starts can land before either Finish.
        # Drive that interleaving (start A, start B, finish A, finish B) and
        # assert each result lands in its own box, matched by (name, args).
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._tool_started_ui("hydrate", {"handle": "cloudflare"})
            app._tool_started_ui("hydrate", {"handle": "linkedin"})
            # Finish out of start order to prove matching, not positional luck.
            app._tool_finished_ui(
                "hydrate", {"handle": "linkedin"}, "scope: 2 web urls", from_cache=True, ok=True
            )
            app._tool_finished_ui(
                "hydrate", {"handle": "cloudflare"}, "scope: 50 assets", from_cache=False, ok=True
            )
            await pilot.pause(0.05)
            colls = list(app.query(".tool-call").results(Collapsible))
            assert len(colls) == 2

            def body(coll: Collapsible) -> str:
                return " ".join(str(s.render()) for s in coll.query(".tool-out").results(Static))

            # colls[0] was started for cloudflare, colls[1] for linkedin - each got
            # its own result despite the finishes arriving in the other order, and
            # neither is stuck "running".
            assert colls[0].title.startswith("> hydrate(handle=cloudflare)")
            assert "50 assets" in body(colls[0])
            assert "running" not in body(colls[0])
            assert "⚡" not in colls[0].title
            assert colls[1].title.startswith("> hydrate(handle=linkedin)")
            assert "2 web urls" in body(colls[1])
            assert "⚡" in colls[1].title  # linkedin came from cache

    async def test_turn_with_no_tool_calls_renders_text_only(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._write_agent("just an answer, no tools")
            await pilot.pause(0.05)
            turn = app.query_one(".agent-turn", Vertical)
            # A tool-free turn is exactly text - no empty or stray tool boxes.
            assert not turn.query(".tool-call")
            text = " ".join(str(s.render()) for s in turn.query(".agent-text").results(Static))
            assert "just an answer, no tools" in text

    async def test_prose_after_a_tool_opens_a_fresh_text_block(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._write_agent("thinking")
            app._tool_started_ui("recon", "x")
            app._tool_finished_ui("recon", "x", "done", from_cache=False, ok=True)
            app._write_agent("the answer")
            await pilot.pause(0.05)
            turn = app.query_one(".agent-turn", Vertical)
            texts = [str(s.render()) for s in turn.query(".agent-text").results(Static)]
            # Two distinct text runs, the tool box between them - not one merged run.
            assert "thinking" in texts[0]
            assert any("the answer" in t for t in texts[1:])
            assert turn.query_one(".tool-call", Collapsible) is not None

    async def test_a_run_deregisters_its_tool_handlers(self, make_crew: MakeCrew) -> None:
        # The bus is a global singleton; a finished run must leave no handler
        # bound to the (now-idle) app, or a later crew would drive a dead UI.
        # Assert the handler is registered while the run is in flight, THEN gone
        # once it ends: a bare "not registered" check is also true before
        # registration, so it would pass even if the run never wired the bus.
        from crewai.events import ToolUsageStartedEvent, crewai_event_bus

        release = threading.Event()
        app = CrewAIPipelineTUI(crew=make_crew(block_until=release))

        def registered() -> bool:
            return app._on_tool_started in crewai_event_bus._sync_handlers.get(
                ToolUsageStartedEvent, frozenset()
            )

        try:
            async with app.run_test() as pilot:
                assert await _wait_for(pilot, registered)  # wired while in flight
                release.set()
                assert await _wait_for(pilot, lambda: not registered())  # gone after
        finally:
            release.set()

    async def test_tool_event_on_the_bus_renders_a_box_end_to_end(
        self, make_crew: MakeCrew
    ) -> None:
        # End to end through the real seam: _start_run must actually register the
        # handlers on the bus, so a ToolUsageStartedEvent emitted the way a live
        # run emits it renders a tool-call box. The other tests call the render
        # methods directly and so would pass even if the bus wiring were dead;
        # this one fails if register_handler is not called.
        from crewai.events import ToolUsageStartedEvent, crewai_event_bus

        release = threading.Event()
        crew = make_crew(block_until=release)
        app = CrewAIPipelineTUI(crew=crew)  # non-dry: _start_run wires the bus
        try:
            async with app.run_test() as pilot:
                # Wait until _start_run has actually registered the handler on the
                # bus (it does so just after opening the first turn box), then
                # emit the way a live run does.
                assert await _wait_for(
                    pilot,
                    lambda: (
                        app._on_tool_started
                        in crewai_event_bus._sync_handlers.get(ToolUsageStartedEvent, frozenset())
                    ),
                )
                crewai_event_bus.emit(
                    crew,
                    ToolUsageStartedEvent(tool_name="search_web", tool_args={"q": "lisbon"}),
                )
                assert await _wait_for(
                    pilot,
                    lambda: any(
                        "search_web" in str(box.title)
                        for box in app.query(".tool-call").results(Collapsible)
                    ),
                )
        finally:
            release.set()

    async def test_retried_tool_in_next_turn_fills_its_own_box(
        self, make_crew: MakeCrew
    ) -> None:
        # A tool started in one turn but never finished (a crash/timeout), then
        # the same (name, args) retried in the next turn, must fill the NEW
        # turn's box - not the stale box left pending in the previous turn - and
        # must not leak a permanent pending entry.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            app._tool_started_ui("recon", "x")  # turn 0: started, never finished
            await pilot.pause(0.02)
            app._open_agent_turn(1)
            app._tool_started_ui("recon", "x")  # turn 1: the retry
            app._tool_finished_ui("recon", "x", "found it", False, True)
            await pilot.pause(0.02)
            turns = list(app.query(".agent-turn").results(Vertical))
            box_a = turns[0].query_one(".tool-call", Collapsible)
            box_b = turns[1].query_one(".tool-call", Collapsible)
            assert "✓" in str(box_b.title)  # the retry's own box is filled
            assert "✓" not in str(box_a.title)  # the stale box stays running
            assert app._tool_key("recon", "x") not in app._pending_tools  # no leak


class TestReasoning:
    """The agent's extended thinking streams into a collapsed reasoning box at
    the top of the turn, fed by LLMThinkingChunkEvent - provider-agnostic, so it
    is exercised on the UI thread with fake events like the tool-call tests.
    """

    async def test_chunks_accumulate_in_a_collapsed_box(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._thinking_chunk_ui("Let me ")
            app._thinking_chunk_ui("think it through.")
            await pilot.pause(0.05)
            box = app.query_one(".reasoning-box", Collapsible)
            assert box.collapsed is True
            assert box.title == "reasoning"
            # Both chunks land in one box, in order.
            out = app.query_one(".reasoning-out", Static)
            assert "Let me think it through." in str(out.render())

    async def test_bus_handler_streams_the_chunk(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._on_thinking_chunk(None, SimpleNamespace(chunk="weighing the options"))
            await pilot.pause(0.05)
            assert "weighing the options" in str(app.query_one(".reasoning-out", Static).render())

    async def test_empty_chunk_is_a_noop(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._on_thinking_chunk(None, SimpleNamespace(chunk=""))
            await pilot.pause(0.05)
            assert not app.query(".reasoning-box")

    async def test_bus_handler_dispatches_without_blocking_render(
        self, make_crew: MakeCrew
    ) -> None:
        # The streaming handler must not render synchronously on the calling
        # (LLM worker) thread - it posts a message so the producer is never
        # blocked per token. So nothing renders until the message pump runs.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._on_thinking_chunk(None, SimpleNamespace(chunk="deferred"))
            assert not app.query(".reasoning-box")  # queued, not rendered inline
            await pilot.pause(0.05)
            assert "deferred" in str(app.query_one(".reasoning-out", Static).render())

    async def test_reasoning_buffer_is_capped(self, make_crew: MakeCrew) -> None:
        # A long reasoning stream must not grow the buffer (or the re-render cost)
        # without bound: it is capped like format_tool_output, keeping the head
        # and marking the truncation.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            for _ in range(60):
                app._thinking_chunk_ui("x" * 200)  # 12000 chars total
            await pilot.pause(0.02)
            rendered = str(app.query_one(".reasoning-out", Static).render())
            assert len(rendered) <= 4100  # bounded near the 4000 cap, not 12000
            assert "…" in rendered  # truncation marked

    async def test_a_tool_ends_the_thinking_run(self, make_crew: MakeCrew) -> None:
        # Think, tool, think again -> two reasoning boxes, the tool between them,
        # not the second thought merged into the first box.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._thinking_chunk_ui("first thought")
            app._tool_started_ui("recon", "x")
            app._thinking_chunk_ui("second thought")
            await pilot.pause(0.05)
            boxes = list(app.query(".reasoning-box").results(Collapsible))
            assert len(boxes) == 2

            def body(box: Collapsible) -> str:
                outs = box.query(".reasoning-out").results(Static)
                return " ".join(str(s.render()) for s in outs)

            assert "first thought" in body(boxes[0])
            assert "second thought" in body(boxes[1])

    async def test_thinking_lazily_opens_a_turn(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            assert app._turn_box is None
            app._thinking_chunk_ui("thinking with no turn open")
            await pilot.pause(0.05)
            assert app._turn_box is not None
            assert app.query_one(".reasoning-box", Collapsible) is not None

    async def test_thinking_without_a_session_is_a_noop(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            await app.query_one("#agent-session", VerticalScroll).remove()
            await pilot.pause(0.05)
            app._thinking_chunk_ui("nowhere to go")
            assert not app.query(".reasoning-box")

    async def test_a_run_deregisters_its_thinking_handler(self, make_crew: MakeCrew) -> None:
        # Registered in flight, then gone - see the tool-handler test for why the
        # bare "not registered" check would pass vacuously.
        from crewai.events import crewai_event_bus
        from crewai.events.types.llm_events import LLMThinkingChunkEvent

        release = threading.Event()
        app = CrewAIPipelineTUI(crew=make_crew(block_until=release))

        def registered() -> bool:
            return app._on_thinking_chunk in crewai_event_bus._sync_handlers.get(
                LLMThinkingChunkEvent, frozenset()
            )

        try:
            async with app.run_test() as pilot:
                assert await _wait_for(pilot, registered)
                release.set()
                assert await _wait_for(pilot, lambda: not registered())
        finally:
            release.set()


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

            worker = threading.Thread(target=emit, daemon=True)
            worker.start()
            # emit() parks on a blocking call_from_thread until the event loop
            # pumps, so it finishes during the pause below, not the join.
            worker.join(timeout=2)
            await pilot.pause(0.1)
            assert not worker.is_alive()  # a wedged worker fails here, never hangs the suite
            assert any(
                "agent line" in str(box.render())
                for box in app.query(".agent-text").results(Static)
            )
            assert len(app.query_one("#crew-log", RichLog).lines) >= 1

    async def test_crew_routed_log_record_with_brackets_does_not_crash(
        self, make_crew: MakeCrew
    ) -> None:
        # Third-party libraries log paths and URLs as ordinary text; a record
        # routed to the crew pane (a markup=True RichLog) must not MarkupError on
        # a bracketed token - it is not markup and should appear literally.
        app = CrewAIPipelineTUI(crew=make_crew(), record_prefix="myapp", dry_run=True)
        async with app.run_test() as pilot:

            def emit() -> None:
                logging.getLogger("httpx").warning("GET /x -> wrote [/var/log/app.log]")

            worker = threading.Thread(target=emit, daemon=True)
            worker.start()
            worker.join(timeout=2)
            await pilot.pause(0.1)
            assert not worker.is_alive()
            assert "/var/log/app.log" in _crew_log_text(app)


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
            await app.query_one("#agent-session", VerticalScroll).remove()
            await app.query_one("#crew-log", RichLog).remove()
            await pilot.pause(0.02)
            app._write_agent("nowhere")
            app._write_crew("nowhere")

    async def test_write_crew_survives_unbalanced_markup(self, make_crew: MakeCrew) -> None:
        # Mirror of _write_agent's defence: a caller handing _write_crew stray
        # markup (an unescaped "[/path]") must fall back to plain text, not raise
        # out of the crew-log write.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._write_crew("boom [/etc/hosts]")  # must not raise
            await pilot.pause(0.05)
            assert "/etc/hosts" in _crew_log_text(app)

    def test_step_callback_swallows_formatting_error(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew())
        callback = app._make_step_callback()

        class Explosive:
            def __str__(self) -> str:
                raise ValueError("cannot render")

        # format_step_message falls through to str(step); this raises there, and
        # the callback must catch it rather than propagate into the crew's run.
        callback(Explosive())

    def test_step_callback_dispatches_formatted_message(self, make_crew: MakeCrew) -> None:
        from unittest.mock import MagicMock

        app = CrewAIPipelineTUI(crew=make_crew())
        # Not running here, so stub the worker->UI bridge; the seam under test is
        # that a well-formed step is formatted and handed to the agent-pane writer
        # (the success path, distinct from the swallow-on-error path above).
        app.call_from_thread = MagicMock()  # type: ignore[method-assign]
        app._make_step_callback()("a plain step line")
        app.call_from_thread.assert_called_once()
        dispatched, message = app.call_from_thread.call_args.args
        assert dispatched == app._write_agent
        assert message == "a plain step line"


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
        # already streamed it into the agent session).
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
    """``_on_ui`` picks the hand-off for the current thread: a caller on the UI
    thread calls the widget method directly (``call_from_thread`` would crash
    there), a worker-thread caller bounces through ``call_from_thread``.
    Exercised against a light stand-in so no Textual event loop is needed.
    (``_ui_dispatch`` is a thin single-string wrapper over this, covered by the
    log-handler tests.)
    """

    def test_direct_call_when_on_ui_thread(self) -> None:
        calls: list[tuple[str, str]] = []
        stub = SimpleNamespace(
            _ui_thread_id=threading.get_ident(),
            call_from_thread=lambda fn, *a: calls.append(("bounced", *a)),
        )

        CrewAIPipelineTUI._on_ui(
            cast("CrewAIPipelineTUI", stub), lambda m: calls.append(("direct", m)), "hi"
        )

        assert calls == [("direct", "hi")]

    def test_bounces_through_call_from_thread_when_off_ui_thread(self) -> None:
        calls: list[tuple[str, str]] = []
        stub = SimpleNamespace(
            _ui_thread_id=-1,  # never matches a real thread id
            call_from_thread=lambda fn, *a: calls.append(("bounced", *a)),
        )

        CrewAIPipelineTUI._on_ui(
            cast("CrewAIPipelineTUI", stub), lambda m: calls.append(("direct", m)), "hi"
        )

        assert calls == [("bounced", "hi")]


class _FakeProc:
    """Stand-in for a psutil.Process child; records terminate/kill and can
    raise psutil.Error to exercise the swallow branches."""

    def __init__(self, raises: bool = False) -> None:
        self.raises = raises
        self.terminated = False
        self.killed = False

    def terminate(self) -> None:
        if self.raises:
            raise psutil.NoSuchProcess(1)
        self.terminated = True

    def kill(self) -> None:
        if self.raises:
            raise psutil.NoSuchProcess(1)
        self.killed = True


class TestBreakGlass:
    """Ctrl+Q teardown. The in-flight hard-exit path ends in os._exit and is
    unreachable in-process (pragma: no cover) - it is proven by the subprocess
    smoke test in test_breakglass_smoke.py. Here we cover the branches that do
    run in-process: the graceful idle quit, terminal restore, and the
    child-kill logic (with psutil mocked so nothing is actually signalled)."""

    async def test_quit_is_graceful_when_idle(self, make_crew: MakeCrew, monkeypatch) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test():
            assert app._pipeline_running is False
            called: list[bool] = []
            monkeypatch.setattr(app, "exit", lambda *a, **k: called.append(True))
            await app.action_quit()
            assert called == [True]  # graceful exit, not a hard teardown

    def test_restore_terminal_runs_driver_teardown(self, make_crew: MakeCrew) -> None:
        # Fake driver (not the real one) so this doesn't collide with Textual's
        # own stop_application_mode call during pilot teardown.
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        called: list[bool] = []
        app._driver = SimpleNamespace(stop_application_mode=lambda: called.append(True))  # type: ignore[assignment]
        app._restore_terminal()
        assert called == [True]

    def test_restore_terminal_swallows_driver_error(self, make_crew: MakeCrew) -> None:
        def boom() -> None:
            raise RuntimeError("boom")

        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        app._driver = SimpleNamespace(stop_application_mode=boom)  # type: ignore[assignment]
        app._restore_terminal()  # error swallowed, no raise

    def test_restore_terminal_is_noop_without_driver(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)  # never mounted -> _driver is None
        app._restore_terminal()  # no-op, must not raise

    def test_kill_run_children_terminates_then_kills_stragglers(self, monkeypatch) -> None:
        clean, straggler, doomed = _FakeProc(), _FakeProc(), _FakeProc(raises=True)
        procs = [clean, straggler, doomed]
        monkeypatch.setattr(
            "psutil.Process", lambda: SimpleNamespace(children=lambda recursive: procs)
        )
        # wait_procs reports straggler + doomed still alive after SIGTERM
        monkeypatch.setattr(
            "psutil.wait_procs", lambda children, timeout: ([clean], [straggler, doomed])
        )
        CrewAIPipelineTUI._kill_run_children()
        assert clean.terminated and straggler.terminated  # SIGTERM the tree
        assert straggler.killed  # SIGKILL the straggler
        assert not clean.killed  # the one that died on SIGTERM is left alone

    def test_kill_run_children_swallows_process_error(self, monkeypatch) -> None:
        def boom() -> None:
            raise psutil.AccessDenied()

        monkeypatch.setattr("psutil.Process", boom)
        CrewAIPipelineTUI._kill_run_children()  # returns cleanly, no raise
