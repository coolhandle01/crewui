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
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Collapsible, Label, RichLog, Static

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
            assert " Total:   0" in block
            # Nothing ran, so no task callback fired and statuses stay Waiting.
            assert called == []
            assert _statuses(app) == ["Waiting", "Waiting", "Waiting"]


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
            assert " Input:   100" in block
            assert " Output:  40" in block
            assert " Total:   140" in block
            assert " Cost:    $0.5000" in block
            assert " Status:  done" in block

    async def test_completion_renders_a_system_done_box(self, make_crew: MakeCrew) -> None:
        # The run's end is a system note (like the review gate), not the last
        # agent's box: a standalone done-box holds the final deliverable.
        result = FakeResult(raw="the final plan")
        app = CrewAIPipelineTUI(crew=make_crew(result=result))
        async with app.run_test() as pilot:
            assert await _wait_for(pilot, lambda: bool(app.query(".done-box")))
            box = app.query_one(".done-box", Static)
            assert str(box.border_title) == "Pipeline Complete"
            assert "the final plan" in str(box.render())
            # It is not an agent turn box.
            assert "agent-turn" not in box.classes

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

            worker = threading.Thread(target=drive)
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

    def test_agent_label_falls_back_when_task_index_out_of_range(
        self, make_crew: MakeCrew
    ) -> None:
        # A box opened for a task that is not introspectable (index past the end,
        # or an agent that raises on access) must still get a generic title
        # rather than crash the session.
        app = CrewAIPipelineTUI(crew=make_crew())
        assert app._agent_label(99) == "Agent"

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

            worker = threading.Thread(target=ask)
            worker.start()
            inp = app.query_one("#human-input", FeedbackArea)
            assert await _wait_for(pilot, lambda: not inp.disabled)
            inp.focus()
            await pilot.press(*"pick basecamp")
            await pilot.press("enter")
            worker.join(timeout=2)
            await pilot.pause(0.05)
            you = list(app.query(".you-box").results(Static))
            assert len(you) == 1
            assert you[0].border_title == "you"
            assert "pick basecamp" in str(you[0].render())

    async def test_empty_feedback_echoes_no_you_box(self, make_crew: MakeCrew) -> None:
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
            await pilot.press("enter")  # empty submit = accept as-is
            worker.join(timeout=2)
            await pilot.pause(0.05)
            assert list(app.query(".you-box")) == []

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

    async def test_second_turn_same_agent_shows_only_the_delta(
        self, make_crew: MakeCrew
    ) -> None:
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

    async def test_out_of_range_task_index_leaves_subtitle_blank(
        self, make_crew: MakeCrew
    ) -> None:
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

    async def test_usage_with_no_open_turn_box_is_a_noop(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            await pilot.pause(0.05)
            # No agent turn is open (e.g. a stray callback after the session
            # closed one) - there is nothing to stamp, so it must simply return.
            assert app._turn_box is None
            app._apply_turn_usage(0)
            assert app._turn_box is None


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
            app._tool_finished_ui("mild ~22C; Alfama", from_cache=True, ok=True)
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
            # Through the error bus handler, which pulls .error off the event.
            app._on_tool_error(None, SimpleNamespace(error="boom"))
            await pilot.pause(0.05)
            coll = app.query_one(".tool-call", Collapsible)
            assert "✗" in coll.title
            assert "boom" in str(app.query_one(".tool-out", Static).render())

    async def test_started_lazily_opens_a_turn_when_none_is_open(
        self, make_crew: MakeCrew
    ) -> None:
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

    async def test_bus_handlers_extract_event_fields_and_render(
        self, make_crew: MakeCrew
    ) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            # Fake events with the fields the handlers read; calling on the UI
            # thread dispatches the render directly (no worker bounce).
            app._on_tool_started(None, SimpleNamespace(tool_name="recon", tool_args={"host": "x"}))
            app._on_tool_finished(None, SimpleNamespace(output="found 2 hosts", from_cache=False))
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
            app._tool_finished_ui("orphan", from_cache=False, ok=True)
            await pilot.pause(0.05)
            assert not app.query(".tool-call")

    async def test_prose_after_a_tool_opens_a_fresh_text_block(self, make_crew: MakeCrew) -> None:
        app = CrewAIPipelineTUI(crew=make_crew(), dry_run=True)
        async with app.run_test() as pilot:
            app._open_agent_turn(0)
            await pilot.pause(0.05)
            app._write_agent("thinking")
            app._tool_started_ui("recon", "x")
            app._tool_finished_ui("done", from_cache=False, ok=True)
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
        from crewai.events import ToolUsageStartedEvent, crewai_event_bus

        app = CrewAIPipelineTUI(crew=make_crew())
        async with app.run_test() as pilot:
            assert await _wait_for(
                pilot,
                lambda: app._on_tool_started
                not in crewai_event_bus._sync_handlers.get(ToolUsageStartedEvent, frozenset()),
            )


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
            assert any(
                "agent line" in str(box.render())
                for box in app.query(".agent-text").results(Static)
            )
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
            await app.query_one("#agent-session", VerticalScroll).remove()
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
