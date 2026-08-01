"""crewui.app - the Textual App for a sequential CrewAI pipeline.

``CrewAIPipelineTUI`` renders a sidebar task tracker, an agent output log, and
a pipeline log for a sequential crew. The host builds the crew and hands it in,
so the TUI stays out of crew construction.

Human review (``Task(human_input=True)``) is handled by routing CrewAI's
feedback prompt to the input box instead of a blocking terminal ``input()`` -
see ``_make_tui_human_input_provider`` and ``_await_feedback``.

The sidebar title is the host-supplied ``pipeline_name``; ``record_prefix`` is
used only to route log records to the agent vs pipeline pane. The sidebar reads
each task's display name (``Task.name``) and agent role straight off
``crew.tasks``, so the caller only wires the crew - no separate task map.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import psutil
from crewai import Crew
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Label, RichLog, Rule, Static, TextArea

from crewui._helpers import (
    compact_tokens,
    dispatch_on_ui_thread,
    format_metrics_block,
    format_step_message,
    route_log_record,
    task_layout,
)

if TYPE_CHECKING:
    from crewai.core.providers.human_input import SyncHumanInputProvider

logger = logging.getLogger(__name__)


class FeedbackArea(TextArea):
    """Multi-line human-review input for the feedback gate.

    Enter submits, matching CrewAI's out-of-the-box feedback prompt; Ctrl+J or
    Shift+Enter inserts a newline for multi-paragraph feedback. This inverts
    TextArea's default (where Enter inserts the newline). An empty submit
    accepts the result as-is, per CrewAI's feedback loop.

    Ctrl+J is the portable newline: it is a raw line-feed byte, so it reaches
    the app on every terminal. Shift+Enter only arrives on terminals whose
    keyboard protocol distinguishes it from Enter (not Windows Terminal / WSL,
    for example); where it does not, Ctrl+J still works and Enter still submits.
    """

    _NEWLINE_KEYS = frozenset({"ctrl+j", "shift+enter"})

    class Submitted(Message):
        """Posted when the operator submits their feedback with Enter."""

        def __init__(self, value: str) -> None:
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if event.key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)


class CrewAIPipelineTUI(App[None]):
    # The class owns the default theme. Absolute (not the bare "default.tcss")
    # because Textual resolves a relative CSS_PATH against the *concrete*
    # class's module file - so a derived class would otherwise look for the
    # stylesheet next to its own module. A derived class overrides the theme
    # by setting its own CSS_PATH.
    CSS_PATH = str(Path(__file__).parent / "default.tcss")

    def __init__(
        self,
        crew: Crew,
        record_prefix: str = "pipeline",
        pipeline_name: str = "",
        dry_run: bool = False,
        on_start: Callable[[], None] | None = None,
        on_complete: Callable[[object], None] | None = None,
        get_token_cost: Callable[[int, int], float] | None = None,
    ) -> None:
        super().__init__()
        self._crew = crew
        self._record_prefix = record_prefix
        self._pipeline_name = pipeline_name
        self._dry_run = dry_run
        self._on_start = on_start
        self._on_complete = on_complete
        self._get_token_cost = get_token_cost
        self._task_widgets: list[tuple[Label, Label]] = []
        # Agent Session: each agent turn is a bordered Static we accumulate into.
        # _turn_box is the box currently streamed into (None -> the next agent
        # write opens a fresh one); _turn_lines is its accumulated markup; and
        # _current_task_idx names whose role/model titles a freshly opened box
        # (e.g. the re-invoked answer after a human-review round).
        self._turn_box: Static | None = None
        self._turn_lines: list[str] = []
        self._current_task_idx: int = 0
        # Per-turn token spend: each agent holds a live token accumulator that
        # crewai's LLM callback ticks up as a task runs, cumulative across that
        # agent's turns. We snapshot it (keyed by agent id) at each task
        # boundary; a turn's own spend is the running total now minus its total
        # at the agent's previous turn.
        self._usage_snapshots: dict[int, tuple[int, int, int]] = {}
        # Human-review bridge: the worker thread parks on this event while the
        # operator types feedback into the input box on the UI thread.
        self._feedback_event: threading.Event | None = None
        self._feedback_value: str = ""
        self._log_handler: _TUILogHandler | None = None
        # Captured in on_mount (which runs on the UI thread). Lets log records
        # emitted during a UI-thread callback dispatch directly instead of
        # bouncing through call_from_thread, which refuses same-thread calls.
        self._ui_thread_id: int | None = None
        # True while kickoff() is in flight. Ctrl+Q consults it: a running
        # kickoff cannot be cancelled, so quitting mid-run must break the glass
        # (see action_quit) rather than hang on the worker join.
        self._pipeline_running: bool = False
        self._crew.step_callback = self._make_step_callback()

    def compose(self) -> ComposeResult:
        with Horizontal():
            with Vertical(id="sidebar"):
                yield Label(self._pipeline_name, id="sidebar-title")

                for heading, role in task_layout(self._crew.tasks):
                    yield Label(heading, classes="phase-heading")
                    name_lbl = Label(role, classes="task-name")
                    status_lbl = Label("Waiting", classes="task-status")
                    self._task_widgets.append((name_lbl, status_lbl))
                    yield name_lbl
                    yield status_lbl

                yield Static("", id="metrics")

            with Vertical(id="main"):
                with Vertical(id="messages-pane"):
                    yield Label("Agent Session", classes="pane-title")
                    yield VerticalScroll(id="agent-session")
                    yield FeedbackArea(
                        "",
                        id="human-input",
                        soft_wrap=True,
                        show_line_numbers=False,
                        disabled=True,
                        placeholder="Human review (idle)",
                    )
                with Vertical(id="logs-pane"):
                    yield Label("Pipeline Logs", id="logs-title", classes="pane-title")
                    yield RichLog(id="crew-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        # Capture the UI thread id before the log handler is registered, so no
        # record can route (and consult _ui_dispatch) before it is set.
        self._ui_thread_id = threading.get_ident()
        # Held so on_unmount can detach it. A handler left on the root logger
        # after the app stops would route later records through call_from_thread
        # into a dead app - harmless in a one-shot CLI, a leak anywhere the host
        # keeps running after the TUI closes.
        self._log_handler = _TUILogHandler(self)
        logging.getLogger().addHandler(self._log_handler)
        # The feedback box is the operator's turn; label its border "you" so a
        # submitted round reads like a chat turn beside the agent boxes.
        with contextlib.suppress(NoMatches):
            self.query_one("#human-input", FeedbackArea).border_title = "you"
        if self._dry_run:
            self._write_crew("[yellow]Dry run mode: pipeline not started.[/yellow]")
            # Render the metrics block zeroed so the sidebar reads as a complete
            # preview rather than a blank panel - no run happened, so the
            # figures are zero and the status says so.
            self.query_one("#metrics", Static).update(
                format_metrics_block(
                    input_tokens=0,
                    output_tokens=0,
                    cached_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=0.0,
                    status="dry run",
                )
            )
        else:
            self._start_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    @work(thread=True)
    def _start_run(self) -> None:
        from crewai.core.providers.human_input import reset_provider, set_provider

        if self._on_start is not None:
            self._on_start()

        for i, task in enumerate(self._crew.tasks):
            orig: Callable[..., None] | None = task.callback
            task.callback = self._make_task_callback(i, orig)

        self.call_from_thread(self._set_task_running, 0)

        # Route CrewAI's human_input feedback prompt to the input box instead of
        # a blocking terminal input(). Set in this (worker) thread so kickoff's
        # get_provider() - same thread - picks it up; reset when the run ends.
        token = set_provider(_make_tui_human_input_provider(self))
        self._pipeline_running = True
        try:
            result = self._crew.kickoff()
            self.call_from_thread(self._on_done, result)
        except Exception as exc:
            self.call_from_thread(self._write_agent, f"[bold red]Pipeline error: {exc}[/bold red]")
            self.call_from_thread(self._write_crew, f"[bold red]Pipeline error: {exc}[/bold red]")
        finally:
            self._pipeline_running = False
            reset_provider(token)

    def _make_task_callback(
        self, idx: int, orig: Callable[..., None] | None
    ) -> Callable[..., None]:
        def _cb(output: object) -> None:
            # Stamp the turn's token spend on its box subtitle before the sidebar
            # advances (which opens the next box). The spend is read from the
            # agent's live accumulator, not the output - crewai's TaskOutput
            # carries no per-task usage.
            self.call_from_thread(self._apply_turn_usage, idx)
            self.call_from_thread(self._set_task_done, idx)
            if orig is not None:
                orig(output)

        return _cb

    def _apply_turn_usage(self, idx: int) -> None:
        """Stamp task ``idx``'s turn box with *this* turn's real token spend.

        crewai's ``TaskOutput`` carries no per-task usage, but every agent holds
        a live ``_token_process`` accumulator that the LLM callback ticks up as
        the task runs; this fires after the agent finishes (see
        ``Task._execute_core``), so the total is complete by now. One agent may
        run several tasks and the accumulator is cumulative, so the turn's own
        spend is a per-agent diff against the previous snapshot. A no-op when the
        box is gone or the agent exposes no accumulator (subtitle stays blank)."""
        if self._turn_box is None:
            return
        try:
            agent = self._crew.tasks[idx].agent
        except (IndexError, AttributeError):
            return
        proc = getattr(agent, "_token_process", None)
        if proc is None:
            return
        summary = proc.get_summary()
        now = (
            getattr(summary, "prompt_tokens", 0),
            getattr(summary, "completion_tokens", 0),
            getattr(summary, "cached_prompt_tokens", 0),
        )
        prev = self._usage_snapshots.get(id(agent), (0, 0, 0))
        self._usage_snapshots[id(agent)] = now
        inp, out, cached = (now[0] - prev[0], now[1] - prev[1], now[2] - prev[2])
        if inp <= 0 and out <= 0:
            # Nothing was recorded for this turn (e.g. a step with no LLM call);
            # leave the subtitle blank rather than show a zero rail.
            return
        # up-arrow = input, down-arrow = output, recycle = cached (reused) input.
        # The cached rail is omitted when zero so an uncached turn stays terse.
        parts = [f"↑{compact_tokens(inp)}", f"↓{compact_tokens(out)}"]
        if cached:
            parts.append(f"↻{compact_tokens(cached)}")
        self._turn_box.border_subtitle = " · ".join(parts)

    def _make_step_callback(self) -> Callable[[object], None]:
        def _cb(step: object) -> None:
            try:
                msg = format_step_message(step)
            except Exception as exc:
                logger.debug("step callback error: %s", exc)
                return
            self.call_from_thread(self._write_agent, msg)

        return _cb

    def _set_task_running(self, idx: int) -> None:
        # A running task is a fresh agent turn: open its box so its role/model
        # titles the border and its output streams in below.
        self._current_task_idx = idx
        self._open_agent_turn(idx)
        if idx < len(self._task_widgets):
            name_lbl, status_lbl = self._task_widgets[idx]
            name_lbl.add_class("running")
            status_lbl.add_class("running")
            status_lbl.update("Running...")

    def _set_task_done(self, idx: int) -> None:
        if idx < len(self._task_widgets):
            name_lbl, status_lbl = self._task_widgets[idx]
            name_lbl.remove_class("running")
            name_lbl.add_class("done")
            status_lbl.remove_class("running")
            status_lbl.add_class("done")
            status_lbl.update("Done")
        next_idx = idx + 1
        if next_idx < len(self._task_widgets):
            self._set_task_running(next_idx)

    def _on_done(self, result: object) -> None:
        raw = getattr(result, "raw", str(result))
        # The run finishing is a system event, not the last agent still talking,
        # so render it as a standalone note (like the review gate) rather than
        # appending into the final agent's box. This also ends that turn. Body is
        # the full deliverable, not truncated - the session scrolls.
        self._mount_note("done-box", "Pipeline Complete", raw)

        # Hand the result to the host for persistence; a save failure must not
        # take the UI down, so swallow and surface it in the pipeline log.
        if self._on_complete is not None:
            try:
                self._on_complete(result)
            except Exception as exc:
                logger.debug("on_complete callback error: %s", exc)
                self._write_crew(f"[yellow]Metrics error: {exc}[/yellow]")

        usage = getattr(result, "token_usage", None)
        if usage is None:
            return

        input_tokens = getattr(usage, "prompt_tokens", 0)
        output_tokens = getattr(usage, "completion_tokens", 0)
        cached_tokens = getattr(usage, "cached_prompt_tokens", 0)
        cost = self._get_token_cost(input_tokens, output_tokens) if self._get_token_cost else 0.0
        try:
            self.query_one("#metrics", Static).update(
                format_metrics_block(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=getattr(usage, "total_tokens", input_tokens + output_tokens),
                    estimated_cost_usd=cost,
                )
            )
        except NoMatches:
            logger.debug("metrics widget not mounted")

    def _agent_label(self, idx: int) -> str:
        """``role · model`` for task ``idx``'s agent, best-effort.

        Falls back to the role alone (or ``"Agent"``) when the model or agent
        is not introspectable - a partially-wired crew must never break a box.
        """
        try:
            agent = self._crew.tasks[idx].agent
        except (IndexError, AttributeError):
            return "Agent"
        role = getattr(agent, "role", None) or "Agent"
        model = getattr(getattr(agent, "llm", None), "model", None)
        return f"{role} · {model}" if model else role

    def _separate(self, session: VerticalScroll) -> None:
        """A rule between turns - never above the first one."""
        if session.children:
            session.mount(Rule())

    def _open_agent_turn(self, idx: int) -> None:
        """Mount a fresh agent box, titled with the agent's role and model, and
        make it the box that subsequent output streams into."""
        try:
            session = self.query_one("#agent-session", VerticalScroll)
        except NoMatches:
            logger.debug("agent-session not mounted, cannot open a turn")
            return
        self._separate(session)
        box = Static("", classes="agent-turn")
        box.border_title = self._agent_label(idx)
        session.mount(box)
        self._turn_box = box
        self._turn_lines = []
        session.scroll_end(animate=False)

    def _mount_note(self, css_class: str, title: str, body: str) -> None:
        """Mount a non-agent box - the review prompt, or an echoed operator turn
        - and end any in-progress agent turn so the next agent output opens
        its own fresh box. ``body`` is rendered literally (markup off): operator
        feedback may contain brackets."""
        try:
            session = self.query_one("#agent-session", VerticalScroll)
        except NoMatches:
            logger.debug("agent-session not mounted, dropping %s box", css_class)
            self._turn_box = None
            return
        box = Static(body, classes=css_class, markup=False)
        box.border_title = title
        session.mount(box)
        self._turn_box = None
        session.scroll_end(animate=False)

    def _write_agent(self, msg: str) -> None:
        if self._turn_box is None:
            self._open_agent_turn(self._current_task_idx)
        if self._turn_box is None:  # session not mounted - nothing to write into
            return
        self._turn_lines.append(msg)
        self._turn_box.update(Text.from_markup("\n\n".join(self._turn_lines)))
        with contextlib.suppress(NoMatches):
            self.query_one("#agent-session", VerticalScroll).scroll_end(animate=False)

    def _write_crew(self, msg: str) -> None:
        try:
            self.query_one("#crew-log", RichLog).write(msg)
        except NoMatches:
            logger.debug("crew-log widget not mounted, dropping message")

    def _ui_dispatch(self, fn: Callable[[str], None], msg: str) -> None:
        """Run a UI update, from either the worker thread or the UI thread.

        Worker-thread updates go through ``call_from_thread``; a caller already
        on the UI thread (a log record emitted inside a UI callback) calls
        ``fn`` directly, because ``call_from_thread`` raises when invoked from
        the app thread.
        """
        if dispatch_on_ui_thread(threading.get_ident(), self._ui_thread_id):
            fn(msg)
        else:
            self.call_from_thread(fn, msg)

    # -- teardown --

    async def action_quit(self) -> None:
        """Ctrl+Q. Graceful when idle, break-glass when a run is in flight.

        A running ``kickoff()`` cannot be cancelled, so a normal quit would hang
        asyncio's default-executor join on the abandoned worker thread (up to
        300s) as the app tears down. When a run is in flight this is the panic
        key: restore the terminal, kill the run's child processes, and hard-exit
        so nothing wedges the exit or keeps running against the host's targets.
        When nothing is running the graceful path is safe - the worker has
        finished, so the host tears down and exits with no hang.
        """
        if not self._pipeline_running:
            self.exit()
            return
        # In-flight, uncancellable run: hard teardown. Ends in os._exit, so it
        # is unreachable in-process - exercised by the subprocess smoke test.
        self._restore_terminal()  # pragma: no cover
        self._kill_run_children()  # pragma: no cover
        os._exit(0)  # pragma: no cover

    def _restore_terminal(self) -> None:
        """Put the terminal back before a hard exit.

        ``os._exit`` skips Textual's own driver teardown, which would otherwise
        leave the terminal in raw / alt-screen mode - a different kind of mess.
        """
        driver = getattr(self, "_driver", None)
        if driver is not None:
            try:
                driver.stop_application_mode()
            except Exception as exc:  # best-effort during panic teardown
                logger.debug("terminal restore failed during break-glass: %s", exc)

    @staticmethod
    def _kill_run_children() -> None:
        """Terminate every descendant process on a break-glass quit.

        A run's children are the subprocesses it spawned - MCP servers and any
        external tool the crew invoked. Hard-exiting without this would orphan
        them. SIGTERM first, then SIGKILL the stragglers. psutil scopes this to
        our own process tree (and to the container's PID namespace when
        containerised).
        """
        try:
            children = psutil.Process().children(recursive=True)
        except psutil.Error:
            return
        for child in children:
            with contextlib.suppress(psutil.Error):
                child.terminate()
        _, alive = psutil.wait_procs(children, timeout=1.5)
        for child in alive:
            with contextlib.suppress(psutil.Error):
                child.kill()

    # -- human review --

    def _await_feedback(self) -> str:
        """Worker-thread side of the human-review gate.

        Opens the input box on the UI thread, parks until the operator submits,
        then returns what they typed. Empty (just Enter) means "accept", per
        CrewAI's feedback loop.
        """
        self._feedback_event = threading.Event()
        self._feedback_value = ""
        self.call_from_thread(self._open_feedback_gate)
        self._feedback_event.wait()
        return self._feedback_value

    def _open_feedback_gate(self) -> None:
        # A bordered box in the session makes the gate unmissable and ends the
        # agent's turn; the input box below is where the operator answers.
        self._mount_note(
            "gate-box",
            "Human Review Requested",
            "Review the result above.\n\n"
            "Type your feedback, then press Enter to submit "
            "(Ctrl+J for a newline).\n"
            "Submit an empty box to accept the result as-is.",
        )
        inp = self.query_one("#human-input", FeedbackArea)
        inp.placeholder = "Enter submits - Ctrl+J for a newline - empty accepts"
        inp.disabled = False
        inp.focus()

    def on_feedback_area_submitted(self, event: FeedbackArea.Submitted) -> None:
        if self._feedback_event is None:
            return
        self._feedback_value = event.value
        # Echo the operator's turn into the session so a submitted round is
        # visible; an empty submit ("accept as-is") has nothing to show.
        if event.value.strip():
            self._mount_note("you-box", "you", event.value)
        inp = self.query_one("#human-input", FeedbackArea)
        inp.text = ""
        inp.disabled = True
        inp.placeholder = "Human review (idle)"
        self._feedback_event.set()


def _make_tui_human_input_provider(app: CrewAIPipelineTUI) -> SyncHumanInputProvider:
    """Build a CrewAI human-input provider that routes the feedback prompt to
    the TUI input box instead of a blocking terminal ``input()``.

    Isolated here, with a deferred import, because it leans on crewai's
    semi-internal provider API (``crewai.core.providers.human_input``). That is
    the sanctioned injection point but may move between versions - this is the
    one place to fix if it does. ``output_to_review`` is accepted (crewai passes the
    answer under review) but unused: the step-callback already streams that
    answer into the agent-log pane before the gate opens.
    """
    from crewai.core.providers.human_input import SyncHumanInputProvider

    # crewai ships no stubs, so SyncHumanInputProvider is Any and mypy --strict
    # rejects subclassing it; the subclass is the sanctioned injection point.
    class _TUIHumanInputProvider(SyncHumanInputProvider):  # type: ignore[misc]
        @staticmethod
        def _prompt_input(crew: Crew | None = None, output_to_review: str | None = None) -> str:
            return app._await_feedback()

    return _TUIHumanInputProvider()


class _TUILogHandler(logging.Handler):
    def __init__(self, app: CrewAIPipelineTUI) -> None:
        super().__init__()
        self._app = app

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        target = route_log_record(record.name, self._app._record_prefix)
        # A record can be emitted from the worker thread (during kickoff) or
        # from the UI thread (a widget callback that logs) - _ui_dispatch picks
        # the right hand-off for the current thread. The human-review gate is
        # the UI-thread case: opening the input box logs, and a naive
        # call_from_thread there would crash the run.
        if target == "agent":
            self._app._ui_dispatch(self._app._write_agent, msg)
        else:
            self._app._ui_dispatch(self._app._write_crew, msg)
