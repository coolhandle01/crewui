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
from crewai.events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)

# Re-exported (explicit alias) so the demo can import it from here rather than
# open a second import of this private crewai submodule - app.py is the one seam.
from crewai.events.types.llm_events import (  # pylint: disable=useless-import-alias
    LLMThinkingChunkEvent as LLMThinkingChunkEvent,
)
from rich.errors import MarkupError
from rich.markup import escape
from rich.rule import Rule as RichRule
from rich.text import Text
from textual import events, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Collapsible, Label, RichLog, Rule, Static, TextArea

from crewui._helpers import (
    compact_tokens,
    dispatch_on_ui_thread,
    format_metrics_block,
    format_step_message,
    format_tool_output,
    format_tool_title,
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
            # replace (not insert) so a newline overwrites an active selection,
            # matching how a printable key behaves.
            self.replace("\n", *self.selection)
            return
        await super()._on_key(event)


# Bounds the reasoning buffer (and thus each re-render), matching
# format_tool_output's cap - a streaming provider can emit unbounded thinking.
_REASONING_CAP = 4000


class ReasoningChunk(Message):
    """A chunk of streamed agent reasoning, handed from the LLM worker thread to
    the UI thread. Posted (not call_from_thread'd) so the producer is never
    blocked per token waiting on a UI round-trip."""

    def __init__(self, chunk: str) -> None:
        self.chunk = chunk
        super().__init__()


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
        # crew-task index -> sidebar-row index. Rows exist only for agent-bearing
        # tasks (task_layout skips the rest), but callbacks fire by crew-task
        # index, so the two diverge whenever a task has no agent; this map keeps
        # the running/done marks on the right row.
        self._row_of_task: dict[int, int] = {}
        # Agent Session: each agent turn is a bordered container (.agent-turn)
        # holding the agent's text (an .agent-text Static that output accumulates
        # into) plus any tool-call boxes, mounted in the order they happen.
        # _turn_box is the container currently written into (None -> the next
        # agent write opens a fresh one); _turn_text is the Static within it that
        # text accumulates into (None -> the next write mounts a fresh one, so a
        # tool box between two runs of text does not merge them); _turn_lines is
        # that Static's accumulated markup; _current_task_idx names whose
        # role/model titles a freshly opened box (e.g. the re-invoked answer
        # after a human-review round).
        self._turn_box: Vertical | None = None
        self._turn_text: Static | None = None
        self._turn_lines: list[str] = []
        self._current_task_idx: int = 0
        # In-flight tool calls, keyed by (tool_name, args) - agents fire tools in
        # parallel and the event bus dispatches concurrently, so a single slot
        # loses calls. Each key holds a FIFO of (collapsible, body Static) still
        # awaiting their Finished event; a finish pops the matching box and fills
        # it (see _tool_started_ui / _tool_finished_ui).
        self._pending_tools: dict[tuple[str, str], list[tuple[Collapsible, Static]]] = {}
        # The agent's extended thinking, streamed in chunks. _reasoning_body is
        # the Static the current reasoning box accumulates into (None -> the next
        # chunk opens a fresh box); it closes when anything else joins the turn
        # (a tool, the answer, a new turn) so a later thinking run gets its own.
        self._reasoning_body: Static | None = None
        self._reasoning_text: str = ""
        # Per-turn token spend: each agent holds a live token accumulator that
        # crewai's LLM callback ticks up as a task runs, cumulative across that
        # agent's turns. We snapshot it (keyed by agent id) at each task
        # boundary; a turn's own spend is the running total now minus its total
        # at the agent's previous turn.
        self._usage_snapshots: dict[int, tuple[int, int, int]] = {}
        # Running totals for the sidebar metrics, summed from each turn's spend so
        # the panel updates live rather than only at completion.
        self._run_input = 0
        self._run_output = 0
        self._run_cached = 0
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

                for crew_idx, heading, role in task_layout(self._crew.tasks):
                    yield Label(heading, classes="phase-heading")
                    name_lbl = Label(role, classes="task-name")
                    status_lbl = Label("Waiting", classes="task-status")
                    self._row_of_task[crew_idx] = len(self._task_widgets)
                    self._task_widgets.append((name_lbl, status_lbl))
                    yield name_lbl
                    yield status_lbl

                yield Static("", id="metrics")

            with Vertical(id="main"):
                with Vertical(id="messages-pane"):
                    yield Label("Agent Session", classes="pane-title")
                    yield VerticalScroll(id="agent-session")
                    # Wrap the input so a derived theme can inset the box via a
                    # margin on #human-input: a margin on the bare input shrinks
                    # the whole pane (Textual reserves a child's margin from the
                    # pane's content width), which drags the scrollbar out of
                    # alignment. Inside a full-width wrapper the margin is local.
                    with Vertical(id="input-wrap"):
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
            # Populate the sidebar zeroed + running from the off, so the panel
            # reads as live from the first frame rather than blank until the end.
            self._render_run_metrics(status="running")
            self._start_run()

    def on_unmount(self) -> None:
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None

    @work(thread=True)
    def _start_run(self) -> None:
        from crewai.core.providers.human_input import reset_provider, set_provider

        # Mark the run in flight first: on_start (arbitrary host code) and the
        # initial call_from_thread below run before kickoff, and Ctrl+Q in that
        # window must take the break-glass hard-teardown, not the graceful path
        # (which would hang on the uncancellable run once it starts).
        self._pipeline_running = True

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
        # Render tool calls in the agent session. CrewAI emits these around every
        # tool execution regardless of LLM provider - unlike the ReAct step
        # callback, which the native tool-calling path never fires. Registered
        # alongside crewai's own handlers (not via scoped_handlers, which would
        # clear them) and removed in the finally, so a second run - or a second
        # App in one process, as in the tests - never stacks a dead handler.
        crewai_event_bus.register_handler(ToolUsageStartedEvent, self._on_tool_started)
        crewai_event_bus.register_handler(ToolUsageFinishedEvent, self._on_tool_finished)
        crewai_event_bus.register_handler(ToolUsageErrorEvent, self._on_tool_error)
        # Likewise the agent's thinking, streamed as chunks by any provider that
        # surfaces extended reasoning.
        crewai_event_bus.register_handler(LLMThinkingChunkEvent, self._on_thinking_chunk)
        try:
            result = self._crew.kickoff()
            self.call_from_thread(self._on_done, result)
        except Exception as exc:
            # escape the exception text: it can carry brackets (a "[/path]" in
            # the message) that would otherwise raise in the markup renderer and
            # lose the error - keep the [bold red] framing, render exc literally.
            err = f"[bold red]Pipeline error: {escape(str(exc))}[/bold red]"
            self.call_from_thread(self._write_agent, err)
            self.call_from_thread(self._write_crew, err)
        finally:
            self._pipeline_running = False
            reset_provider(token)
            crewai_event_bus.off(ToolUsageStartedEvent, self._on_tool_started)
            crewai_event_bus.off(ToolUsageFinishedEvent, self._on_tool_finished)
            crewai_event_bus.off(ToolUsageErrorEvent, self._on_tool_error)
            crewai_event_bus.off(LLMThinkingChunkEvent, self._on_thinking_chunk)
            # Drop any tool boxes left pending by an aborted run so a second run
            # in the same process starts clean.
            self._pending_tools.clear()

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

    @staticmethod
    def _agent_usage_snapshot(agent: object) -> tuple[tuple[int, int, int], int] | None:
        """Cumulative ``(prompt, completion, cached)`` tokens for ``agent``, plus
        the identity to diff that snapshot against - or ``None`` when the agent
        exposes no usage at all.

        Two accumulators can carry the spend, and which one is live depends on
        the LLM path. A native provider (crewai's Anthropic / Bedrock providers)
        ticks the *LLM instance* (``llm.get_token_usage_summary()``) and leaves
        the agent's ``_token_process`` at zero, because it never runs the token
        callback. The litellm path ticks both; a scripted demo may tick only
        ``_token_process``. So prefer the LLM instance once it has recorded
        anything, and fall back to ``_token_process`` otherwise.

        The returned key is the *source's* identity: an LLM instance is often
        shared across agents, so keying the diff by the instance (not the agent)
        keeps a shared instance's per-turn deltas correct."""
        llm = getattr(agent, "llm", None)
        getter = getattr(llm, "get_token_usage_summary", None)
        if getter is not None:
            summary = getter()
            counts = (
                getattr(summary, "prompt_tokens", 0),
                getattr(summary, "completion_tokens", 0),
                getattr(summary, "cached_prompt_tokens", 0),
            )
            if counts[0] or counts[1]:
                return counts, id(llm)
        proc = getattr(agent, "_token_process", None)
        if proc is None:
            return None
        summary = proc.get_summary()
        return (
            getattr(summary, "prompt_tokens", 0),
            getattr(summary, "completion_tokens", 0),
            getattr(summary, "cached_prompt_tokens", 0),
        ), id(agent)

    def _render_run_metrics(self, status: str) -> None:
        """Refresh the sidebar metrics from the running totals.

        Used for the live states (zeroed at start, growing per turn); the final
        state is rendered separately in ``_on_done`` from the crew's own
        authoritative ``token_usage``. A no-op if the widget is not mounted."""
        cost = (
            self._get_token_cost(self._run_input, self._run_output) if self._get_token_cost else 0.0
        )
        try:
            self.query_one("#metrics", Static).update(
                format_metrics_block(
                    input_tokens=self._run_input,
                    output_tokens=self._run_output,
                    cached_tokens=self._run_cached,
                    total_tokens=self._run_input + self._run_output,
                    estimated_cost_usd=cost,
                    status=status,
                )
            )
        except NoMatches:
            logger.debug("metrics widget not mounted")

    def _apply_turn_usage(self, idx: int) -> None:
        """Stamp task ``idx``'s turn box with *this* turn's real token spend.

        crewai's ``TaskOutput`` carries no per-task usage, but the spend is
        accumulated live as the task runs (see ``_agent_usage_snapshot`` for the
        two accumulators and which path ticks which). Those totals are
        cumulative, so the turn's own spend is a diff against the previous
        snapshot. A no-op when the box is gone or the agent exposes no usage
        (subtitle stays blank)."""
        try:
            agent = self._crew.tasks[idx].agent
        except (IndexError, AttributeError):
            return
        snapshot = self._agent_usage_snapshot(agent)
        if snapshot is None:
            return
        # Store the snapshot *unconditionally*, before the turn-box guard: a
        # human-review gate clears _turn_box (via _mount_note), so gating the
        # snapshot on it would drop that turn's tokens and leak them into the
        # next turn's delta. Only the subtitle render below needs the box.
        now, key = snapshot
        prev = self._usage_snapshots.get(key, (0, 0, 0))
        self._usage_snapshots[key] = now
        inp, out, cached = (now[0] - prev[0], now[1] - prev[1], now[2] - prev[2])
        # Fold this turn's spend into the running totals and refresh the sidebar,
        # so the panel tracks the run live. Clamp negatives (a snapshot should
        # only ever grow) so a stray reset cannot walk a total backwards.
        self._run_input += max(inp, 0)
        self._run_output += max(out, 0)
        self._run_cached += max(cached, 0)
        self._render_run_metrics(status="running")
        if inp <= 0 and out <= 0:
            # Nothing was recorded for this turn (e.g. a step with no LLM call);
            # leave the subtitle blank rather than show a zero rail.
            return
        # up-arrow = input, down-arrow = output, recycle = cached (reused) input.
        # The cached rail is omitted when zero so an uncached turn stays terse.
        parts = [f"↑{compact_tokens(inp)}", f"↓{compact_tokens(out)}"]
        if cached:
            parts.append(f"↻{compact_tokens(cached)}")
        if self._turn_box is not None:
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
        # titles the border and its output streams in below. idx is a crew-task
        # index; the sidebar row is looked up through the map (None for an
        # agent-less task, which has no row).
        self._current_task_idx = idx
        self._open_agent_turn(idx)
        row = self._row_of_task.get(idx)
        if row is not None:
            name_lbl, status_lbl = self._task_widgets[row]
            name_lbl.add_class("running")
            status_lbl.add_class("running")
            status_lbl.update("Running...")

    def _set_task_done(self, idx: int) -> None:
        row = self._row_of_task.get(idx)
        if row is not None:
            name_lbl, status_lbl = self._task_widgets[row]
            name_lbl.remove_class("running")
            name_lbl.add_class("done")
            status_lbl.remove_class("running")
            status_lbl.add_class("done")
            status_lbl.update("Done")
        # Advance by crew-task index (not row) so the chain still reaches the
        # next agent-bearing task across any agent-less one between them.
        next_idx = idx + 1
        if next_idx < len(self._crew.tasks):
            self._set_task_running(next_idx)

    def _on_done(self, result: object) -> None:
        # The run finishing is a system event, not content: mark it with a
        # centred labelled rule, nothing inside. The final deliverable is already
        # rendered in the last agent turn, so repeating it here would be noise.
        self._mount_rule("Pipeline Complete")

        # Hand the result to the host for persistence; a save failure must not
        # take the UI down, so swallow and surface it in the pipeline log.
        if self._on_complete is not None:
            try:
                self._on_complete(result)
            except Exception as exc:
                logger.debug("on_complete callback error: %s", exc)
                self._write_crew(f"[yellow]Metrics error: {escape(str(exc))}[/yellow]")

        usage = getattr(result, "token_usage", None)
        if usage is None:
            # No authoritative totals from the crew: keep the running totals we
            # accumulated per turn, but mark the run done rather than leave the
            # sidebar stuck on "running".
            self._render_run_metrics(status="done")
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
        """Mount a fresh agent turn: a bordered container titled with the agent's
        role and model. Text runs and tool-call boxes mount into it as they
        happen (see _write_agent / _tool_started_ui); the container starts empty
        so a turn that opens with a tool has no stray blank line above it."""
        self._turn_box = None
        self._turn_text = None
        self._reasoning_body = None
        # Tool boxes are turn-scoped: any tool still pending from the previous
        # turn (a Started with no Finished - a crash or timeout) will never be
        # filled now, and leaving it keyed would let this turn's retry of the
        # same (name, args) fill the stale box instead of its own.
        self._pending_tools.clear()
        try:
            session = self.query_one("#agent-session", VerticalScroll)
        except NoMatches:
            logger.debug("agent-session not mounted, cannot open a turn")
            return
        self._separate(session)
        box = Vertical(classes="agent-turn")
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
            self._turn_text = None
            return
        box = Static(body, classes=css_class, markup=False)
        box.border_title = title
        session.mount(box)
        self._turn_box = None
        self._turn_text = None
        session.scroll_end(animate=False)

    def _mount_rule(self, label: str) -> None:
        """Mount a centred labelled rule (``-- label --``) and end any in-progress
        turn - a system marker in the flow, carrying no content of its own."""
        try:
            session = self.query_one("#agent-session", VerticalScroll)
        except NoMatches:
            logger.debug("agent-session not mounted, dropping %s rule", label)
            self._turn_box = None
            self._turn_text = None
            return
        session.mount(Static(RichRule(label, style="green"), classes="finish-rule"))
        self._turn_box = None
        self._turn_text = None
        session.scroll_end(animate=False)

    def _write_agent(self, msg: str) -> None:
        if self._turn_box is None:
            self._open_agent_turn(self._current_task_idx)
        if self._turn_box is None:  # session not mounted - nothing to write into
            return
        # Prose joining the turn ends any open thinking run.
        self._reasoning_body = None
        if self._turn_text is None:
            # First prose after a tool box: start a fresh Static below it so the
            # text lands under the tool call, not merged into the run above it.
            self._turn_lines = []
            self._turn_text = Static("", classes="agent-text")
            self._turn_box.mount(self._turn_text)
        self._turn_lines.append(msg)
        joined = "\n\n".join(self._turn_lines)
        # Defence in depth: the buffer accumulates and re-parses every write, so
        # one message with stray markup (an unescaped "[/path]") would otherwise
        # raise here and poison every later write in the turn. Fall back to plain
        # text if the markup does not parse, keeping the turn readable.
        try:
            rendered: Text = Text.from_markup(joined)
        except MarkupError:
            rendered = Text(joined)
        self._turn_text.update(rendered)
        with contextlib.suppress(NoMatches):
            self.query_one("#agent-session", VerticalScroll).scroll_end(animate=False)

    def _write_crew(self, msg: str) -> None:
        try:
            log = self.query_one("#crew-log", RichLog)
        except NoMatches:
            logger.debug("crew-log widget not mounted, dropping message")
            return
        # #crew-log is markup=True, so stray markup in the message (an unescaped
        # "[/path]" from a log record or an error) would raise. Fall back to a
        # plain Text render, mirroring _write_agent, so no caller can crash it.
        try:
            log.write(msg)
        except MarkupError:
            log.write(Text(msg))

    def _on_ui(self, fn: Callable[..., None], *args: object) -> None:
        """Run a UI update from either the worker thread or the UI thread.

        Worker-thread updates go through ``call_from_thread``; a caller already
        on the UI thread (a log record, or a tool event emitted inside a
        UI-thread callback) calls ``fn`` directly, because ``call_from_thread``
        raises when invoked from the app thread.
        """
        if dispatch_on_ui_thread(threading.get_ident(), self._ui_thread_id):
            fn(*args)
        else:
            self.call_from_thread(fn, *args)

    def _ui_dispatch(self, fn: Callable[[str], None], msg: str) -> None:
        """Thread-aware dispatch of a single-string UI update (log records)."""
        self._on_ui(fn, msg)

    def _on_tool_started(self, _source: object, event: object) -> None:
        """Bus handler (worker thread): a tool is about to run - open its box."""
        name = str(getattr(event, "tool_name", "tool"))
        args = getattr(event, "tool_args", "")
        self._on_ui(self._tool_started_ui, name, args)

    def _on_tool_finished(self, _source: object, event: object) -> None:
        """Bus handler (worker thread): a tool returned - fill its box."""
        name = str(getattr(event, "tool_name", "tool"))
        args = getattr(event, "tool_args", "")
        output = getattr(event, "output", "")
        from_cache = bool(getattr(event, "from_cache", False))
        self._on_ui(self._tool_finished_ui, name, args, output, from_cache, True)

    def _on_tool_error(self, _source: object, event: object) -> None:
        """Bus handler (worker thread): a tool raised - close its box as failed."""
        name = str(getattr(event, "tool_name", "tool"))
        args = getattr(event, "tool_args", "")
        error = getattr(event, "error", "")
        self._on_ui(self._tool_finished_ui, name, args, error, False, False)

    @staticmethod
    def _tool_key(name: str, args: object) -> tuple[str, str]:
        """Identity for matching a Finished event to its Started box. Agents fire
        tool calls in parallel and the event bus dispatches them concurrently, so
        a single pending slot loses calls; key by (name, args) instead - distinct
        args (hydrate(cloudflare) vs hydrate(linkedin)) never collide, and repeat
        identical calls queue FIFO under the same key."""
        return (name, str(args))

    def _tool_started_ui(self, name: str, args: object) -> None:
        """Mount a collapsed tool-call box into the current turn, output pending.

        Collapsed by default so a long result does not swamp the session - the
        header carries ``> tool(args)``; expanding it shows the output once the
        matching finished/error event fills it in."""
        if self._turn_box is None:
            self._open_agent_turn(self._current_task_idx)
        if self._turn_box is None:  # session not mounted - nothing to mount into
            return
        body = Static("running...", classes="tool-out", markup=False)
        coll = Collapsible(
            body, title=format_tool_title(name, args), collapsed=True, classes="tool-call"
        )
        self._turn_box.mount(coll)
        self._pending_tools.setdefault(self._tool_key(name, args), []).append((coll, body))
        # Prose after this tool opens a fresh Static below the box (see _write_agent);
        # a tool joining the turn also ends any open thinking run.
        self._turn_text = None
        self._reasoning_body = None
        with contextlib.suppress(NoMatches):
            self.query_one("#agent-session", VerticalScroll).scroll_end(animate=False)

    def _tool_finished_ui(
        self, name: str, args: object, output: object, from_cache: bool, ok: bool
    ) -> None:
        """Fill the matching in-flight tool box with its output and stamp the
        header. Matched by (name, args) so parallel calls each fill their own box;
        a no-op if nothing is pending for that key (a stray or duplicate finish)."""
        key = self._tool_key(name, args)
        waiting = self._pending_tools.get(key)
        if not waiting:
            return
        coll, body = waiting.pop(0)
        if not waiting:  # last one for this key - drop it so the dict stays clean
            del self._pending_tools[key]
        body.update(format_tool_output(output))
        marker = " ✓" if ok else " ✗"
        if from_cache:
            marker += " ⚡"
        coll.title = f"{coll.title}{marker}"
        with contextlib.suppress(NoMatches):
            self.query_one("#agent-session", VerticalScroll).scroll_end(animate=False)

    def _on_thinking_chunk(self, _source: object, event: object) -> None:
        """Bus handler (worker thread): a chunk of the agent's thinking. Posted
        as a message rather than dispatched inline, so a token-level streaming
        provider never blocks its LLM thread on a per-chunk UI round-trip."""
        chunk = str(getattr(event, "chunk", ""))
        if chunk:
            self.post_message(ReasoningChunk(chunk))

    def on_reasoning_chunk(self, message: ReasoningChunk) -> None:
        self._thinking_chunk_ui(message.chunk)

    def _thinking_chunk_ui(self, chunk: str) -> None:
        """Stream a thinking chunk into the turn's reasoning box, opening a
        collapsed one on the first chunk. Folded away by default - reasoning is
        long and secondary; expand it to read how the agent got there.

        The buffer is a single capped string, not a growing list re-joined every
        chunk: appending to a bounded string keeps each update O(cap) rather than
        O(total), so a long stream stays O(n) over the turn instead of O(n^2)."""
        if self._turn_box is None:
            self._open_agent_turn(self._current_task_idx)
        if self._turn_box is None:  # session not mounted - nothing to mount into
            return
        body = self._reasoning_body
        if body is None:
            body = Static("", classes="reasoning-out", markup=False)
            self._reasoning_text = ""
            self._turn_box.mount(
                Collapsible(body, title="reasoning", collapsed=True, classes="reasoning-box")
            )
            self._reasoning_body = body
        if len(self._reasoning_text) > _REASONING_CAP:
            return  # already capped (marked): stop growing and skip the re-render
        combined = self._reasoning_text + chunk
        if len(combined) > _REASONING_CAP:
            combined = combined[:_REASONING_CAP] + " …"
        self._reasoning_text = combined
        body.update(combined)
        with contextlib.suppress(NoMatches):
            self.query_one("#agent-session", VerticalScroll).scroll_end(animate=False)

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
        # 130 (128 + SIGINT), the interrupt convention: break-glass aborts the
        # run, so a wrapper or CI step must not read the exit as a clean 0.
        os._exit(130)  # pragma: no cover

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
        gate = self._feedback_event = threading.Event()
        self._feedback_value = ""
        self.call_from_thread(self._open_feedback_gate)
        # Poll rather than park forever: if the app tears down while the gate is
        # open - any exit that is not a submit - bail with "accept" instead of
        # hanging the worker on wait(), which would wedge teardown on the
        # default-executor join for up to 300s. The is_running check releases the
        # worker within one poll interval of any such exit.
        #
        # Wait on the local `gate`, not self._feedback_event: the submit handler
        # nulls self._feedback_event as it claims the gate, so re-reading it here
        # would race into a None.wait(). The handler sets this same object.
        while not gate.wait(0.25):
            if not self.is_running:
                return ""
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
        # Claim the gate atomically: swap the event out before doing any work, so
        # a duplicate Submitted (a double-tapped or auto-repeated Enter posts one
        # per keypress, and disabling the box does not retract already-queued
        # messages) finds None and returns. Without this the duplicate overwrites
        # _feedback_value - and once the next gate has opened, answers *it* with
        # the previous round's text before the operator sees it.
        gate, self._feedback_event = self._feedback_event, None
        if gate is None:
            return
        # Set the event in a finally: it is the only wakeup for the parked
        # worker, so a raise while echoing or clearing the box (e.g. query_one)
        # must not strand it on wait() forever.
        try:
            # Strip so a whitespace-only submit reads as "accept" to CrewAI too,
            # not as real feedback - otherwise the box echoes "Accepted" while
            # CrewAI runs another review round on the blank text.
            self._feedback_value = event.value.strip()
            # Echo the operator's turn so every round is visible: feedback
            # verbatim, or "Accepted" when they submit empty (accept as-is).
            self._mount_note("you-box", "you", self._feedback_value or "Accepted")
            inp = self.query_one("#human-input", FeedbackArea)
            inp.text = ""
            inp.disabled = True
            inp.placeholder = "Human review (idle)"
        finally:
            gate.set()


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
        # A formatted log record is never intended as markup, and third-party
        # libraries log bracketed paths / URLs freely - escape so a "[/path]"
        # token renders literally instead of raising in the markup=True panes.
        msg = escape(self.format(record))
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
