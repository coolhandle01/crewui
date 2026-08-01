"""tests/test_helpers.py - branch-coverage of the pure helpers extracted from
the CrewAIPipelineTUI class.

The Textual App / widget / threading layer in crewui/app.py is exercised by the
pilot-harness tests in test_app.py; the helpers here are pure functions so
every conditional path can be covered by ordinary unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace

from crewai.agents.parser import AgentAction, AgentFinish

from crewui._helpers import (
    dispatch_on_ui_thread,
    format_metrics_block,
    format_step_message,
    format_tool_title,
    route_log_record,
    task_layout,
    truncate,
)


def _make_action(
    tool: str = "recon",
    tool_input: str = "example.com",
    thought: str = "planning",
    result: str | None = "found 2 hosts",
) -> AgentAction:
    """Construct a real AgentAction for tests - it's a Pydantic model, no
    LLM call or token cost involved."""
    return AgentAction(thought=thought, tool=tool, tool_input=tool_input, text="", result=result)


class TestTruncate:
    def test_returns_text_unchanged_when_under_limit(self) -> None:
        assert truncate("hello", 10) == "hello"

    def test_returns_text_unchanged_when_at_limit(self) -> None:
        assert truncate("hello", 5) == "hello"

    def test_truncates_when_over_limit(self) -> None:
        assert truncate("hello world", 5) == "hello"


class TestRouteLogRecord:
    def test_routes_to_agent_when_record_starts_with_prefix(self) -> None:
        assert route_log_record("crewui.demo.scout", "crewui.demo") == "agent"

    def test_routes_to_crew_when_record_does_not_start_with_prefix(self) -> None:
        assert route_log_record("urllib3.connectionpool", "crewui.demo") == "crew"

    def test_empty_prefix_routes_everything_to_agent(self) -> None:
        # Every string starts with "" so the prefix-empty case lands on agent.
        assert route_log_record("anything", "") == "agent"


class TestDispatchOnUiThread:
    """Guards the human-review-gate crash: a log record emitted while already
    on the UI thread must dispatch directly, because Textual's
    ``call_from_thread`` refuses same-thread calls."""

    def test_same_thread_dispatches_directly(self) -> None:
        # Caller thread == the app's captured UI thread -> call fn directly.
        assert dispatch_on_ui_thread(42, 42) is True

    def test_worker_thread_uses_call_from_thread(self) -> None:
        # Caller thread != UI thread (a worker) -> bounce via call_from_thread.
        assert dispatch_on_ui_thread(7, 42) is False

    def test_unmounted_app_uses_call_from_thread(self) -> None:
        # Before on_mount captures the id there is no UI-thread code running.
        assert dispatch_on_ui_thread(42, None) is False


def _task(name: str | None, role: str | None) -> SimpleNamespace:
    """Stand-in for a crewai.Task: only ``.name`` and ``.agent.role`` are read
    by ``task_layout``. ``role=None`` models a task with no assigned agent."""
    agent = SimpleNamespace(role=role) if role is not None else None
    return SimpleNamespace(name=name, agent=agent)


class TestTaskLayout:
    def test_empty_input_yields_empty_layout(self) -> None:
        assert task_layout([]) == []

    def test_uses_task_name_as_heading_and_role_as_row(self) -> None:
        assert task_layout([_task("Reconnaissance", "scout")]) == [("Reconnaissance", "scout")]

    def test_one_agent_two_tasks_keep_distinct_headings(self) -> None:
        # One agent runs two phases: same role, distinct per-task names.
        layout = task_layout(
            [
                _task("Research", "analyst"),
                _task("Triage", "analyst"),
            ]
        )
        assert layout == [("Research", "analyst"), ("Triage", "analyst")]

    def test_missing_name_falls_back_to_role(self) -> None:
        # A task with no name (None) uses the agent role as the heading.
        assert task_layout([_task(None, "scribe")]) == [("scribe", "scribe")]

    def test_task_without_agent_is_skipped(self) -> None:
        layout = task_layout([_task("Orphan", None), _task("Recon", "scout")])
        assert layout == [("Recon", "scout")]


class TestFormatMetricsBlock:
    def test_splits_tokens_and_renders_separators_and_decimals(self) -> None:
        block = format_metrics_block(
            input_tokens=9000,
            output_tokens=3345,
            cached_tokens=512,
            total_tokens=12345,
            estimated_cost_usd=0.0418,
        )
        assert " Input:   9,000" in block
        assert " Output:  3,345" in block
        assert " Cached:  512" in block
        assert " Total:   12,345" in block
        assert " Cost:    $0.0418" in block
        assert " Status:  done" in block

    def test_status_override_renders_custom_status(self) -> None:
        # The dry-run sidebar renders the block zeroed with a "dry run" status.
        block = format_metrics_block(
            input_tokens=0,
            output_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            status="dry run",
        )
        assert " Total:   0" in block
        assert " Cost:    $0.0000" in block
        assert " Status:  dry run" in block


class TestCompactTokens:
    def test_compacts_thousands_and_leaves_small_counts(self) -> None:
        from crewui._helpers import compact_tokens

        assert compact_tokens(1200) == "1.2k"
        assert compact_tokens(1610) == "1.6k"
        assert compact_tokens(340) == "340"
        assert compact_tokens(0) == "0"


class TestFormatStepMessage:
    def test_agent_action_with_result_includes_thought_tool_call_and_result(self) -> None:
        msg = format_step_message(_make_action())
        assert "[yellow]Thought:[/yellow] planning" in msg
        assert "[cyan]> recon[/cyan](example.com)" in msg
        assert "[dim]found 2 hosts[/dim]" in msg

    def test_agent_action_without_result_omits_result_block(self) -> None:
        msg = format_step_message(_make_action(result=""))
        assert "[dim]" not in msg

    def test_agent_action_long_inputs_are_truncated(self) -> None:
        msg = format_step_message(_make_action(tool_input="x" * 500, result="y" * 1000))
        # tool_input clipped to 120 chars inside the parens
        assert "[cyan]> recon[/cyan](" + "x" * 120 + ")" in msg
        # result clipped to 300 chars inside [dim]...[/dim]
        assert msg.endswith("[dim]" + "y" * 300 + "[/dim]")

    def test_agent_finish_returns_answer_in_full(self) -> None:
        # The answer is the operator's review target at a human_input gate, so
        # it is rendered in full - not clipped like intermediate tool progress.
        finish = AgentFinish(thought="done", output="y" * 700, text="t")
        msg = format_step_message(finish)
        assert msg.startswith("[bold green]Answer:[/bold green] ")
        assert msg.count("y") == 700

    def test_other_step_type_returns_truncated_repr(self) -> None:
        msg = format_step_message("random output " + "z" * 500)
        assert msg.startswith("random output ")
        assert len(msg) == 300


class TestFormatToolTitle:
    def test_dict_args_render_as_key_value_pairs(self) -> None:
        title = format_tool_title("search_web", {"q": "lisbon", "n": 3})
        assert title == "> search_web(q=lisbon, n=3)"

    def test_string_args_pass_through(self) -> None:
        assert format_tool_title("recon", "example.com") == "> recon(example.com)"

    def test_long_args_are_clipped_to_keep_the_header_short(self) -> None:
        title = format_tool_title("t", "x" * 200)
        # Clipped to 80 chars inside the parens; the full input is in the body.
        assert title == "> t(" + "x" * 80 + ")"
