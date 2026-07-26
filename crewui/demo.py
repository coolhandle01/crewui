"""crewui.demo - a self-contained, offline demonstration of the TUI.

``crewui demo`` builds a small three-phase sequential crew and drives it through
``CrewAIPipelineTUI`` without any LLM call or network access. It exists so that
``pipx install crewui`` yields something immediately runnable - a live look at
the sidebar tracker, the agent-output stream, and the pipeline log - and so the
release smoke test has an entry point that exercises the real App end to end.

The crew is a genuine ``crewai.Crew`` of genuine ``Agent`` / ``Task`` objects
(so the sidebar reads real ``Task.name`` / ``agent.role`` values), but its
``kickoff`` is replaced with a scripted walk that emits canned step messages
and fires each task callback in turn. That keeps the demo deterministic and
free of API keys while still driving every code path the App uses to render a
run: step callbacks, per-task status transitions, and the final result +
token-usage block.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass

from crewai import Agent, Crew, Process, Task
from crewai.agents.parser import AgentAction, AgentFinish

from crewui.app import CrewAIPipelineTUI

# The demo's scripted phases: (task name, agent role, thought, tool, tool input,
# tool result, final answer). Nothing here is sent anywhere - it is canned prose
# rendered into the panes so the layout and status flow are visible.
_PHASES = [
    (
        "Reconnaissance",
        "Scout",
        "Map the surface before touching anything.",
        "enumerate",
        "example.com",
        "found 3 subdomains, 2 open ports",
        "Surface mapped: api.example.com, www.example.com, mail.example.com.",
    ),
    (
        "Analysis",
        "Analyst",
        "Weigh what the scout turned up.",
        "assess",
        "3 subdomains",
        "api.example.com exposes an unauthenticated debug route",
        "One notable exposure on api.example.com; the rest look routine.",
    ),
    (
        "Report",
        "Scribe",
        "Write it up so a human can act on it.",
        "compose",
        "1 finding",
        "drafted a 1-paragraph summary",
        "Report ready: 1 finding worth a closer look on api.example.com.",
    ),
]


@dataclass
class _Usage:
    """Stand-in for CrewAI's token-usage object; only these three fields are
    read by the App's metrics block."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _Result:
    """Stand-in for a CrewOutput: the App reads ``.raw`` and ``.token_usage``."""

    raw: str
    token_usage: _Usage


def _scripted_kickoff(crew: Crew) -> _Result:
    """Walk the crew's tasks without calling an LLM.

    For each task: stream a Thought + tool-call and an Answer through the crew's
    ``step_callback`` (both set by the App before this runs), pause briefly so
    the transition is visible, then fire the task's ``callback`` to advance the
    sidebar. Returns a canned result carrying ``raw`` + ``token_usage`` so the
    App renders the completion panel and metrics block.
    """
    for task, phase in zip(crew.tasks, _PHASES, strict=True):
        _, _, thought, tool, tool_input, tool_result, answer = phase
        if crew.step_callback is not None:
            crew.step_callback(
                AgentAction(
                    thought=thought, tool=tool, tool_input=tool_input, text="", result=tool_result
                )
            )
        time.sleep(0.6)
        if crew.step_callback is not None:
            crew.step_callback(AgentFinish(thought=thought, output=answer, text=answer))
        if task.callback is not None:
            task.callback(answer)
        time.sleep(0.4)

    return _Result(
        raw="Demo pipeline complete - 3 phases, 1 finding.",
        token_usage=_Usage(prompt_tokens=1200, completion_tokens=340, total_tokens=1540),
    )


def build_demo_crew() -> Crew:
    """Build the offline demo crew with a scripted ``kickoff``.

    A dummy ``OPENAI_API_KEY`` is set only if none is present so that
    constructing the agents never prompts; no request is ever made because
    ``kickoff`` is replaced before the App can call it.
    """
    os.environ.setdefault("OPENAI_API_KEY", "crewui-demo-no-network")

    agents = []
    tasks = []
    for name, role, *_ in _PHASES:
        agent = Agent(
            role=role,
            goal=f"{role} phase of the demo pipeline",
            backstory=f"The {role.lower()} in a small demonstration crew.",
            llm="gpt-4o-mini",
            verbose=False,
        )
        agents.append(agent)
        tasks.append(
            Task(
                description=f"{name} phase",
                expected_output="a short summary",
                agent=agent,
                name=name,
            )
        )

    crew = Crew(agents=agents, tasks=tasks, process=Process.sequential)
    # Replace kickoff with the scripted walk. Crew is a pydantic model, so a
    # plain attribute assignment is rejected; object.__setattr__ installs the
    # bound method past that guard. This is the whole reason the demo can run
    # with no API key and no network.
    object.__setattr__(crew, "kickoff", types.MethodType(_scripted_kickoff, crew))
    return crew


def run_demo(dry_run: bool = False) -> None:
    """Launch the TUI against the offline demo crew."""
    CrewAIPipelineTUI(
        crew=build_demo_crew(),
        record_prefix="crewui.demo",
        pipeline_name="crewui demo pipeline",
        dry_run=dry_run,
        get_token_cost=lambda inp, out: (inp * 3 + out * 15) / 1_000_000,
    ).run()
