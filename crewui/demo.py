"""crewui.demo - a self-contained, offline demonstration of the TUI.

``crewui demo`` builds a small three-phase sequential crew and drives it through
``CrewAIPipelineTUI`` without any LLM call or network access. It exists so that
``pipx install crewui`` yields something immediately runnable - a live look at
the agent session, the sidebar tracker, and the pipeline log - and so the
release smoke test has an entry point that exercises the real App end to end.

The pipeline is a domain-neutral one (planning a long weekend), deliberately
*not* tied to any downstream app: crewui is a UI for CrewAI pipelines in
general, so its demo reads like anyone's crew, not one particular product's.

The crew is a genuine ``crewai.Crew`` of genuine ``Agent`` / ``Task`` objects
(so the sidebar reads real ``Task.name`` / ``agent.role`` values), but its
``kickoff`` is replaced with a scripted walk that emits canned step messages
and fires each task callback in turn. That keeps the demo deterministic and
free of API keys while still driving every code path the App uses to render a
run: step callbacks, per-task status transitions, per-turn token subtitles,
and the final result + token-usage block.

Per-turn token counts are faked here because crewai's ``TaskOutput`` carries no
per-task usage - so on a *real* run the App leaves each box's subtitle blank,
while the demo supplies one so the feature is visible.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass
from typing import NamedTuple

from crewai import Agent, Crew, Process, Task
from crewai.agents.parser import AgentAction, AgentFinish

from crewui.app import CrewAIPipelineTUI


class _Phase(NamedTuple):
    """One scripted turn: the prose rendered into the panes, plus the per-turn
    token counts hung on the box's subtitle. Nothing is sent anywhere."""

    name: str
    role: str
    thought: str
    tool: str
    tool_input: str
    tool_result: str
    answer: str
    input_tokens: int
    output_tokens: int


# The model shown on each agent box's title rail. The demo never calls it
# (kickoff is scripted), so this is display-only - see build_demo_crew for why
# it is a relabel rather than the constructed provider.
_DEMO_MODEL = "claude-sonnet-5"


# "Plan a long weekend in Lisbon" - a neutral, widely legible pipeline in the
# shape CrewAI's own examples use (research -> plan -> cost).
_PHASES = [
    _Phase(
        name="Destination Research",
        role="Destination Researcher",
        thought="Get the lay of the land before planning any days.",
        tool="search_web",
        tool_input="Lisbon, long weekend in October",
        tool_result="mild ~22C; Alfama, Belem, Sintra; trams busy midday",
        answer="Lisbon in October is warm and walkable. Anchors: Alfama, Belem, a Sintra day trip.",
        input_tokens=1180,
        output_tokens=260,
    ),
    _Phase(
        name="Itinerary",
        role="Itinerary Planner",
        thought="Turn the highlights into a sane day-by-day plan.",
        tool="build_itinerary",
        tool_input="3 days: Alfama, Belem, Sintra",
        tool_result="Fri Alfama + Fado; Sat Belem + Time Out Market; Sun Sintra",
        answer="Fri: Alfama + Fado. Sat: Belem + Time Out Market. Sun: Sintra day trip.",
        input_tokens=1610,
        output_tokens=430,
    ),
    _Phase(
        name="Budget",
        role="Local Budget Expert",
        thought="Price it up so there are no surprises.",
        tool="estimate_costs",
        tool_input="3 days, mid-range, 1 traveller",
        tool_result="stay 330, food 150, transit + Sintra 60 (EUR)",
        answer="Rough budget ~EUR 540: stay EUR 330, food EUR 150, transit and Sintra EUR 60.",
        input_tokens=980,
        output_tokens=210,
    ),
]


@dataclass
class _Usage:
    """Stand-in for CrewAI's UsageMetrics - the fields the App's metrics block
    and per-turn subtitle read."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_prompt_tokens: int = 0


@dataclass
class _Result:
    """Stand-in for a CrewOutput: the App reads ``.raw`` and ``.token_usage``."""

    raw: str
    token_usage: _Usage


@dataclass
class _TaskOut:
    """Stand-in for a crewai ``TaskOutput`` handed to a task callback. Real
    TaskOutput carries no per-task usage, so the App reads ``token_usage``
    defensively; the demo supplies it to drive the per-turn subtitle."""

    raw: str
    token_usage: _Usage


def _scripted_kickoff(crew: Crew) -> _Result:
    """Walk the crew's tasks without calling an LLM.

    For each task: stream a Thought + tool-call and an Answer through the crew's
    ``step_callback`` (both set by the App before this runs), pause briefly so
    the transition is visible, then fire the task's ``callback`` with a
    ``_TaskOut`` carrying that turn's token usage (which the App renders on the
    box subtitle). Returns a canned result whose ``token_usage`` is the sum of
    the turns, so the App's metrics block totals match.
    """
    agg_in = agg_out = 0
    for task, phase in zip(crew.tasks, _PHASES, strict=True):
        if crew.step_callback is not None:
            crew.step_callback(
                AgentAction(
                    thought=phase.thought,
                    tool=phase.tool,
                    tool_input=phase.tool_input,
                    text="",
                    result=phase.tool_result,
                )
            )
        time.sleep(0.6)
        if crew.step_callback is not None:
            crew.step_callback(
                AgentFinish(thought=phase.thought, output=phase.answer, text=phase.answer)
            )
        agg_in += phase.input_tokens
        agg_out += phase.output_tokens
        if task.callback is not None:
            task.callback(
                _TaskOut(
                    raw=phase.answer,
                    token_usage=_Usage(
                        prompt_tokens=phase.input_tokens,
                        completion_tokens=phase.output_tokens,
                        total_tokens=phase.input_tokens + phase.output_tokens,
                    ),
                )
            )
        time.sleep(0.4)

    return _Result(
        raw="Demo pipeline complete - a 3-day Lisbon plan, ~EUR 540.",
        token_usage=_Usage(
            prompt_tokens=agg_in,
            completion_tokens=agg_out,
            total_tokens=agg_in + agg_out,
            cached_prompt_tokens=512,
        ),
    )


def build_demo_crew() -> Crew:
    """Build the offline demo crew with a scripted ``kickoff``.

    A dummy ``OPENAI_API_KEY`` is set only if none is present so that
    constructing the agents never prompts; no request is ever made because
    ``kickoff`` is replaced before the App can call it. The LLM is built against
    a provider crewai bundles by default (openai) - so crewui, a generic UI,
    needs no per-provider extra to run its own demo - then relabelled to the
    model a reader would really run, purely for the box title.
    """
    os.environ.setdefault("OPENAI_API_KEY", "crewui-demo-no-network")

    agents = []
    tasks = []
    for phase in _PHASES:
        agent = Agent(
            role=phase.role,
            goal=f"The {phase.name.lower()} step of a trip-planning pipeline.",
            backstory=f"The {phase.role.lower()} on a small trip-planning crew.",
            llm="gpt-4o-mini",
            verbose=False,
        )
        # Display-only relabel: the demo never calls the LLM, so it is built
        # against a bundled provider (above) but shown as the model a crewui
        # user would really run. object.__setattr__ because LLM is a pydantic
        # model; pulling in a native Anthropic provider just to render a title
        # would force crewai[anthropic] onto a generic UI.
        object.__setattr__(agent.llm, "model", _DEMO_MODEL)
        agents.append(agent)
        tasks.append(
            Task(
                description=f"{phase.name} phase",
                expected_output="a short summary",
                agent=agent,
                name=phase.name,
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
        pipeline_name="crewui demo - Lisbon weekend",
        dry_run=dry_run,
        get_token_cost=lambda inp, out: (inp * 3 + out * 15) / 1_000_000,
    ).run()
