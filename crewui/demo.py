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

The per-turn token subtitle is *real*: on a live run the App reads each agent's
``_token_process`` accumulator, which crewai's LLM callback ticks up as the task
runs. This demo makes no LLM call, so it ticks that same accumulator by hand
with canned counts - driving the identical code path a live run does, rather
than faking numbers onto the output.
"""

from __future__ import annotations

import os
import time
import types
from dataclasses import dataclass
from datetime import datetime
from typing import NamedTuple

from crewai import Agent, Crew, Process, Task
from crewai.agents.parser import AgentFinish
from crewai.events import (
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
    crewai_event_bus,
)
from crewai.events.types.llm_events import LLMThinkingChunkEvent

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
    cached_tokens: int


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
        cached_tokens=0,  # cold start - nothing cached yet, so this turn shows no cache rail
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
        cached_tokens=896,  # the research turn's context is now a cache hit
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
        cached_tokens=1408,  # research + itinerary context both cached by now
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
    """Stand-in for a crewai ``TaskOutput`` handed to a task callback. Like the
    real thing it carries no per-task usage - the App reads the turn's spend
    from the agent's accumulator, not from here - so ``raw`` is all it needs."""

    raw: str


def _scripted_kickoff(crew: Crew) -> _Result:
    """Walk the crew's tasks without calling an LLM.

    For each task: emit a tool call on CrewAI's event bus (which the App renders
    as a collapsed box), pause so the tool reads as running, emit its result,
    then stream the Answer through the crew's ``step_callback``. Finally fire the
    task's ``callback`` (which drives the box subtitle off the token accumulator
    ticked below). Returns a canned result whose ``token_usage`` is the sum of
    the turns, so the App's metrics block totals match.

    The tool events are the genuine article, emitted exactly as a live run's
    tool execution emits them - so the demo drives the real collapsible path,
    not a faked step. The brief pause between Started and Finished stands in for
    the tool actually running.
    """
    agg_in = agg_out = agg_cached = 0
    for task, phase in zip(crew.tasks, _PHASES, strict=True):
        # Thinking first: stream the phase's thought as reasoning chunks, exactly
        # as a thinking model's provider does, so the demo drives the real
        # collapsed-reasoning path. Split so the box visibly accumulates.
        words = phase.thought.split()
        half = len(words) // 2 or 1
        call_id = f"demo-{phase.tool}"
        crewai_event_bus.emit(
            crew, LLMThinkingChunkEvent(chunk=" ".join(words[:half]) + " ", call_id=call_id)
        )
        time.sleep(0.2)
        crewai_event_bus.emit(
            crew, LLMThinkingChunkEvent(chunk=" ".join(words[half:]), call_id=call_id)
        )
        started = datetime.now()
        crewai_event_bus.emit(
            crew, ToolUsageStartedEvent(tool_name=phase.tool, tool_args=phase.tool_input)
        )
        time.sleep(0.6)
        crewai_event_bus.emit(
            crew,
            ToolUsageFinishedEvent(
                tool_name=phase.tool,
                tool_args=phase.tool_input,
                output=phase.tool_result,
                started_at=started,
                finished_at=datetime.now(),
                from_cache=phase.cached_tokens > 0,
            ),
        )
        if crew.step_callback is not None:
            crew.step_callback(
                AgentFinish(thought=phase.thought, output=phase.answer, text=phase.answer)
            )
        # Tick the agent's real token accumulator, exactly as crewai's
        # TokenCalcHandler does on a live run - so the App's per-turn subtitle
        # is driven through its real path, not handed faked numbers.
        proc = task.agent._token_process
        proc.sum_prompt_tokens(phase.input_tokens)
        proc.sum_completion_tokens(phase.output_tokens)
        proc.sum_cached_prompt_tokens(phase.cached_tokens)
        agg_in += phase.input_tokens
        agg_out += phase.output_tokens
        agg_cached += phase.cached_tokens
        if task.callback is not None:
            task.callback(_TaskOut(raw=phase.answer))
        time.sleep(0.4)

    return _Result(
        raw="Demo pipeline complete - a 3-day Lisbon plan, ~EUR 540.",
        token_usage=_Usage(
            prompt_tokens=agg_in,
            completion_tokens=agg_out,
            total_tokens=agg_in + agg_out,
            cached_prompt_tokens=agg_cached,
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
