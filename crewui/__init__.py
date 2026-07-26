"""crewui - a Textual TUI for any sequential CrewAI crew.

``CrewAIPipelineTUI`` is a Textual App that renders a sidebar task tracker, an
agent output log, and a pipeline log for a CrewAI ``Crew``. The host builds the
crew and hands it in; the TUI stays out of crew construction entirely.

The package depends only on ``crewai`` and ``textual``. Live metrics come
straight from CrewAI (token usage off ``kickoff()``'s result); everything
host-specific is delegated to injected callbacks, so nothing here reaches up
into the host application:

- ``on_start()`` - fired in the worker thread right before kickoff (e.g. to
  bind a run id and stamp the start time)
- ``on_complete(result)`` - fired right after kickoff (e.g. to persist run
  metrics)
- ``get_token_cost(input_tokens, output_tokens)`` - returns the USD estimate to
  display; cost is not a CrewAI metric, so the host supplies it

Human review (``Task(human_input=True)``) is routed to the TUI's input box
instead of a blocking terminal ``input()`` - see ``crewui.app``.

Typical usage::

    from crewui import CrewAIPipelineTUI

    CrewAIPipelineTUI(
        crew=build_my_crew(),
        record_prefix="myapp",
        pipeline_name="My Pipeline",
    ).run()

The class owns a default theme; a derived class ships its own look by setting
its own ``CSS_PATH``.
"""

from __future__ import annotations

from crewui.app import CrewAIPipelineTUI

__version__ = "0.1.1"

__all__ = ["CrewAIPipelineTUI", "__version__"]
