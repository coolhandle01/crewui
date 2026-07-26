# crewui

A [Textual](https://textual.textualize.io/) terminal UI for a sequential
[CrewAI](https://github.com/crewAIInc/crewAI) pipeline. You build a `Crew`;
crewui renders it — a sidebar task tracker, a live agent-output stream, a
pipeline log, and a token/cost summary — and, when a task asks for human
review, routes the feedback prompt to an input box instead of a blocking
terminal `input()`.

crewui is a presentation layer and nothing more. It reads a `Crew` and touches
only two things beyond it: the step/task callback contract and, for human
review, one semi-internal crewai provider hook. Everything application-specific
— where a run id is recorded, how run metrics are computed and priced — is
injected through callbacks, so the package never reaches up into your app.

## Install

```bash
pipx install crewui        # the CLI + the demo, isolated
# or, as a library in your project:
pip install crewui
```

See it run without writing any code or setting an API key:

```bash
crewui demo                # drives a scripted three-phase pipeline in the TUI
crewui demo --dry-run      # render the layout without walking the pipeline
```

The demo is fully offline — no LLM call, no network — so it doubles as a smoke
test that the install works.

## Use it in your app

```python
from crewui import CrewAIPipelineTUI

CrewAIPipelineTUI(
    crew=build_my_crew(),          # any sequential crewai.Crew
    record_prefix="myapp",         # log records under this name render in the agent pane
    pipeline_name="My Pipeline",   # sidebar title
).run()
```

The sidebar reads each task's display name (`Task.name`) and agent role straight
off `crew.tasks`, so wiring the crew is the whole job — there is no separate task
map to maintain.

### Injecting host behaviour

Three optional callbacks carry everything crewui deliberately does not know:

| Callback | When it fires | Typical use |
|---|---|---|
| `on_start()` | worker thread, right before `kickoff()` | bind a run id, stamp the start time |
| `on_complete(result)` | right after `kickoff()` | persist run metrics |
| `get_token_cost(input_tokens, output_tokens)` | when rendering the summary | return the USD estimate to display |

A save failure in `on_complete` is swallowed and surfaced in the pipeline log —
persistence never takes the UI down.

### Theming

`CrewAIPipelineTUI` ships a usable dark theme and owns it as an absolute
`CSS_PATH`. A subclass ships its own look by setting its own `CSS_PATH`:

```python
class MyTUI(CrewAIPipelineTUI):
    CSS_PATH = "my_app.tcss"
```

### Human review

A `Task(human_input=True)` pauses the run for feedback. crewui opens the input
box, parks the worker thread until you submit, and feeds what you type back into
crewai's feedback loop (empty input — just Enter — means "accept"). The routing
leans on `crewai.core.providers.human_input`, a semi-internal API isolated to a
single factory in `crewui/app.py` so there is one place to fix if it moves
between crewai versions.

## The crewai version pin, and the fork

crewui declares `crewai>=1.15.6,<1.16`.

- **The floor (1.15.6)** is the first release where the agent executor's
  `ask_for_human_input` compatibility property is back. A 1.14.5 executor
  regression crashed `human_input=True` with
  `'AgentExecutor' object has no attribute 'ask_for_human_input'`; below the
  floor, the human-review feature does not work.
- **The ceiling (<1.16)** is deliberate, not lazy. The human-input routing is
  the one place crewui reaches past crewai's public surface, so a minor bump is
  a review event: re-check that seam, then move the ceiling.

There is a **second, separate** human-input bug that a stock PyPI crewai still
carries: the feedback prompt tells the operator to review "the Final Result
above", but crewai only renders that result when `verbose=True` — so under a TUI
with verbose off, the operator is asked to review output that was never printed.
The downstream project this library was extracted from pins a
[crewai fork](https://github.com/coolhandle01/crewai) that adds a
Result-for-Review panel so the referenced output is always shown. **crewui does
not pin that fork** — it works against stock crewai, and simply streams each
answer into the agent-output pane before the review gate opens, which is the
generic equivalent of what the fork's panel does. If you need the fork's exact
behaviour, pin it in *your* application, not here.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ruff check .
mypy
pylint crewui
pytest --cov=crewui --cov-branch --cov-report=term-missing --cov-fail-under=95
```

The pure helpers in `crewui/_helpers.py` are unit-tested; the App itself is
driven through Textual's `pilot` harness in `tests/test_app.py` — status
transitions, the dry-run preview, the metrics block, the error path, and the
human-review gate all run against a fake offline crew.

## Licence

MIT. See [LICENSE](LICENSE).
