# crewui - AI Contributor Guide

**Read `CONTRIBUTING.md` first.** It carries the universal rules — the two-
dependency promise, the single crewai seam, the "Before you commit" CI parity
stack, Conventional Commits, and the force-push policy. This file is the
AI-contributor layer on top.

## The one thing to internalise

crewui **reads a `Crew` and renders it**, and does nothing else. Every time you
are tempted to import something from a host application, or to add a third
runtime dependency, or to teach the TUI what a "run id" or a "finding" is — the
answer is a callback the host passes in, not code in this package. `on_start`,
`on_complete`, and `get_token_cost` are the whole extension surface. Keep it
that way.

## The seam to be careful around

Human review reaches into `crewai.core.providers.human_input`, which is
semi-internal to crewai. It lives in exactly one place —
`_make_tui_human_input_provider` in `crewui/app.py`, behind a deferred import.
When crewai moves it, that is the one file to touch, and it is why the crewai
pin carries a `<1.16` ceiling: a minor bump is a review event for this seam, not
an automatic upgrade. Do not scatter crewai-internal imports anywhere else.

## Layout

| File | What it is |
|---|---|
| `crewui/app.py` | The Textual App: widgets, the worker-thread run, the human-review gate, the log handler. The only file that touches crewai internals. |
| `crewui/_helpers.py` | Pure functions (truncation, log routing, step formatting, sidebar layout, metrics block). No Textual, no threading — unit-tested to full branch coverage. |
| `crewui/demo.py` | A fully offline scripted crew so `crewui demo` runs with no API key. |
| `crewui/cli.py` / `crewui/__main__.py` | The `crewui` console script and `python -m crewui`; both route through `cli:main`. |
| `crewui/default.tcss` | The bundled dark theme, shipped as package data. |

## Testing

The App is driven through Textual's `pilot` harness against the **fake offline
crew** in `tests/conftest.py` — never a real `crewai.Crew`. That is deliberate:
the suite stays deterministic, needs no API key, and makes no network call. A
new App code path gets a pilot test; a new helper gets a unit test. The coverage
gate is branch coverage at 95%.
