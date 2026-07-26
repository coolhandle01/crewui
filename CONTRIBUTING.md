# Contributing to crewui

crewui is a small, focused package: a Textual TUI for a sequential CrewAI
pipeline. The bar is that it stays small and stays honest about its one
dependency seam. These are the rules that keep it there.

## The line the package holds

**crewui reads a `Crew` and renders it. It does not reach up into the host.**

Everything application-specific — a run id, metrics, pricing, persistence — is
injected through the `on_start` / `on_complete` / `get_token_cost` callbacks.
If you find yourself importing anything from a host application, or adding a
third runtime dependency, stop: that logic belongs in the host, passed in as a
callback, not in crewui.

There are exactly **two** runtime dependencies — `crewai` and `textual` — and
that is a promise, not an accident. A presentation layer that drags in a
transitive dependency tree is a liability in someone else's app.

## The one seam that reaches past the public surface

Human review leans on `crewai.core.providers.human_input`, which is
semi-internal. It is isolated to a single factory (`_make_tui_human_input_provider`)
and a deferred import in `crewui/app.py`, on purpose: when crewai moves it, that
is the one place to fix. This is also why the crewai pin has a ceiling
(`<1.16`) — a minor bump is a review event for that seam, not an automatic
upgrade. See the README's version-pin section.

## Before you commit

Run the same stack CI runs, inside a `.venv` with `pip install -e ".[dev]"`:

```bash
ruff check .
mypy
pylint crewui
pytest --cov=crewui --cov-branch --cov-report=term-missing --cov-fail-under=95
```

Then `git diff origin/main --stat` to confirm only what you meant to change is
staged. Running the linters is part of the work, not a formality left for
review.

- **Branch** from current `main`, named `<type>/<short-description>` where
  `<type>` matches the commit type (`feat/`, `fix/`, `docs/`, `chore/`,
  `refactor/`). Do not work on `main`.
- **Commit** with Conventional Commits — `<type>(<scope>)?: <subject>`,
  lowercase imperative subject. CI checks this, and `cz bump` derives the
  version from it.
- **Never** force-push, `push --delete`, or `branch -D` a shared or PR branch
  without explicit authorisation. `--force-with-lease` is no exception.
- Never put session URLs in commit messages or PR bodies.

## Testing discipline

The pure helpers (`crewui/_helpers.py`) are unit-tested to full branch
coverage. The App layer is driven through Textual's `pilot` harness against the
fake offline crew in `tests/conftest.py` — never against a real `crewai.Crew`,
so the suite needs no API key and makes no network call. A new code path in the
App gets a pilot test; a new helper gets a unit test.

An assertion that still passes on empty output is not an assertion — check the
value, not just that nothing raised.

## Releasing

Versions are derived from commit messages, not chosen by hand — and the bump is
made by CI, not by you.

**Run the `release` workflow from the Actions tab.** That is the whole
procedure.

Do **not** run `cz bump` locally. It writes a version commit to your `main` and
tags it, so if `main` has moved on meanwhile that commit cannot be pushed
without a force-push — leaving a local `main` diverged from the remote and a tag
that is not an ancestor of either. The workflow computes the bump on the runner
from current `main` and pushes the commit and its tag together, so the two
cannot disagree. `cz` stays in the dev extra for `cz check` and for inspecting
what the next version would be (`cz bump --dry-run`).

Dispatching `release.yml` runs the full PR gate against `main` first — a failing
check stops the release while the tree is still untagged — then bumps and tags,
builds, runs `twine check`, smoke-tests the built wheel on 3.10 and 3.13 (both
entry points must agree), publishes to PyPI via Trusted Publishing, and cuts a
GitHub release. Nothing publishes on an ordinary push or pull request, and a
hand-pushed tag no longer triggers anything.

Do not cut the release through the GitHub UI — `release.yml` runs
`gh release create` itself, and a hand-made release makes that step fail after
PyPI has already published.
