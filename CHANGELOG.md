# Changelog

All notable changes to this project are documented here. The format follows
[Conventional Commits](https://www.conventionalcommits.org/), and releases are
cut with `cz bump`, which regenerates the entries below from the commit history.

## v0.1.0

Initial release. A Textual TUI for a sequential CrewAI pipeline, extracted into
a standalone package from the downstream project it was written for.

- `CrewAIPipelineTUI`: sidebar task tracker, agent-output stream, pipeline log,
  and token/cost summary for any sequential `crewai.Crew`.
- Host behaviour injected through `on_start` / `on_complete` / `get_token_cost`
  callbacks; the package never reaches up into the host app.
- Human review (`Task(human_input=True)`) routed to an input box instead of a
  blocking terminal `input()`.
- `crewui demo`: a fully offline scripted pipeline, so `pipx install crewui`
  yields something runnable without an API key.
