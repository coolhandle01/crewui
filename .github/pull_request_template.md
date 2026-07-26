## Summary

<!-- What does this PR do, and why? Link the issue it closes: "Closes #NN". -->

## Test plan

<!--
How you verified the change. Paste the relevant output from the "Before you
commit" stack in CONTRIBUTING.md:

    ruff check .
    mypy
    pylint crewui
    pytest --cov=crewui --cov-branch --cov-report=term-missing --cov-fail-under=95

New App code paths get a pilot test; new helpers get a unit test. Drive tests
against the fake offline crew in tests/conftest.py - never a real crewai.Crew.
-->

## Out of scope

<!-- What this PR deliberately does NOT do, and where that work is tracked (issue number, or "follow-up under #NN"). Write "none" if not applicable. -->

---

- [ ] Branch follows `<type>/<short-description>` and the commit follows Conventional Commits.
- [ ] `git diff origin/main --stat` shows only changes related to this PR.
- [ ] New or changed behaviour is covered; branch coverage stays at or above the 95% gate.
- [ ] No new runtime dependency (crewui is `crewai` + `textual`, and nothing else). If this touches the `crewai.core.providers.human_input` seam, the crewai pin ceiling was reconsidered.
