# CLAUDE.md

Project-level rules for all agents (architect, coder, reviewer, tester,
security-auditor, debugger, api-integration, data-flow-checker) working in
this repo.

## Stop-and-report rule

If an agent finds a problem outside the module it was asked to touch, it
logs the problem in `BUILD_STATUS.md` (under "Known gaps outside the phase
tables" or the relevant phase's Notes column) but does not fix it itself.
The original requester decides what happens next.

This keeps changes scoped to what was actually asked for and keeps
`BUILD_STATUS.md` as the single place where cross-cutting issues surface,
instead of having them fixed silently as drive-by edits inside unrelated
work.

## Python interpreter

This repo lives on a Windows drive (`/mnt/j/...`) and its real, working
Python interpreter is **Windows `python.exe`**, not `python3` in WSL.
`python3` in this WSL environment has no packages installed (no pandas, no
pip, nothing) and will produce false "module not found" import failures
that look like broken code but aren't.

Any agent running Python — scripts, `pytest`, one-off checks — should use
`python.exe` (or `python.exe -m pytest`), not `python3` or `python`.
