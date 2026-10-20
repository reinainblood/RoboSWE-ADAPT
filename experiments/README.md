# Verification experiments

`action_replay_audit.py` explores a separate physical audit of recorded native
actions. It does not rerun authored Python, so random sampling and clock-dependent
branches in a program do not resample during the audit.

The prototype accepts an explicitly supplied trusted runtime adapter. That
adapter must preserve the original predicate-evaluation schedule and provide
non-mutating cached observations and predicates. The prototype validates recorded
provenance and actions, recomputes physical outcomes, and writes a separate report.
It never modifies the existing code-replay verification or score.

The tests use mocked runtimes, including stochastic-program and stateful-predicate
cases. No live runtime adapter or command is included. These experiments are not
invoked by `roboswe run` and are not part of the published runtime package.

```sh
python -m pytest -q tests/test_action_replay_audit.py
```
