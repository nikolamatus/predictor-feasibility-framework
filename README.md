# Predictive feasibility framework

A dataset-agnostic framework for pre-registered predictive feasibility studies.

The framework calls your code. Subclass `FeasibilityStudy`, implement the required hooks, and lock the configuration before any model is fit. Stage order, the leakage guard, and the events-per-variable gate are enforced by the engine.

The command-line interface is the primary way to run a study:

```text
pip install -e .
feasibility validate my_study.py
feasibility lock my_study.py
feasibility run my_study.py
feasibility report my_study/results/my_study.result.json
```

`lock` writes `locks/<study_id>.lock.json` next to `my_study.py`, wherever you invoke the command from. The same hash a second time prints `already locked, hash unchanged` and leaves the file alone. A changed config is refused until `feasibility lock my_study.py --amend`, which moves the old lock into `locks/history/` and writes a new one. `run` will not start without that lock. By default `run` writes `<study dir>/results/<study name>.result.json` next to the study file, not into the current working directory.

A real study must commit its lock file before `load_data` runs so the commit timestamp is the external witness of the pre-registration. The toy study under `examples/` is exempt: `examples/locks/` and `examples/results/` are gitignored. See `docs/architecture.md` and `examples/toy_study.py`.
