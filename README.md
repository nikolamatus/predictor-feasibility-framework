# Predictive feasibility framework

A dataset-agnostic framework for pre-registered predictive feasibility studies.

The framework calls your code. Subclass `FeasibilityStudy`, implement the required hooks, and lock the configuration before any model is fit. Stage order, the leakage guard, and the events-per-variable gate are enforced by the engine.

The command-line interface is the primary way to run a study:

```text
pip install -e .
feasibility validate my_study.py
feasibility run my_study.py
feasibility report result.json
```

`run` refuses to start until `lock_preregistration()` has written a lock file for the current configuration. See `docs/architecture.md` and `examples/toy_study.py`.
