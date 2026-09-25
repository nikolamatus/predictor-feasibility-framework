# Examples

`toy_study.py` is a minimal `FeasibilityStudy` subclass that builds invented rows in memory. Use it to exercise the CLI:

```text
feasibility validate examples/toy_study.py
feasibility lock examples/toy_study.py
feasibility run examples/toy_study.py
feasibility report examples/results/toy_study.<hash8>.result.json
```

`examples/locks/` and `examples/results/` are local run artifacts. They are listed in `.gitignore` and are not part of the repository. Anyone who runs the demo regenerates them. Do not commit those directories.

A real study (outside `examples/`) must commit its lock file before `load_data` runs so the commit timestamp is the external witness of the pre-registration.
