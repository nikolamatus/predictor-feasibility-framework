# Predictive feasibility framework

A dataset-agnostic framework for pre-registered predictive feasibility studies.

The framework calls your code. Subclass `FeasibilityStudy`, implement the required hooks, and lock the configuration before any model is fit. Stage order, the leakage guard, and the events-per-variable gate are enforced by the engine.

The command-line interface is the primary way to run a study:

```text
pip install -e .
feasibility validate my_study.py
feasibility lock my_study.py
feasibility run my_study.py
feasibility report my_study/results/my_study.<hash8>.result.json
```

`lock` writes `locks/<study_id>.lock.json` next to `my_study.py`, wherever you invoke the command from. The same hash a second time prints `already locked, hash unchanged` and leaves the file alone. A changed config is refused until `feasibility lock my_study.py --amend`, which moves the old lock into `locks/history/` and writes a new one. `run` will not start without that lock. By default `run` writes `<study dir>/results/<study name>.<first 8 of config hash>.result.json` next to the study file, not into the current working directory, and refuses to overwrite that path unless `--force` is passed.

A real study must commit its lock file before `load_data` runs so the commit timestamp is the external witness of the pre-registration. The toy study under `examples/` is exempt: `examples/locks/` and `examples/results/` are gitignored. See `docs/architecture.md` and `examples/toy_study.py`.

## Quickstart: your own study

Copy `examples/toy_study.py` as a starting point and change three things.

1. **`define_config`** -- describe your outcome, cutoff date, id key, and predictor blocks as a `StudyConfig`. Every predictor needs `confirmed=True` and an `availability` date; the leakage guard rejects anything dated after the cutoff. Set `epv_floor` and `epv_basis` here too, since both are locked.
2. **`load_data`** -- return a `dict[str, pandas.DataFrame]` keyed by the source names your config expects. `path_map` is there if your sources are files, but nothing requires it: query a database, call an API, whatever gets you a DataFrame. See "Non-file data sources" in `docs/extending.md`.
3. **A unique `study_id`** on the config, since lock and result files are named from it.

Then run the same four commands against your file instead of the toy study:

```text
feasibility validate my_study.py
feasibility lock my_study.py
feasibility run my_study.py
feasibility report my_study/results/my_study.<hash8>.result.json
```

Commit the lock file (`locks/<study_id>.lock.json`) before `load_data` touches any data. That commit's timestamp is what makes the pre-registration verifiable later; see the "self-attested lock" note in `docs/limitations.md`.
