# Changelog

## v0.1.0

Initial release.

- `FeasibilityStudy` base class and engine: a locked, ordered state machine running ingest, linkage, the leakage guard, feature construction, the EPV gate, modeling, and inference. Modeling cannot start before the configuration is locked, and it cannot start before the EPV gate clears.
- Pre-registration lock via `feasibility lock`, with `--amend` preserving superseded locks under `locks/history/` by their original timestamp instead of overwriting them.
- EPV gate uses the minimum cross-validation training-fold event count by default (`epv_basis`), not the fold mean, so one underpowered fold cannot be averaged away. The basis is part of the locked configuration hash.
- Leakage guard: any predictor dated after the configured cutoff is rejected; availability exactly on the cutoff is allowed.
- Plugin registries for readers, composite strategies, and inference methods. Built-ins: `csv` and `parquet` readers, `sum-score` and `mean-score` composites, and `nadeau_bengio_holm` inference (Nadeau-Bengio corrected resampled t-test with Holm correction, applied within the exploratory contrast family only).
- CLI: `validate`, `lock`, `run`, `report`. Results are written next to the study file, named by the study and a hash of its locked configuration, and are not overwritten without `--force`.
- Synthetic toy study (`examples/toy_study.py`) runnable end to end as a template and as the test fixture for the full suite.
- 65 tests, including adversarial tests against bypassing the lock, stage order, and the EPV gate; leakage-guard edge cases (missing dates, boundary dates, unparseable dates); hand-checked numeric tests for Nadeau-Bengio and Holm; and a recorded mutation-testing run (`docs/mutation_testing.md`) confirming each of three structural invariants is actually caught by the suite if broken.
- Documentation: `docs/architecture.md`, `docs/methodology.md`, `docs/extending.md`, `docs/limitations.md`, including the limitation that the pre-registration lock is self-attested and the amendment history is a mitigation, not a proof.
