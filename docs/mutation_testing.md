Mutation: EPV clears_floor always True
Failing tests: 5
  tests/test_engine.py::test_epv_failure_does_not_call_modeling
  tests/test_epv.py::test_mean_clears_but_minimum_fold_does_not
  tests/test_epv.py::test_min_basis_underpowered_blocks_modeling
  tests/test_epv.py::test_minimum_exactly_at_floor_clears_and_one_below_does_not
  tests/test_mutation_invariants.py::test_epv_clears_floor_uses_comparison_not_constant_true

Mutation: lock hash comparison disabled
Failing tests: 3
  tests/test_adversarial.py::test_run_rejects_config_edited_after_lock
  tests/test_mutation_invariants.py::test_lock_hash_mismatch_still_raises
  tests/test_state_machine.py::test_amended_config_does_not_match_the_old_lock

Mutation: cutoff comparison changed from > to >=
Failing tests: 7
  tests/test_cli.py::test_lock_then_run_succeeds
  tests/test_cli.py::test_run_from_another_cwd_uses_the_study_lock
  tests/test_cli.py::test_run_writes_result_next_to_the_study
  tests/test_cli.py::test_run_refuses_overwrite_without_force
  tests/test_cli.py::test_cli_run_and_report
  tests/test_leakage.py::test_availability_exactly_on_cutoff_is_allowed
  tests/test_mutation_invariants.py::test_cutoff_rejects_only_dates_after_cutoff