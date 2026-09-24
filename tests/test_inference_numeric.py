"""Hand-checked numeric tests for Nadeau-Bengio and Holm. Synthetic only."""

from __future__ import annotations

import math

import pytest

from feasibility_framework.registries.inference import holm_adjust, nadeau_bengio_ttest


def test_nadeau_bengio_t_statistic_matches_hand_arithmetic():
    # Fixed fold differences (k = 4):
    #   d = [0.10, 0.00, -0.05, 0.15]
    # mean = (0.10 + 0.00 - 0.05 + 0.15) / 4 = 0.20 / 4 = 0.05
    # s^2 = sum((d_i - mean)^2) / (k - 1)
    #     = ((0.05)^2 + (-0.05)^2 + (-0.10)^2 + (0.10)^2) / 3
    #     = (0.0025 + 0.0025 + 0.01 + 0.01) / 3
    #     = 0.025 / 3
    # n_test/n_train = 1/3
    # Var_NB = (1/k + n_test/n_train) * s^2
    #        = (1/4 + 1/3) * (0.025/3)
    #        = (7/12) * (0.025/3)
    # t = 0.05 / sqrt(Var_NB) = 0.7171371656... → 0.7171 to 4 decimals
    deltas = [0.10, 0.00, -0.05, 0.15]
    ratio = 1.0 / 3.0
    result = nadeau_bengio_ttest(deltas, ratio)
    mean = 0.05
    s2 = 0.025 / 3.0
    var_nb = (1.0 / 4.0 + ratio) * s2
    expected_t = mean / math.sqrt(var_nb)
    assert result.n == 4
    assert result.mean == pytest.approx(mean)
    assert result.sample_variance == pytest.approx(s2)
    assert result.corrected_variance == pytest.approx(var_nb)
    assert result.t_statistic == pytest.approx(expected_t, abs=5e-5)
    assert round(result.t_statistic, 4) == 0.7171


def test_holm_adjusted_p_values_match_hand_arithmetic():
    # Raw p = [0.01, 0.04, 0.03], m = 3, ordered ascending: 0.01, 0.03, 0.04
    # rank 0 (smallest): min(1, (3-0)*0.01) = 0.03
    # rank 1:           min(1, (3-1)*0.03) = 0.06; running max = 0.06
    # rank 2:           min(1, (3-2)*0.04) = 0.04; running max = 0.06
    # Map back to input order:
    #   0.01 → 0.03
    #   0.04 → 0.06
    #   0.03 → 0.06
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])
