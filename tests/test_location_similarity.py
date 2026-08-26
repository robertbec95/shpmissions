"""Tests for the core Gower-distance + k-NN propagation pipeline in
`shopping_missions.location_similarity` - the method validated (v4) and
re-tested against 4 further attempts (v5-v8) throughout this project.
Uses small, self-contained synthetic fixtures rather than the real
6-store feature table, so these tests don't depend on any data file."""

import numpy as np
import pandas as pd
import pytest

from shopping_missions.location_similarity import propagate_profile, weighted_gower_distance

FEATURE_GROUPS = {"cat": "primary", "num_main": "main", "num_ignored": "ignored"}
CATEGORICAL_COLUMNS = {"cat"}


def make_stores(ignored_values):
    return pd.DataFrame(
        {
            "cat": ["X", "X", "Y", "Y"],
            "num_main": [1.0, 2.0, 10.0, 11.0],
            "num_ignored": ignored_values,
        },
        index=["A", "B", "C", "D"],
    )


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    X = make_stores([0.0, 1.0, 2.0, 3.0])
    D = weighted_gower_distance(
        X, feature_groups=FEATURE_GROUPS, group_weights={"primary": 1.0, "main": 1.0, "ignored": 1.0},
        categorical_columns=CATEGORICAL_COLUMNS,
    )
    assert np.allclose(np.diag(D.values), 0.0)
    assert np.allclose(D.values, D.values.T)


def test_zero_weight_group_has_no_effect_on_distance():
    """The whole reason weighted Gower is used over a flat/unweighted
    version in this project (see v3's regression, fixed by v4's
    weighting) - a group's weight must actually control its influence."""
    X_a = make_stores([0.0, 1.0, 2.0, 3.0])
    X_b = make_stores([100.0, -50.0, 999.0, 0.0])  # wildly different "ignored" column

    weights = {"primary": 1.0, "main": 1.0, "ignored": 0.0}
    D_a = weighted_gower_distance(X_a, FEATURE_GROUPS, weights, CATEGORICAL_COLUMNS)
    D_b = weighted_gower_distance(X_b, FEATURE_GROUPS, weights, CATEGORICAL_COLUMNS)

    assert np.allclose(D_a.values, D_b.values, atol=1e-9)


def test_missing_feature_group_raises():
    X = make_stores([0.0, 1.0, 2.0, 3.0])
    X["extra_col"] = [1, 2, 3, 4]
    with pytest.raises(ValueError, match="extra_col"):
        weighted_gower_distance(X, FEATURE_GROUPS, {"primary": 1.0, "main": 1.0, "ignored": 1.0}, CATEGORICAL_COLUMNS)


def test_categorical_columns_contribute_zero_distance_when_equal():
    X = pd.DataFrame({"cat": ["X", "X"]}, index=["A", "B"])
    D = weighted_gower_distance(X, {"cat": "primary"}, {"primary": 1.0}, {"cat"})
    assert D.loc["A", "B"] == pytest.approx(0.0)


def test_categorical_columns_contribute_nonzero_distance_when_different():
    X = pd.DataFrame({"cat": ["X", "Y"]}, index=["A", "B"])
    D = weighted_gower_distance(X, {"cat": "primary"}, {"primary": 1.0}, {"cat"})
    assert D.loc["A", "B"] > 0.0


# --- propagate_profile ---

DISTANCE = pd.DataFrame(
    [
        [0.0, 0.1, 0.5, 0.9],
        [0.1, 0.0, 0.4, 0.8],
        [0.5, 0.4, 0.0, 0.2],
        [0.9, 0.8, 0.2, 0.0],
    ],
    index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"],
)
PROFILES = {"A": [1.0, 0.0], "B": [0.0, 1.0], "C": [2.0, 2.0], "D": [4.0, 4.0]}


def test_propagate_picks_k_nearest_by_distance():
    # A's neighbors sorted by distance: B (0.1), C (0.5), D (0.9)
    pred = propagate_profile(DISTANCE, PROFILES, "A", k=2)
    expected = np.mean([PROFILES["B"], PROFILES["C"]], axis=0)
    assert np.allclose(pred, expected)


def test_propagate_excludes_target_by_default():
    # exclude defaults to {target} - A must never vote on its own prediction
    pred = propagate_profile(DISTANCE, PROFILES, "A", k=3)
    expected = np.mean([PROFILES["B"], PROFILES["C"], PROFILES["D"]], axis=0)
    assert np.allclose(pred, expected)


def test_propagate_respects_additional_excludes():
    pred = propagate_profile(DISTANCE, PROFILES, "A", k=1, exclude={"A", "B"})
    # with B excluded too, the nearest remaining is C
    assert np.allclose(pred, PROFILES["C"])


def test_propagate_raises_when_not_enough_candidates():
    with pytest.raises(ValueError, match="Only 3 candidate"):
        propagate_profile(DISTANCE, PROFILES, "A", k=5)
