import numpy as np
import pytest

from reliability_ai.detection import (
    apply_threshold,
    calculate_percentile_threshold,
    calculate_threshold_and_anomalies,
    validate_percentile,
    validate_scores,
)


# ============================================================
# Test data
# ============================================================

REFERENCE_SCORES = np.array(
    [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ],
    dtype=np.float64,
)


# ============================================================
# Score validation tests
# ============================================================

def test_valid_scores_pass_validation() -> None:
    """
    A finite one-dimensional score array should pass.
    """

    validate_scores(
        scores=REFERENCE_SCORES
    )


def test_scores_must_be_numpy_array() -> None:
    """
    Lists should not silently replace the documented
    NumPy score-array contract.
    """

    with pytest.raises(
        TypeError,
        match="NumPy array",
    ):
        validate_scores(
            scores=[
                1.0,
                2.0,
                3.0,
            ]
        )


def test_scores_must_be_one_dimensional() -> None:
    """
    Reconstruction scores should contain exactly one value
    per sequence.
    """

    scores = np.zeros(
        (
            5,
            2,
        )
    )

    with pytest.raises(
        ValueError,
        match="one-dimensional",
    ):
        validate_scores(
            scores=scores
        )


def test_empty_scores_raise_error() -> None:
    """
    A threshold cannot be calculated from an empty
    reference population.
    """

    scores = np.array(
        [],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="at least one value",
    ):
        validate_scores(
            scores=scores
        )


def test_nan_score_raises_error() -> None:
    """
    NaN values should never silently enter threshold
    calculation.
    """

    scores = np.array(
        [
            1.0,
            np.nan,
            3.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_scores(
            scores=scores
        )


def test_infinite_score_raises_error() -> None:
    """
    Infinite scores should also fail validation.
    """

    scores = np.array(
        [
            1.0,
            np.inf,
            3.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        validate_scores(
            scores=scores
        )


# ============================================================
# Percentile validation tests
# ============================================================

@pytest.mark.parametrize(
    "percentile",
    [
        0.0,
        50.0,
        95.0,
        99.0,
        99.5,
        100.0,
    ],
)
def test_valid_percentiles_pass_validation(
    percentile: float,
) -> None:
    """
    Percentiles from 0 through 100 inclusive are valid.
    """

    validate_percentile(
        percentile=percentile
    )


@pytest.mark.parametrize(
    "percentile",
    [
        -0.1,
        100.1,
        -5.0,
        101.0,
    ],
)
def test_percentile_outside_valid_range_raises_error(
    percentile: float,
) -> None:
    """
    Percentiles outside [0, 100] should fail.
    """

    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        validate_percentile(
            percentile=percentile
        )


@pytest.mark.parametrize(
    "percentile",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_nonfinite_percentile_raises_error(
    percentile: float,
) -> None:
    """
    NaN and infinite percentiles should fail.
    """

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        validate_percentile(
            percentile=percentile
        )


def test_nonnumeric_percentile_raises_error() -> None:
    """
    Percentile must be numeric.
    """

    with pytest.raises(
        TypeError,
        match="numeric",
    ):
        validate_percentile(
            percentile="99"
        )


# ============================================================
# Threshold calculation tests
# ============================================================

def test_calculate_100th_percentile_threshold() -> None:
    """
    P100 should equal the maximum reference score.
    """

    threshold = calculate_percentile_threshold(
        reference_scores=REFERENCE_SCORES,
        percentile=100.0,
    )

    assert threshold == pytest.approx(
        5.0
    )


def test_calculate_0th_percentile_threshold() -> None:
    """
    P0 should equal the minimum reference score.
    """

    threshold = calculate_percentile_threshold(
        reference_scores=REFERENCE_SCORES,
        percentile=0.0,
    )

    assert threshold == pytest.approx(
        1.0
    )


def test_calculate_50th_percentile_threshold() -> None:
    """
    P50 should reproduce the median for the test array.
    """

    threshold = calculate_percentile_threshold(
        reference_scores=REFERENCE_SCORES,
        percentile=50.0,
    )

    assert threshold == pytest.approx(
        3.0
    )


def test_default_threshold_uses_p99() -> None:
    """
    The default threshold percentile should be P99.
    """

    expected = float(
        np.percentile(
            REFERENCE_SCORES,
            99.0,
        )
    )

    threshold = calculate_percentile_threshold(
        reference_scores=REFERENCE_SCORES
    )

    assert threshold == pytest.approx(
        expected
    )


def test_threshold_returns_python_float() -> None:
    """
    Threshold should be convenient to serialize and store.
    """

    threshold = calculate_percentile_threshold(
        reference_scores=REFERENCE_SCORES,
        percentile=99.0,
    )

    assert isinstance(
        threshold,
        float,
    )


# ============================================================
# Threshold application tests
# ============================================================

def test_apply_threshold_uses_strict_greater_than_by_default() -> None:
    """
    With the default policy, a score exactly equal to the
    threshold is not anomalous.
    """

    scores = np.array(
        [
            0.5,
            1.0,
            1.5,
        ]
    )

    result = apply_threshold(
        scores=scores,
        threshold=1.0,
    )

    expected = np.array(
        [
            False,
            False,
            True,
        ]
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_apply_threshold_can_use_inclusive_comparison() -> None:
    """
    Inclusive thresholding should classify values equal to
    the threshold as anomalies.
    """

    scores = np.array(
        [
            0.5,
            1.0,
            1.5,
        ]
    )

    result = apply_threshold(
        scores=scores,
        threshold=1.0,
        inclusive=True,
    )

    expected = np.array(
        [
            False,
            True,
            True,
        ]
    )

    np.testing.assert_array_equal(
        result,
        expected,
    )


def test_apply_threshold_returns_boolean_array() -> None:
    """
    Anomaly classification should produce Boolean values.
    """

    result = apply_threshold(
        scores=REFERENCE_SCORES,
        threshold=3.0,
    )

    assert result.dtype == np.bool_

    assert result.shape == (
        len(REFERENCE_SCORES),
    )


def test_threshold_must_be_numeric() -> None:
    """
    String thresholds should fail clearly.
    """

    with pytest.raises(
        TypeError,
        match="threshold must be numeric",
    ):
        apply_threshold(
            scores=REFERENCE_SCORES,
            threshold="3.0",
        )


@pytest.mark.parametrize(
    "threshold",
    [
        np.nan,
        np.inf,
        -np.inf,
    ],
)
def test_threshold_must_be_finite(
    threshold: float,
) -> None:
    """
    NaN and infinite thresholds should fail.
    """

    with pytest.raises(
        ValueError,
        match="threshold must be finite",
    ):
        apply_threshold(
            scores=REFERENCE_SCORES,
            threshold=threshold,
        )


def test_inclusive_must_be_boolean() -> None:
    """
    Threshold comparison policy should be explicit.
    """

    with pytest.raises(
        TypeError,
        match="inclusive must be a Boolean",
    ):
        apply_threshold(
            scores=REFERENCE_SCORES,
            threshold=3.0,
            inclusive="yes",
        )


# ============================================================
# Combined threshold-and-classification tests
# ============================================================

def test_calculate_threshold_and_anomalies() -> None:
    """
    Convenience function should calculate the threshold
    from reference scores and classify target scores.
    """

    target_scores = np.array(
        [
            1.0,
            2.5,
            3.0,
            4.0,
        ]
    )

    threshold, anomalies = (
        calculate_threshold_and_anomalies(
            reference_scores=REFERENCE_SCORES,
            scores=target_scores,
            percentile=50.0,
        )
    )

    assert threshold == pytest.approx(
        3.0
    )

    expected = np.array(
        [
            False,
            False,
            False,
            True,
        ]
    )

    np.testing.assert_array_equal(
        anomalies,
        expected,
    )


def test_reference_scores_and_target_scores_can_differ() -> None:
    """
    The threshold population and evaluated population should
    be independent arrays.

    This protects the intended workflow:

        reference normal data
            -> threshold

        evaluation data
            -> anomaly classification
    """

    reference_scores = np.array(
        [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]
    )

    target_scores = np.array(
        [
            0.25,
            0.45,
            0.75,
        ]
    )

    threshold, anomalies = (
        calculate_threshold_and_anomalies(
            reference_scores=reference_scores,
            scores=target_scores,
            percentile=80.0,
        )
    )

    expected_threshold = float(
        np.percentile(
            reference_scores,
            80.0,
        )
    )

    assert threshold == pytest.approx(
        expected_threshold
    )

    expected_anomalies = (
        target_scores
        > expected_threshold
    )

    np.testing.assert_array_equal(
        anomalies,
        expected_anomalies,
    )