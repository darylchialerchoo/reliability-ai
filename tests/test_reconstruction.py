import numpy as np
import pytest

from reliability_ai.scoring import (
    calculate_feature_mse,
    calculate_sequence_mse,
    calculate_squared_error,
    get_top_contributors,
)


def test_perfect_reconstruction_has_zero_error():
    actual = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    reconstructed = actual.copy()

    squared_error = calculate_squared_error(
        actual=actual,
        reconstructed=reconstructed,
    )

    assert np.all(
        squared_error == 0.0
    )


def test_squared_error_is_correct():
    actual = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    reconstructed = np.array(
        [
            [
                [0.0, 2.0],
                [3.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    expected = np.array(
        [
            [
                [1.0, 0.0],
                [0.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    result = calculate_squared_error(
        actual=actual,
        reconstructed=reconstructed,
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_sequence_mse_is_correct():
    actual = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    reconstructed = np.array(
        [
            [
                [0.0, 2.0],
                [3.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    result = calculate_sequence_mse(
        actual=actual,
        reconstructed=reconstructed,
    )

    # Squared errors:
    #
    # 1, 0
    # 0, 4
    #
    # Mean:
    #
    # (1 + 0 + 0 + 4) / 4
    # = 1.25

    assert result[0] == pytest.approx(
        1.25
    )


def test_feature_mse_is_correct():
    actual = np.array(
        [
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float32,
    )

    reconstructed = np.array(
        [
            [
                [0.0, 2.0],
                [3.0, 2.0],
            ]
        ],
        dtype=np.float32,
    )

    result = calculate_feature_mse(
        actual=actual,
        reconstructed=reconstructed,
    )

    # Feature 1:
    #
    # (1 + 0) / 2 = 0.5
    #
    # Feature 2:
    #
    # (0 + 4) / 2 = 2.0

    expected = np.array(
        [
            [0.5, 2.0]
        ]
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_top_contributor_is_correct():
    feature_errors = np.array(
        [
            [0.5, 2.0],
            [4.0, 1.0],
        ]
    )

    feature_names = [
        "feature_1",
        "feature_2",
    ]

    result = get_top_contributors(
        feature_errors=feature_errors,
        feature_names=feature_names,
    )

    assert result == [
        "feature_2",
        "feature_1",
    ]


def test_mismatched_shapes_raise_error():
    actual = np.zeros(
        (2, 48, 5)
    )

    reconstructed = np.zeros(
        (2, 48, 6)
    )

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        calculate_squared_error(
            actual=actual,
            reconstructed=reconstructed,
        )


def test_top_contributor_feature_count_mismatch_raises_error():
    feature_errors = np.zeros(
        (10, 3)
    )

    feature_names = [
        "feature_1",
        "feature_2",
    ]

    with pytest.raises(
        ValueError,
        match="number of feature names",
    ):
        get_top_contributors(
            feature_errors=feature_errors,
            feature_names=feature_names,
        )