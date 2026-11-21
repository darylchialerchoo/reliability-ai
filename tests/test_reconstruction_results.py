import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler

from reliability_ai.results import (
    build_reconstruction_dataframe,
)


# ============================================================
# Test configuration
# ============================================================

FEATURE_NAMES = [
    "pressure",
    "temperature",
]

SEQUENCE_LENGTH = 4


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def engineering_values() -> np.ndarray:
    """
    Create known engineering-unit values for fitting
    the StandardScaler.
    """

    return np.array(
        [
            [10.0, 100.0],
            [20.0, 200.0],
            [30.0, 300.0],
            [40.0, 400.0],
            [50.0, 500.0],
            [60.0, 600.0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def scaler(
    engineering_values: np.ndarray,
) -> StandardScaler:
    """
    Create a fitted StandardScaler.
    """

    fitted_scaler = StandardScaler()

    fitted_scaler.fit(
        engineering_values
    )

    return fitted_scaler


def build_test_sequences(
    scaler: StandardScaler,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[pd.Timestamp],
]:
    """
    Build deterministic actual and reconstructed sequences.

    Values are first defined in engineering units and then
    transformed into the scaled model-input space.

    The reconstructed values intentionally differ from the
    actual values at the final timestep.
    """

    actual_engineering = np.array(
        [
            [
                [10.0, 100.0],
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
            ],
            [
                [20.0, 200.0],
                [30.0, 300.0],
                [40.0, 400.0],
                [50.0, 500.0],
            ],
        ],
        dtype=np.float64,
    )

    reconstructed_engineering = np.array(
        [
            [
                [11.0, 105.0],
                [19.0, 195.0],
                [31.0, 295.0],
                [38.0, 390.0],
            ],
            [
                [21.0, 205.0],
                [29.0, 295.0],
                [41.0, 395.0],
                [47.0, 480.0],
            ],
        ],
        dtype=np.float64,
    )

    number_of_sequences = (
        actual_engineering.shape[0]
    )

    number_of_features = (
        actual_engineering.shape[2]
    )

    actual_scaled = scaler.transform(
        actual_engineering.reshape(
            -1,
            number_of_features,
        )
    ).reshape(
        actual_engineering.shape
    )

    reconstructed_scaled = scaler.transform(
        reconstructed_engineering.reshape(
            -1,
            number_of_features,
        )
    ).reshape(
        reconstructed_engineering.shape
    )

    end_times = [
        pd.Timestamp(
            "2025-01-01 03:00:00"
        ),
        pd.Timestamp(
            "2025-01-01 04:00:00"
        ),
    ]

    assert len(
        end_times
    ) == number_of_sequences

    return (
        actual_scaled,
        reconstructed_scaled,
        end_times,
    )


# ============================================================
# Basic DataFrame tests
# ============================================================

def test_returns_dataframe(
    scaler: StandardScaler,
) -> None:
    """
    The result builder must return a pandas DataFrame.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )


def test_correct_number_of_rows(
    scaler: StandardScaler,
) -> None:
    """
    One sequence should produce one result row.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    assert len(
        result
    ) == actual.shape[0]


def test_correct_columns(
    scaler: StandardScaler,
) -> None:
    """
    Result columns must follow the reconstruction
    output schema.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    expected_columns = [
        "datetime",
        "original__pressure",
        "reconstructed__pressure",
        "original__temperature",
        "reconstructed__temperature",
    ]

    assert result.columns.tolist() == (
        expected_columns
    )


# ============================================================
# Timestamp-alignment tests
# ============================================================

def test_timestamps_are_preserved(
    scaler: StandardScaler,
) -> None:
    """
    Sequence-end timestamps must be preserved in order.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    expected_timestamps = pd.Series(
        pd.to_datetime(
            end_times
        ),
        name="datetime",
    )

    pd.testing.assert_series_equal(
        result["datetime"],
        expected_timestamps,
    )


def test_first_result_uses_first_sequence_end_timestamp(
    scaler: StandardScaler,
) -> None:
    """
    The first result must align with the final timestamp
    of the first sequence.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    assert (
        result.loc[
            0,
            "datetime",
        ]
        == end_times[0]
    )


# ============================================================
# Engineering-unit tests
# ============================================================

def test_actual_values_are_inverse_transformed(
    scaler: StandardScaler,
) -> None:
    """
    Actual final-timestep values must be returned in their
    original engineering units.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    # First sequence final timestep:
    #
    # pressure    = 40.0
    # temperature = 400.0

    assert result.loc[
        0,
        "original__pressure",
    ] == pytest.approx(
        40.0
    )

    assert result.loc[
        0,
        "original__temperature",
    ] == pytest.approx(
        400.0
    )

    # Second sequence final timestep:
    #
    # pressure    = 50.0
    # temperature = 500.0

    assert result.loc[
        1,
        "original__pressure",
    ] == pytest.approx(
        50.0
    )

    assert result.loc[
        1,
        "original__temperature",
    ] == pytest.approx(
        500.0
    )


def test_reconstructed_values_are_inverse_transformed(
    scaler: StandardScaler,
) -> None:
    """
    Reconstructed final-timestep values must also be
    returned in original engineering units.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    # First sequence reconstructed final timestep:
    #
    # pressure    = 38.0
    # temperature = 390.0

    assert result.loc[
        0,
        "reconstructed__pressure",
    ] == pytest.approx(
        38.0
    )

    assert result.loc[
        0,
        "reconstructed__temperature",
    ] == pytest.approx(
        390.0
    )

    # Second sequence reconstructed final timestep:
    #
    # pressure    = 47.0
    # temperature = 480.0

    assert result.loc[
        1,
        "reconstructed__pressure",
    ] == pytest.approx(
        47.0
    )

    assert result.loc[
        1,
        "reconstructed__temperature",
    ] == pytest.approx(
        480.0
    )


def test_only_final_timestep_is_used(
    scaler: StandardScaler,
) -> None:
    """
    The reconstruction result must represent only the
    final timestep of each sequence.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    first_sequence_values = (
        scaler.inverse_transform(
            actual[
                0,
                :,
                :,
            ]
        )
    )

    expected_final_pressure = (
        first_sequence_values[
            -1,
            0,
        ]
    )

    expected_final_temperature = (
        first_sequence_values[
            -1,
            1,
        ]
    )

    assert result.loc[
        0,
        "original__pressure",
    ] == pytest.approx(
        expected_final_pressure
    )

    assert result.loc[
        0,
        "original__temperature",
    ] == pytest.approx(
        expected_final_temperature
    )


# ============================================================
# Feature-order tests
# ============================================================

def test_feature_order_is_preserved(
    scaler: StandardScaler,
) -> None:
    """
    Feature ordering must remain consistent with the model
    input and scaler feature order.
    """

    (
        actual,
        reconstructed,
        end_times,
    ) = build_test_sequences(
        scaler
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=end_times,
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    assert result.columns.tolist() == [
        "datetime",
        "original__pressure",
        "reconstructed__pressure",
        "original__temperature",
        "reconstructed__temperature",
    ]


# ============================================================
# Empty-input tests
# ============================================================

def test_empty_input_returns_empty_dataframe_with_correct_columns(
    scaler: StandardScaler,
) -> None:
    """
    Empty sequence arrays should produce an empty
    DataFrame while preserving the expected schema.
    """

    actual = np.empty(
        (
            0,
            SEQUENCE_LENGTH,
            len(FEATURE_NAMES),
        ),
        dtype=np.float32,
    )

    reconstructed = np.empty_like(
        actual
    )

    result = build_reconstruction_dataframe(
        actual=actual,
        reconstructed=reconstructed,
        end_times=[],
        feature_names=FEATURE_NAMES,
        scaler=scaler,
    )

    assert result.empty

    assert result.columns.tolist() == [
        "datetime",
        "original__pressure",
        "reconstructed__pressure",
        "original__temperature",
        "reconstructed__temperature",
    ]


# ============================================================
# Validation tests
# ============================================================

def test_mismatched_actual_and_reconstructed_shapes_raise_error(
    scaler: StandardScaler,
) -> None:
    """
    Actual and reconstructed tensors must have identical
    shapes.
    """

    actual = np.zeros(
        (
            2,
            4,
            2,
        )
    )

    reconstructed = np.zeros(
        (
            2,
            4,
            3,
        )
    )

    end_times = [
        pd.Timestamp(
            "2025-01-01 03:00:00"
        ),
        pd.Timestamp(
            "2025-01-01 04:00:00"
        ),
    ]

    with pytest.raises(
        ValueError,
        match="identical shapes",
    ):
        build_reconstruction_dataframe(
            actual=actual,
            reconstructed=reconstructed,
            end_times=end_times,
            feature_names=FEATURE_NAMES,
            scaler=scaler,
        )


def test_actual_must_be_three_dimensional(
    scaler: StandardScaler,
) -> None:
    """
    Actual values must use the sequence tensor contract:
    (number_of_sequences, sequence_length, number_of_features).
    """

    actual = np.zeros(
        (
            4,
            2,
        )
    )

    reconstructed = actual.copy()

    with pytest.raises(
        ValueError,
        match="actual must have shape",
    ):
        build_reconstruction_dataframe(
            actual=actual,
            reconstructed=reconstructed,
            end_times=[],
            feature_names=FEATURE_NAMES,
            scaler=scaler,
        )


def test_reconstructed_must_be_three_dimensional(
    scaler: StandardScaler,
) -> None:
    """
    Reconstructed values must use the sequence tensor
    contract.
    """

    actual = np.zeros(
        (
            2,
            4,
            2,
        )
    )

    reconstructed = np.zeros(
        (
            4,
            2,
        )
    )

    end_times = [
        pd.Timestamp(
            "2025-01-01 03:00:00"
        ),
        pd.Timestamp(
            "2025-01-01 04:00:00"
        ),
    ]

    with pytest.raises(
        ValueError,
        match="reconstructed must have shape",
    ):
        build_reconstruction_dataframe(
            actual=actual,
            reconstructed=reconstructed,
            end_times=end_times,
            feature_names=FEATURE_NAMES,
            scaler=scaler,
        )


def test_wrong_number_of_end_times_raises_error(
    scaler: StandardScaler,
) -> None:
    """
    Every sequence must have exactly one sequence-end
    timestamp.
    """

    actual = np.zeros(
        (
            2,
            4,
            2,
        )
    )

    reconstructed = actual.copy()

    end_times = [
        pd.Timestamp(
            "2025-01-01 03:00:00"
        )
    ]

    with pytest.raises(
        ValueError,
        match="number of end_times",
    ):
        build_reconstruction_dataframe(
            actual=actual,
            reconstructed=reconstructed,
            end_times=end_times,
            feature_names=FEATURE_NAMES,
            scaler=scaler,
        )


def test_wrong_number_of_feature_names_raises_error(
    scaler: StandardScaler,
) -> None:
    """
    Feature names must match the final tensor dimension.
    """

    actual = np.zeros(
        (
            2,
            4,
            2,
        )
    )

    reconstructed = actual.copy()

    end_times = [
        pd.Timestamp(
            "2025-01-01 03:00:00"
        ),
        pd.Timestamp(
            "2025-01-01 04:00:00"
        ),
    ]

    feature_names = [
        "pressure",
    ]

    with pytest.raises(
        ValueError,
        match="number of feature_names",
    ):
        build_reconstruction_dataframe(
            actual=actual,
            reconstructed=reconstructed,
            end_times=end_times,
            feature_names=feature_names,
            scaler=scaler,
        )