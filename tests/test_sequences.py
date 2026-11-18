import numpy as np
import pandas as pd
import pytest

from sklearn.preprocessing import StandardScaler

from reliability_ai.preprocessing import (
    make_sequences_from_valid_df,
)


TIME_COLUMN = "datetime"

FEATURE_COLUMNS = [
    "feature_1",
    "feature_2",
]

SEQUENCE_LENGTH = 4


def make_hourly_dataframe(
    periods: int = 8,
) -> pd.DataFrame:
    """
    Create deterministic hourly test data.
    """

    return pd.DataFrame(
        {
            TIME_COLUMN: pd.date_range(
                start="2025-01-01 00:00:00",
                periods=periods,
                freq="h",
            ),
            "feature_1": np.arange(
                periods,
                dtype=float,
            ),
            "feature_2": np.arange(
                100,
                100 + periods,
                dtype=float,
            ),
        }
    )


def make_fitted_scaler(
    df: pd.DataFrame,
) -> StandardScaler:
    """
    Fit the scaler using the configured feature order.
    """

    scaler = StandardScaler()

    scaler.fit(
        df[FEATURE_COLUMNS]
    )

    return scaler


def test_contiguous_hourly_data_produces_expected_number_of_sequences():
    """
    Eight hourly rows with a four-hour window should
    generate five overlapping sequences.

    Number of sequences:

        n_rows - sequence_length + 1

        8 - 4 + 1 = 5
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.shape == (
        5,
        4,
        2,
    )

    assert len(
        end_times
    ) == 5


def test_sequence_end_timestamp_is_last_timestamp_in_window():
    """
    The first four-hour sequence should be assigned
    to its final timestamp.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    scaler = make_fitted_scaler(
        df
    )

    _, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert end_times[0] == pd.Timestamp(
        "2025-01-01 03:00:00"
    )

    assert end_times[-1] == pd.Timestamp(
        "2025-01-01 07:00:00"
    )


def test_sequences_are_scaled():
    """
    Sequence values should contain the transformed
    feature values rather than the original values.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    scaler = make_fitted_scaler(
        df
    )

    expected_scaled = scaler.transform(
        df[FEATURE_COLUMNS]
    )

    sequences, _ = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    np.testing.assert_allclose(
        sequences[0],
        expected_scaled[0:4],
        rtol=1e-6,
        atol=1e-7,
    )


def test_output_dtype_is_float32():
    """
    Sequence tensors should be returned using float32.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, _ = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.dtype == np.float32


def test_missing_hour_prevents_sequence_crossing_gap():
    """
    A sequence must never cross a missing hourly timestamp.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    # Remove 03:00.
    df = df[
        df[TIME_COLUMN]
        != pd.Timestamp(
            "2025-01-01 03:00:00"
        )
    ].reset_index(
        drop=True
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    # Before the gap:
    #
    # 00, 01, 02
    #
    # Only three rows, so no length-4 sequence.
    #
    # After the gap:
    #
    # 04, 05, 06, 07
    #
    # Exactly one valid sequence.

    assert sequences.shape == (
        1,
        4,
        2,
    )

    assert end_times == [
        pd.Timestamp(
            "2025-01-01 07:00:00"
        )
    ]


def test_unsorted_rows_are_sorted_before_sequence_generation():
    """
    Input data does not need to already be chronologically sorted.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    df = df.sample(
        frac=1.0,
        random_state=42,
    ).reset_index(
        drop=True
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.shape == (
        5,
        4,
        2,
    )

    assert end_times[0] == pd.Timestamp(
        "2025-01-01 03:00:00"
    )

    assert end_times[-1] == pd.Timestamp(
        "2025-01-01 07:00:00"
    )


def test_duplicate_timestamp_keeps_first_row():
    """
    Duplicate hourly timestamps should retain the first
    occurrence, matching the HPC implementation.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    duplicate = df.iloc[
        [2]
    ].copy()

    duplicate[
        "feature_1"
    ] = 999.0

    df = pd.concat(
        [
            df,
            duplicate,
        ],
        ignore_index=True,
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.shape == (
        5,
        4,
        2,
    )

    assert len(
        end_times
    ) == 5


def test_empty_dataframe_returns_empty_sequence_array():
    """
    Empty input should return an empty sequence tensor
    with the expected dimensionality.
    """

    df = pd.DataFrame(
        columns=[
            TIME_COLUMN,
            *FEATURE_COLUMNS,
        ]
    )

    training_df = (
        make_hourly_dataframe(
            periods=8,
        )
    )

    scaler = make_fitted_scaler(
        training_df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.shape == (
        0,
        4,
        2,
    )

    assert sequences.dtype == np.float32

    assert end_times == []


def test_not_enough_rows_returns_empty_sequence_array():
    """
    Fewer rows than the configured sequence length
    cannot produce a sequence.
    """

    df = make_hourly_dataframe(
        periods=3,
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    assert sequences.shape == (
        0,
        4,
        2,
    )

    assert end_times == []


def test_invalid_timestamps_are_removed():
    """
    Invalid timestamps should be coerced to NaT and
    removed before sequence construction.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    # Convert the timestamp column to object so the test
    # can represent dirty/raw timestamp input containing
    # both valid timestamps and an invalid string.
    df[TIME_COLUMN] = df[
        TIME_COLUMN
    ].astype(object)

    df.loc[
        3,
        TIME_COLUMN,
    ] = "not-a-date"

    scaler = make_fitted_scaler(
        df
    )

    sequences, end_times = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    # The invalid 03:00 timestamp is converted to NaT
    # and removed. This creates a continuity gap:

    assert sequences.shape == (
        1,
        4,
        2,
    )

    assert end_times == [
        pd.Timestamp(
            "2025-01-01 07:00:00"
        )
    ]


def test_feature_order_is_preserved():
    """
    Feature ordering is part of the model input contract.
    """

    df = make_hourly_dataframe(
        periods=8,
    )

    scaler = make_fitted_scaler(
        df
    )

    sequences, _ = (
        make_sequences_from_valid_df(
            df_valid=df,
            time_col=TIME_COLUMN,
            feature_cols=FEATURE_COLUMNS,
            scaler=scaler,
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    expected_scaled = scaler.transform(
        df[FEATURE_COLUMNS]
    )

    np.testing.assert_allclose(
        sequences[0, :, 0],
        expected_scaled[0:4, 0],
        rtol=1e-6,
        atol=1e-7,
    )

    np.testing.assert_allclose(
        sequences[0, :, 1],
        expected_scaled[0:4, 1],
        rtol=1e-6,
        atol=1e-7,
    )