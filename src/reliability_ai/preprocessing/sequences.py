import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def make_sequences_from_valid_df(
    df_valid: pd.DataFrame,
    time_col: str,
    feature_cols: list[str],
    scaler: StandardScaler,
    sequence_length: int,
) -> tuple[np.ndarray, list[pd.Timestamp]]:
    """
    Convert valid rows with consecutive hourly timestamps into scaled contiguous hourly sequences. 

    Parameters
    ----------
    df_valid:
        DataFrame containing timestamps and feature values.

    time_col:
        Name of timestamp column.

    feature_cols:
        Ordered list of feature columns expected by scaler & TCN autoencoder.

    scaler:
        StandardScaler already fitted on training data.

    sequence_length:
        Number of consecutive hourly samples in each sequence.

    Returns
    -------
    tuple[np.ndarray, list[pd.Timestamp]]
        A tuple containing:

        - Sequences with shape
          (number_of_sequences, sequence_length, number_of_features)

        - Sequence-end timestamps
    """

    # ========================================================
    # Handle empty input
    # ========================================================

    if df_valid.empty:
        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    len(feature_cols),
                ),
                dtype=np.float32,
            ),
            [],
        )

    # ========================================================
    # Prepare timestamps
    # ========================================================

    work = df_valid.copy()

    work[time_col] = pd.to_datetime(
        work[time_col],
        errors="coerce",
    ).dt.floor("h")

    work = (
        work
        .dropna(subset=[time_col])
        .sort_values(time_col)
        .drop_duplicates(
            subset=[time_col],
            keep="first",
        )
        .reset_index(drop=True)
    )

    # ========================================================
    # Handle empty data after timestamp cleaning
    # ========================================================

    if work.empty:
        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    len(feature_cols),
                ),
                dtype=np.float32,
            ),
            [],
        )

    # ========================================================
    # Prepare feature values
    # ========================================================

    X = work[
        feature_cols
    ].copy()

    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # Preserve existing HPC behavior.
    if X.isna().any().any():
        X = (
            X
            .ffill()
            .bfill()
        )

    # ========================================================
    # Scale features
    # ========================================================

    X_scaled = scaler.transform(
        X
    )

    timestamps = work[
        time_col
    ].tolist()

    # ========================================================
    # Build contiguous hourly sequences
    # ========================================================

    sequences = []
    end_times = []

    for end_idx in range(
        sequence_length - 1,
        len(work),
    ):
        start_idx = (
            end_idx
            - sequence_length
            + 1
        )

        timestamp_window = timestamps[
            start_idx:end_idx + 1
        ]

        differences = (
            pd.Series(timestamp_window)
            .diff()
            .dropna()
        )

        # Require strict hourly continuity.
        if not (
            differences
            == pd.Timedelta(hours=1)
        ).all():
            continue

        sequence = X_scaled[
            start_idx:end_idx + 1
        ]

        sequences.append(
            sequence
        )

        end_times.append(
            timestamp_window[-1]
        )

    # ========================================================
    # Handle case where no valid sequences were produced
    # ========================================================

    if len(sequences) == 0:
        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    X_scaled.shape[1],
                ),
                dtype=np.float32,
            ),
            [],
        )

    # ========================================================
    # Return sequence tensor and sequence-end timestamps
    # ========================================================

    return (
        np.asarray(
            sequences,
            dtype=np.float32,
        ),
        end_times,
    )