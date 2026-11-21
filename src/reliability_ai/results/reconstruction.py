import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

def build_reconstruction_dataframe(
    actual: np.ndarray,
    reconstructed: np.ndarray,
    end_times: list[pd.Timestamp],
    feature_names: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Build a timestamp-aligned reconstruction result DataFrame.

    Actual and reconstructed values from the final timestep of each sequence are inverse-transformed into original engineering units. 

    Parameters
    ----------
    actual:
        Scaled actual input sequences with shape:

        (
            number_of_sequences,
            sequence_length,
            number_of_features,
        )

    reconstructed:
        Scaled reconstructed sequences with the same shape as ``actual``.

    end_times:
        Timestamp corresponding to the final timestep of each sequence.

    feature_names:
        Ordered feature names corresponding to the final dimension of ``actual`` and ``reconstructed``.

        The feature order must be the same order used when fitting the scaler and training the TCN autoencoder.

    scaler:
        StandardScaler fitted using the training features.

    Returns
    -------
    pd.DataFrame
        DataFrame containing one row per sequence and columns for datetime and each actual and reconstructed feature. 
    """

    # ========================================================
    # Validate array dimensions
    # ========================================================

    if actual.ndim != 3:
        raise ValueError(
            "actual must have shape (number_of_sequences, sequence_length, number_of_features). "
            f"Received shape {actual.shape}."
        )

    if reconstructed.ndim != 3:
        raise ValueError(
            "reconstructed must have shape (number_of_sequences, sequence_length, number_of_features). "
            f"Received shape {reconstructed.shape}."
        )

    # ========================================================
    # Validate matching actual/reconstructed shapes
    # ========================================================

    if actual.shape != reconstructed.shape:
        raise ValueError(
            "actual and reconstructed must have identical shapes. "
            f"Received {actual.shape} and {reconstructed.shape}."
        )

    number_of_sequences = actual.shape[0]
    number_of_features = actual.shape[2]

    # ========================================================
    # Validate sequence-end timestamps
    # ========================================================

    if len(end_times) != number_of_sequences:
        raise ValueError(
            "The number of end_times must match the number of sequences. "
            f"Received {len(end_times)} end_times for {number_of_sequences} sequences."
        )

    # ========================================================
    # Validate feature names
    # ========================================================

    if len(feature_names) != number_of_features:
        raise ValueError(
            "The number of feature_names must match the feature dimension. "
            f"Received {len(feature_names)} feature names for {number_of_features} features."
        )

    # ========================================================
    # Handle empty sequence input
    # ========================================================

    if number_of_sequences == 0:
        columns = ["datetime"]

        for feature_name in feature_names:
            columns.extend(
                [
                    f"original__{feature_name}",
                    f"reconstructed__{feature_name}",
                ]
            )

        return pd.DataFrame(
            columns=columns
        )

    # ========================================================
    # Extract final timestep from each sequence
    # ========================================================

    actual_end_scaled = actual[
        :,
        -1,
        :,
    ]

    reconstructed_end_scaled = reconstructed[
        :,
        -1,
        :,
    ]

    # ========================================================
    # Convert back to original engineering units
    # ========================================================

    actual_end = scaler.inverse_transform(
        actual_end_scaled
    )

    reconstructed_end = scaler.inverse_transform(
        reconstructed_end_scaled
    )

    # ========================================================
    # Build timestamp column
    # ========================================================

    result = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                end_times
            )
        }
    )

    # ========================================================
    # Add paired actual and reconstructed feature columns
    # ========================================================

    for feature_index, feature_name in enumerate(
        feature_names
    ):
        result[
            f"original__{feature_name}"
        ] = actual_end[
            :,
            feature_index,
        ]

        result[
            f"reconstructed__{feature_name}"
        ] = reconstructed_end[
            :,
            feature_index,
        ]

    return result